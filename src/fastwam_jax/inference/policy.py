"""Prepare observations and run the cached Fast-WAM action policy.

The policy encodes RGB observations with the Wan VAE, normalizes joint state,
and returns denoised actions as absolute joint targets. The simulator adapter
composes the camera views and executes the action chunk.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

__all__ = ["FastWAMPolicy", "PolicyError"]

# `deploy_policy.py` reads these from `configs/sim_robotwin.yaml`; the `replan_steps: 8`
# in `fastwam_policy/deploy_policy.yml` is a stale default the sim config overrides.
# They live here as constants (not defaults spread over three call sites) so the banner,
# the server's guard and the client adapter can all quote the same numbers.
OFFICIAL_ACTION_HORIZON = 32
OFFICIAL_REPLAN_STEPS = 24
OFFICIAL_NUM_INFERENCE_STEPS = 10
OFFICIAL_SEED = 42  # config.py immutable runtime defaults, reused verbatim
OFFICIAL_CANVAS_HW = (384, 320)
NOISE_MODES = ("fixed", "resample")


class PolicyError(ValueError):
    """An observation or configuration that cannot be a correct FastWAM replan."""


def _encode_frame(vae, tensor: jax.Array) -> jax.Array:
    """The official `_encode_input_image_latents_tensor` + `WanVideoVAE38.encode`.

    `normalize=True` is not optional: upstream folds `(mu - latents_mean) / latents_std`
    into `encode` via `self.scale` (wan_video_vae.py:983-1008), so a raw posterior mean
    would be off by a per-channel affine the DiT has never seen.
    """
    return vae.encode_frame(tensor, normalize=True, channels_first=True)



class FastWAMPolicy:
    """One replan: canvas + qpos + instruction embedding -> [32, 14] absolute qpos.

    Stateless across calls except for the noise draw, so a server can hold one instance
    and serve several episodes as long as `reset()` is called between them.

    Args:
        model: a loaded `FastWAM`.
        vae: anything exposing `encode_frame(tensor, normalize, channels_first)`; in
            production `models/wan/vae.py::WanVAE38`.
        stats: `data/normalize.py::OfficialStats`, parsed from the official
            `dataset_stats.json`. The `action` and `proprio` members are DIFFERENT
            constant sets and are used in opposite directions; see `describe()`.
        action_horizon / num_inference_steps / sigma_shift: the official values.
        canvas_hw: the composed canvas the client is contracted to send. A mismatch
            raises rather than resizing — a silent resize here would mean the server and
            the client disagree about the picture and nothing would ever say so.
        context_len: 128 text tokens (the proprio token is appended by the model, making
            129); `text_dim` defaults to the model's own `text_dim`.
        noise_mode: "fixed" (official property) or "resample".
        noise_seed: the fixed draw's key, and the base for "resample".
        action_dtype: the dtype the denoising loop runs in. Defaults to the action
            expert's compute dtype, matching upstream's `latents_action.to(torch_dtype)`.
    """

    def __init__(
        self,
        *,
        model,
        vae,
        stats,
        action_horizon: int = OFFICIAL_ACTION_HORIZON,
        num_inference_steps: int = OFFICIAL_NUM_INFERENCE_STEPS,
        sigma_shift: float | None = None,
        canvas_hw: tuple[int, int] = OFFICIAL_CANVAS_HW,
        context_len: int = 128,
        text_dim: int | None = None,
        noise_mode: str = "fixed",
        noise_seed: int = OFFICIAL_SEED,
        action_dtype: jnp.dtype | None = None,
        device: jax.Device | None = None,
    ):
        if noise_mode not in NOISE_MODES:
            raise PolicyError(f"noise_mode must be one of {NOISE_MODES}, got {noise_mode!r}")
        if action_horizon <= 0 or num_inference_steps <= 0:
            raise PolicyError(
                f"action_horizon and num_inference_steps must be positive, got "
                f"{action_horizon}, {num_inference_steps}"
            )
        action_dim = model.config.action_dit_config.action_dim
        if stats.action.dim != action_dim:
            raise PolicyError(
                f"the action stats are {stats.action.dim}-D but the model predicts {action_dim}-D"
            )
        if stats.proprio.dim != model.config.proprio_dim:
            raise PolicyError(
                f"the state stats are {stats.proprio.dim}-D but proprio_dim is "
                f"{model.config.proprio_dim}"
            )

        self.device = jax.devices()[0] if device is None else device
        self.model = model
        self.vae = vae
        self.stats = stats
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(model.config.proprio_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        self.context_len = int(context_len)
        self.text_dim = int(
            text_dim if text_dim is not None else model.config.video_dit_config.text_dim
        )
        self.noise_mode = noise_mode
        self.noise_seed = int(noise_seed)
        self.action_dtype = action_dtype if action_dtype is not None else model.mot.action.dtype
        self.replans = 0
        self.episodes = 0

        # The fixed draw. Made once, here, so "fixed" really means fixed across the whole
        # process and not merely within an episode — which is the property the official
        # gets by rebuilding a seeded generator on every call.
        self._base_key = jax.device_put(jax.random.key(self.noise_seed), self.device)
        self.noise = self._draw_noise(self._base_key)

        # Split-once / merge-inside is the pattern the TinyWAM server proved out: it keeps
        # `nnx.split` off the per-replan path, which matters at 1,651 parameter leaves.
        graphdef, state = nnx.split(model)
        steps, shift = self.num_inference_steps, self.sigma_shift

        @jax.jit
        def denoise(model_state, latents, noise, context, context_mask, proprio):
            module = nnx.merge(graphdef, model_state)
            return module.infer_action(
                first_frame_latents=latents,
                noise_action=noise,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                num_inference_steps=steps,
                sigma_shift=shift,
            )

        self._model_state = state
        self._denoise = denoise
        # The VAE encoder is jitted only when it really is an nnx module; the tests inject
        # a plain-python stub, and a `nnx.jit` around that would fail on the split.
        self._encode = nnx.jit(_encode_frame) if isinstance(vae, nnx.Module) else _encode_frame

    # ------------------------------------------------------------------ #
    # episode lifecycle
    # ------------------------------------------------------------------ #
    def _draw_noise(self, key: jax.Array) -> jax.Array:
        """[1, Sa, action_dim]. Drawn in fp32 then cast, matching upstream's
        `torch.randn(..., dtype=torch.float32).to(self.torch_dtype)`."""
        noise = jax.random.normal(key, (1, self.action_horizon, self.action_dim), jnp.float32)
        return noise.astype(self.action_dtype)

    def reset(self) -> None:
        """New episode. In "fixed" mode nothing moves — which is the point. In "resample"
        mode the stream is re-derived from the episode index so a rerun is reproducible."""
        self.episodes += 1
        self.replans = 0
        if self.noise_mode == "resample":
            self.noise = self._draw_noise(jax.random.fold_in(self._base_key, self.episodes))

    # ------------------------------------------------------------------ #
    # the conventions, each callable on its own so a test can pin it
    # ------------------------------------------------------------------ #
    def canvas_to_tensor(self, canvas: np.ndarray) -> jax.Array:
        """uint8 [H, W, 3] RGB -> float32 [1, 3, H, W] in [-1, 1] (deploy_policy.py:229-233).

        uint8 in, not float: the official resizes and concatenates in uint8 and only then
        applies `* (2.0/255.0) - 1.0`, so taking float here would quietly be a different
        quantisation of the same picture.
        """
        canvas = np.asarray(canvas)
        if canvas.dtype != np.uint8:
            raise PolicyError(
                f"canvas must be uint8 RGB as the client composed it, got {canvas.dtype}"
            )
        if canvas.shape != (*self.canvas_hw, 3):
            raise PolicyError(
                f"canvas must be {(*self.canvas_hw, 3)} (H, W, 3); got {canvas.shape}. The "
                "client composes it: head (w=320,h=256) over left|right wrists (w=160,h=128)."
            )
        tensor = jax.device_put(canvas, self.device).astype(jnp.float32) * (2.0 / 255.0) - 1.0
        return jnp.transpose(tensor, (2, 0, 1))[None]

    def normalize_proprio(self, qpos: np.ndarray) -> np.ndarray:
        """Raw measured qpos -> z-score on the STATE stats, clamped to +-5.

        The clamp is forward-only and it is the *state* constants, not the action ones —
        `global_mean[0]` is -0.2251 for state and -0.2436 for action, so a swap survives
        every shape check and only shows up as a robot that does not work.
        """
        qpos = np.asarray(qpos, np.float32)
        if qpos.shape != (self.proprio_dim,):
            raise PolicyError(f"qpos must be [{self.proprio_dim}], got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise PolicyError("qpos contains non-finite values")
        return self.stats.proprio.forward(qpos)

    def denormalize_action(self, chunk: np.ndarray) -> np.ndarray:
        """Predicted chunk -> absolute qpos on the ACTION stats. No clamp (normalizer.py:131)."""
        chunk = np.asarray(chunk, np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
            raise PolicyError(f"chunk must be [T, {self.action_dim}], got {chunk.shape}")
        return self.stats.action.backward(chunk)

    # ------------------------------------------------------------------ #
    # the replan
    # ------------------------------------------------------------------ #
    def infer_action(
        self,
        canvas: np.ndarray,
        qpos: np.ndarray,
        context: np.ndarray,
        context_mask: np.ndarray,
    ) -> np.ndarray:
        """One replan.

        Args:
            canvas: uint8 [H, W, 3] RGB, already composed by the client.
            qpos: float [proprio_dim] — the CURRENT
                measured state at replan time (not the previous step's).
            context: float [128, 4096] — UMT5 embedding of
                `DEFAULT_PROMPT.format(task=instruction)`, padding rows already zeroed.
            context_mask: bool [128], all True by the zero-then-unmask convention.

        Returns: float32 [action_horizon, action_dim] absolute targets.
        """
        context = np.asarray(context, np.float32)
        context_mask = np.asarray(context_mask)
        if context.shape != (self.context_len, self.text_dim):
            raise PolicyError(
                f"context must be [{self.context_len}, {self.text_dim}], got {context.shape}"
            )
        if context_mask.shape != (self.context_len,):
            raise PolicyError(
                f"context_mask must be [{self.context_len}], got {context_mask.shape}"
            )
        if not np.all(context_mask):
            raise PolicyError(
                "context_mask must be all-True: FastWAM zeroes the padded embedding rows and "
                "then declares every position visible (fastwam.py:214-216). A False entry means "
                "the embedding was built some other way."
            )
        if not np.isfinite(context).all():
            raise PolicyError("context contains non-finite values")

        proprio = self.normalize_proprio(qpos)
        latents = self._encode(self.vae, self.canvas_to_tensor(canvas))
        if self.noise_mode == "resample":
            self.noise = self._draw_noise(
                jax.random.fold_in(self._base_key, (self.episodes << 20) + self.replans + 1)
            )
        normalized = self._denoise(
            self._model_state,
            latents,
            self.noise,
            jax.device_put(context[None], self.device),
            jax.device_put(context_mask[None].astype(bool), self.device),
            jax.device_put(proprio[None], self.device),
        )
        self.replans += 1
        return self.denormalize_action(np.asarray(normalized[0], np.float32))

    # ------------------------------------------------------------------ #
    # say it out loud
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """Report the active observation and action conventions."""
        shift = (
            self.sigma_shift
            if self.sigma_shift is not None
            else self.model.config.action_scheduler.infer_shift
        )
        return "\n".join(
            [
                f"canvas={self.canvas_hw[0]}x{self.canvas_hw[1]} RGB uint8, "
                "channel_order=RGB, flip=none, rotation=none",
                f"horizon={self.action_horizon}  replan={OFFICIAL_REPLAN_STEPS}  "
                f"denoise={self.num_inference_steps}  shift={shift}"
                f"{'' if self.sigma_shift is not None else ' (from config)'}",
                f"action=absolute qpos (no anchor row, no row-0 skip, no delta integration), "
                f"dim={self.action_dim}",
                "proprio=z-score(state stats)+clamp(+-5) forward; "
                "action=un-z-score(action stats) NO clamp",
                f"noise={self.noise_mode} (seed {self.noise_seed}); "
                f"cfg=off; kv_cache=ON; text=UMT5 L{self.context_len} zero-pad + all-True mask",
                f"stats: {self.stats.describe()}",
                "latent=normalize_latents applied once",
            ]
        )
