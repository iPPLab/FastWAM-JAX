"""Format Fast-WAM prompts and encode them with UMT5.

The policy uses zero-padded embeddings and an all-true attention mask. The
cache stores only unpadded rows. The encoder imports PyTorch lazily.
"""

import hashlib
import html
import json
import os
import re

import numpy as np

__all__ = [
    "DEFAULT_PROMPT",
    "TextError",
    "TextEmbeddingCache",
    "Umt5Encoder",
    "build_prompt",
    "clean_text",
    "prompt_key",
]

# robot_video_dataset.py:23. Duplicated from `data/text.py` on purpose: that module is the
# training-side loader and imports the `.twp` stack, which has no business being pulled
# into an inference process. A test pins the two strings equal.
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

DEFAULT_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_MAX_LEN = 128  # specs/fastwam.yaml:tokenizer_max_len
DEFAULT_TEXT_DIM = 4096
DEFAULT_DTYPE = "float32"  # see the module docstring: bf16 is 400x slower on this CPU
STORE_DTYPE = np.float16
CACHE_VERSION = 1


class TextError(ValueError):
    """A prompt or cache entry that cannot be a correct FastWAM context."""


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """`whitespace_clean(basic_clean(text))` (wan_video_text_encoder.py:272-282).

    ftfy is optional here and absent from every venv on this box. For the ASCII English
    RoboTwin instructions `ftfy.fix_text` is the identity, so its absence changes nothing
    — but it is applied when available rather than dropped, because "we removed a step
    the reference has" is not a thing to discover from a score.
    """
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(instruction: str) -> str:
    """Instruction -> the template the model was conditioned on, cleaned.

    `_infer_action_chunk` formats first and the tokenizer cleans second
    (deploy_policy.py:245 then HuggingfaceTokenizer.__call__), so the order is
    format-then-clean and not the other way round.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise TextError(
            "instruction must be a non-empty string. RoboTwin draws it per episode from the "
            "'unseen' pool; an empty one means the client never called set_task and the whole "
            "episode would run on the wrong text."
        )
    return clean_text(DEFAULT_PROMPT.format(task=instruction))


def prompt_key(prompt: str, *, repo: str, max_len: int) -> str:
    """sha256 over (cache version, repo, max_len, prompt).

    Keying on the prompt alone would let a cache built at max_len 512, or with a
    different encoder, serve a run silently — the arrays would even have the right
    shape after padding.
    """
    payload = f"{CACHE_VERSION}\x00{repo}\x00{int(max_len)}\x00{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# the cache (numpy only — the model server's side)
# --------------------------------------------------------------------------- #
class TextEmbeddingCache:
    """`sha256(prompt) -> [max_len, 4096] float32` on disk, plus an in-process dict.

    Entries are `.npz` holding the trimmed real rows and a json meta blob; writes are
    tmp+rename so a half-written file is never readable as a hit (the same rule the eval
    harness uses for result files).
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        repo: str = DEFAULT_REPO,
        max_len: int = DEFAULT_MAX_LEN,
        text_dim: int = DEFAULT_TEXT_DIM,
    ):
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        self.repo = repo
        self.max_len = int(max_len)
        self.text_dim = int(text_dim)
        self.hits = 0
        self.misses = 0
        self._memory: dict[str, np.ndarray] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def path_for(self, prompt: str) -> str:
        return os.path.join(self.cache_dir, f"{prompt_key(prompt, repo=self.repo, max_len=self.max_len)}.npz")

    def _inflate(self, rows: np.ndarray, seq_len: int) -> np.ndarray:
        """Trimmed real rows -> the padded [max_len, text_dim] float32 context."""
        out = np.zeros((self.max_len, self.text_dim), np.float32)
        out[:seq_len] = rows[:seq_len].astype(np.float32)
        return out

    def get(self, prompt: str) -> np.ndarray | None:
        """The context for `prompt`, or None on a miss. Never raises on a bad file —
        a corrupt entry is treated as a miss and overwritten, because a cache that can
        wedge a campaign is worse than no cache."""
        cached = self._memory.get(prompt)
        if cached is not None:
            self.hits += 1
            return cached
        path = self.path_for(prompt)
        try:
            with np.load(path, allow_pickle=False) as payload:
                meta = json.loads(str(payload["meta"]))
                rows = payload["rows"]
        except Exception:
            self.misses += 1
            return None
        if (
            meta.get("prompt") != prompt
            or meta.get("repo") != self.repo
            or int(meta.get("max_len", -1)) != self.max_len
            or rows.ndim != 2
            or rows.shape[1] != self.text_dim
        ):
            self.misses += 1
            return None
        context = self._inflate(rows, int(meta["seq_len"]))
        self._memory[prompt] = context
        self.hits += 1
        return context

    def put(self, prompt: str, rows: np.ndarray, seq_len: int) -> np.ndarray:
        """Store the trimmed real rows and return the inflated context."""
        rows = np.asarray(rows)
        seq_len = int(seq_len)
        if rows.ndim != 2 or rows.shape[1] != self.text_dim:
            raise TextError(f"rows must be [seq_len, {self.text_dim}], got {rows.shape}")
        if not 0 < seq_len <= self.max_len or seq_len > rows.shape[0]:
            raise TextError(f"seq_len {seq_len} is not in (0, min({self.max_len}, {rows.shape[0]})]")
        if not np.isfinite(np.asarray(rows[:seq_len], np.float32)).all():
            raise TextError(f"the embedding for {prompt!r} contains non-finite values")

        meta = {
            "prompt": prompt,
            "repo": self.repo,
            "max_len": self.max_len,
            "seq_len": seq_len,
            "version": CACHE_VERSION,
        }
        path = self.path_for(prompt)
        tmp = f"{path}.tmp.{os.getpid()}"
        np.savez(
            tmp,
            rows=rows[:seq_len].astype(STORE_DTYPE),
            meta=np.asarray(json.dumps(meta)),
        )
        os.replace(f"{tmp}.npz" if os.path.exists(f"{tmp}.npz") else tmp, path)
        context = self._inflate(rows, seq_len)
        self._memory[prompt] = context
        return context

    def mask(self) -> np.ndarray:
        """The context mask that always accompanies a context: all True (see the module
        docstring). A method rather than a constant so the length can never drift from
        the embedding it is paired with."""
        return np.ones((self.max_len,), bool)

    def stats(self) -> str:
        return f"text cache {self.cache_dir}: {self.hits} hits / {self.misses} misses"


# --------------------------------------------------------------------------- #
# the encoder (torch — the sidecar's side)
# --------------------------------------------------------------------------- #
class Umt5Encoder:
    """UMT5-XXL encoder and tokenizer from the Wan Diffusers release."""

    def __init__(
        self,
        *,
        repo: str = DEFAULT_REPO,
        max_len: int = DEFAULT_MAX_LEN,
        dtype: str = DEFAULT_DTYPE,
        device: str = "cpu",
    ):
        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel

        if dtype not in ("float32", "bfloat16", "float16"):
            raise TextError(f"dtype must be float32/bfloat16/float16, got {dtype!r}")
        self.repo = repo
        self.max_len = int(max_len)
        self.dtype = dtype
        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
        self.model = (
            UMT5EncoderModel.from_pretrained(
                repo, subfolder="text_encoder", torch_dtype=getattr(torch, dtype)
            )
            .to(device)
            .eval()
        )
        self.text_dim = int(self.model.config.d_model)

    def encode(self, prompts: list[str]) -> list[tuple[np.ndarray, int]]:
        """-> per prompt `(rows [seq_len, 4096] float32, seq_len)`.

        Returns the TRIMMED rows: everything past `seq_len` is zero by the convention, so
        the caller's `TextEmbeddingCache.put` stores exactly what carries information.
        """
        torch = self._torch
        if not prompts:
            return []
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long().tolist()
        with torch.no_grad():
            hidden = self.model(ids, mask).last_hidden_state.to(torch.float32).cpu().numpy()
        out = []
        for index, seq_len in enumerate(seq_lens):
            if seq_len <= 0:
                raise TextError(f"prompt {prompts[index]!r} tokenised to zero real tokens")
            if seq_len >= self.max_len:
                # Truncation is silent in `tokenizer(truncation=True)` and would drop the
                # tail of the instruction. RoboTwin's are ~19 tokens (p99 42, max 53 over
                # the 23,552-prompt training corpus), so hitting 128 means something else
                # is wrong with the text.
                raise TextError(
                    f"prompt {prompts[index]!r} filled all {self.max_len} token slots — it was "
                    "truncated; the instruction is not what this recipe expects"
                )
            out.append((hidden[index, :seq_len].astype(np.float32), int(seq_len)))
        return out

    def encode_one(self, prompt: str) -> tuple[np.ndarray, int]:
        return self.encode([prompt])[0]

    def describe(self) -> str:
        return (
            f"UMT5 encoder: {self.repo}#text_encoder d_model={self.text_dim} "
            f"max_len={self.max_len} dtype={self.dtype} device={self.device} "
            f"clean=whitespace add_special_tokens=True zero-pad+all-True-mask"
        )
