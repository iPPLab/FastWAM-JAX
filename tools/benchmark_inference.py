#!/usr/bin/env python3
"""Time a single RoboTwin action request with precomputed text conditioning."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TORCH_MODES = ("eager", "default", "reduce-overhead", "max-autotune")
JAX_MODES = ("jit", "jit-no-autotune")


def input_request():
    rng = np.random.default_rng(42)
    canvas = rng.integers(0, 256, size=(384, 320, 3), dtype=np.uint8)
    state = np.zeros(14, np.float32)
    context = np.zeros((128, 4096), np.float32)
    context[:24] = rng.standard_normal((24, 4096), dtype=np.float32)
    return canvas, state, context, np.ones(128, dtype=bool)


def jax_runner(args, request):
    if args.mode == "jit-no-autotune":
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
    import jax

    from fastwam_jax import load_config
    from fastwam_jax.inference.load import build_policy

    if jax.default_backend() != "gpu":
        raise RuntimeError("the JAX benchmark requires a CUDA device")
    device = jax.devices("gpu")[0]
    policy = build_policy(
        ckpt_root=str(args.checkpoint), vae_ckpt=str(args.vae), stats_path=str(args.stats),
        config=load_config(args.config), device=device, num_inference_steps=args.steps,
        sigma_shift=5.0, noise_seed=42,
    )

    def infer():
        return policy.infer_action(*request)

    def memory():
        stats = device.memory_stats() or {}
        return stats.get("peak_bytes_in_use")

    return infer, memory, {
        "version": jax.__version__, "device": device.device_kind,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def torch_runner(args, request):
    import torch
    import yaml

    from fastwam_jax.data.normalize import load_official_stats

    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream / "src"))
    from fastwam.models.wan22.fastwam import FastWAM  # noqa: E402

    config = yaml.safe_load((upstream / "configs/model/fastwam.yaml").read_text())
    video = config["video_dit_config"]
    action = config["action_dit_config"]
    for expert in (video, action):
        expert.update(action_dim=14, use_gradient_checkpointing=False)
    model = FastWAM.from_wan22_pretrained(
        video_dit_config=video, action_dit_config=action, proprio_dim=14,
        skip_dit_load_from_pretrain=True, load_text_encoder=False,
        redirect_common_files=False, device="cuda", torch_dtype=torch.bfloat16,
    )
    model.load_checkpoint(str(args.checkpoint))
    model.eval()
    if args.mode != "eager":
        model._prefill_video_cache_compiled = torch.compile(
            model.mot.prefill_video_cache_tensor, mode=args.mode, fullgraph=True,
        )
        model._denoise_action_with_video_cache_compiled = torch.compile(
            model._denoise_action_with_video_cache, mode=args.mode, fullgraph=True,
        )

    stats = load_official_stats(args.stats.read_bytes(), source=str(args.stats))
    canvas, state, context, mask = request

    def infer():
        with torch.no_grad():
            image = torch.from_numpy(canvas).to("cuda", torch.float32).permute(2, 0, 1)
            image = image * (2.0 / 255.0) - 1.0
            result = model.infer_action(
                prompt=None, input_image=image, action_horizon=32,
                proprio=torch.from_numpy(stats.proprio.forward(state)),
                context=torch.from_numpy(context), context_mask=torch.from_numpy(mask),
                num_inference_steps=args.steps, sigma_shift=5.0, seed=42, rand_device="cpu",
                compile_action_infer=args.mode != "eager",
            )
            torch.cuda.synchronize()
        return stats.action.backward(result["action"].numpy())

    torch.cuda.synchronize()
    return infer, torch.cuda.max_memory_allocated, {
        "version": torch.__version__, "device": torch.cuda.get_device_name(),
        "upstream": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
    }


def timed(infer):
    started = time.perf_counter()
    action = np.asarray(infer())
    elapsed_ms = (time.perf_counter() - started) * 1000
    if action.shape != (32, 14) or not np.isfinite(action).all():
        raise ValueError(f"invalid RoboTwin action: shape {action.shape}, dtype {action.dtype}")
    return elapsed_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("jax", "torch"))
    parser.add_argument("--mode", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--config", default="training/full.local.example.yaml")
    parser.add_argument("--vae", type=Path, help="JAX VAE Orbax directory")
    parser.add_argument("--upstream", type=Path, help="upstream FastWAM checkout for the PyTorch baseline")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("steps and trials must be positive, and warmup must be nonnegative")
    if args.backend == "jax" and (args.mode not in JAX_MODES or args.vae is None):
        parser.error(f"JAX requires --vae and a mode from {JAX_MODES}")
    if args.backend == "torch" and (args.mode not in TORCH_MODES or args.upstream is None):
        parser.error(f"PyTorch requires --upstream and a mode from {TORCH_MODES}")

    request = input_request()
    started = time.perf_counter()
    runner = jax_runner if args.backend == "jax" else torch_runner
    infer, memory, metadata = runner(args, request)
    load_ms = (time.perf_counter() - started) * 1000
    print(f"Loaded {args.backend} {args.mode} on {metadata['device']}", file=sys.stderr, flush=True)
    first_ms = timed(infer)
    print(f"First request: {first_ms:.1f} ms", file=sys.stderr, flush=True)
    for _ in range(args.warmup):
        timed(infer)
    times = [timed(infer) for _ in range(args.trials)]
    p50, p95 = np.percentile(times, [50, 95])
    result = {
        **metadata,
        "backend": args.backend, "mode": args.mode, "precision": "bfloat16",
        "input": "synthetic RGB 384x320, precomputed conditioning, batch 1",
        "steps": args.steps, "warmup": args.warmup, "trials": args.trials,
        "load_ms": load_ms, "first_request_ms": first_ms,
        "p50_ms": float(p50), "p95_ms": float(p95),
        "allocator_peak_bytes": memory(), "request_ms": times,
    }
    payload = json.dumps(result, indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
