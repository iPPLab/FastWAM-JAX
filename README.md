# FastWAM-JAX

FastWAM-JAX is a JAX/Flax NNX implementation of the `uncond` variant of
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM). It preserves the upstream
joint video and action training objective and predicts actions from the current
observation and language instruction.

The released checkpoint was trained on Google Cloud TPU. We have also verified
that the implementation can train on NVIDIA GPUs, but have not performed a full
GPU training run.

[Installation](#installation) · [Inference](#inference) · [License](#license)

## Installation

Run all commands from the repository root with **Python 3.12** and **`uv`**.
Use the `cuda` extra on NVIDIA GPUs, or replace it with `tpu` for Cloud TPU.

```bash
uv sync --extra cuda
uv tool install huggingface-hub
```

Define the model and dataset sources, then download the frozen VAE and RoboTwin
dataset statistics. The inference and training instructions below use these
assets and reuse the same shell variables.

```bash
WAN_REPO=ipplab/Wan2.2-TI2V-5B-Orbax
WAN_REV=c83dfe62ff6432dbc6090ade7d3b38ac05733118
DATASET=yuanty/robotwin2.0-fastwam
DATA_REV=aac262c35d02cc71b2f6ef670bd65fd9f2bb2547

hf download "$WAN_REPO" --revision "$WAN_REV" \
  --include 'vae/**' --local-dir checkpoints/wan
hf download "$DATASET" dataset_stats.json --repo-type dataset \
  --revision "$DATA_REV" --local-dir data/robotwin
```

## Inference

### Download the checkpoint

The [RoboTwin BF16 checkpoint](https://huggingface.co/ipplab/FastWAM-JAX-RoboTwin-Uncond/tree/f749afdce5bf14dc8a0d2c9c107ef0c20db3ca8e)
contains the trained video expert, action expert, and proprioception encoder.
It uses the frozen VAE from the
[Wan2.2 Orbax release](https://huggingface.co/ipplab/Wan2.2-TI2V-5B-Orbax/tree/c83dfe62ff6432dbc6090ade7d3b38ac05733118),
downloaded during installation.

```bash
hf download ipplab/FastWAM-JAX-RoboTwin-Uncond \
  --revision f749afdce5bf14dc8a0d2c9c107ef0c20db3ca8e \
  --include 'params_bf16_step0029365/**' --local-dir checkpoints/robotwin-trained
```

### Run the policy

Prepare the following inputs before calling `infer_action`:

| Input          | Description                            |
| -------------- | -------------------------------------- |
| `canvas`       | Composed RGB observation               |
| `qpos`         | Current joint state                    |
| `context`      | Zero-padded UMT5 instruction embedding |
| `context_mask` | All-true Boolean mask                  |

Use `build_prompt` and `Umt5Encoder` from `fastwam_jax.inference.text` to prepare
the text context; these utilities require the `convert` extra.

```python
import jax
from fastwam_jax import load_config
from fastwam_jax.inference import build_policy

policy = build_policy(
    config=load_config("configs/training/full.local.example.yaml"),
    ckpt_root="checkpoints/robotwin-trained/params_bf16_step0029365",
    vae_ckpt="checkpoints/wan/vae",
    stats_path="data/robotwin/dataset_stats.json",
    device=jax.devices()[0],
)
actions = policy.infer_action(canvas, qpos, context, context_mask)
```

`infer_action` returns **absolute joint targets**.

## License

FastWAM-JAX's own code is available under the [MIT License](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party terms.

