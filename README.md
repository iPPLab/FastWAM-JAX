# FastWAM-JAX

FastWAM-JAX is a JAX/Flax NNX implementation of the `uncond` variant of
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM). It preserves the upstream
joint video and action training objective and predicts actions from the current
observation and language instruction.

The released checkpoint was trained on Google Cloud TPU. We have also verified
that the implementation can train on NVIDIA GPUs, but have not performed a full
GPU training run.

[Results](#results) · [Installation](#installation) · [Inference](#inference) ·
[Training](#training) · [Citation](#citation)

## Results

### RoboTwin evaluation

We evaluated the converted official checkpoint and the JAX-trained checkpoint
through the RoboTwin policy adapter across all 50 tasks. Both checkpoints used
25 matched scene seeds per task in each setting. In the randomized setting,
paired evaluations shared the same instruction and initial state.

For reference,
[Table 1 of the Fast-WAM paper](https://arxiv.org/html/2603.16666v1#S4)
reports results from 100 trials per task and setting.

| Checkpoint                           |                Clean |           Randomized |
| ------------------------------------ | -------------------: | -------------------: |
| Fast-WAM paper                       |               91.88% |               91.78% |
| Official checkpoint converted to JAX | 91.92% (1,149/1,250) | 90.40% (1,130/1,250) |
| FastWAM-JAX, step 29,365             | 92.24% (1,153/1,250) | 91.04% (1,138/1,250) |

### Training performance

We trained the released checkpoint for five epochs on a Google Cloud TPU
v6e-64 slice, using 64 chips across 16 worker VMs and a global batch size of
1,024.

Across 26,938 steady-state updates, the mean update time was **1.392 s**, measured
from batch loading through the optimizer step. At this rate, 29,365 updates take
approximately **11 h 21 min** of training.

### Inference latency

We measured latency on a single RTX 5090, timing 30 requests per mode after an
initial request and three warm-up runs. JAX 0.11.0 and PyTorch 2.7.1 used the same
upstream BF16 weights, a fixed synthetic RGB observation, precomputed text, and
ten denoising steps. Measurements include VAE encoding and the transfer of
predicted actions back to the host.

| Runtime                                    | Median (ms) | p95 (ms) |
| ------------------------------------------ | ----------: | -------: |
| JAX JIT                                    |       74.00 |    76.10 |
| PyTorch `torch.compile`, `max-autotune`    |       74.70 |    76.93 |
| PyTorch `torch.compile`, `reduce-overhead` |       79.52 |    79.64 |
| PyTorch `torch.compile`, default           |      123.56 |   124.17 |
| PyTorch eager                              |      318.21 |   321.04 |
| JAX JIT, autotuning disabled               |      406.92 |   408.80 |

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

## Training

To train on RoboTwin, prepare the episodes and instruction embeddings, then
initialize the action expert from the Wan2.2 video weights. The commands below
reuse the model and dataset variables defined in [Installation](#installation).

### Prepare training inputs

Download and unpack the RoboTwin episodes, then convert them to the training
format:

```bash
hf download "$DATASET" --include 'robotwin2.0.tar.gz.part-*' --repo-type dataset \
  --revision "$DATA_REV" --local-dir data/robotwin
cat data/robotwin/robotwin2.0.tar.gz.part-* | tar -xzf - -C data/robotwin

uv run --extra dataset python tools/prepare_robotwin.py \
  --source data/robotwin/robotwin2.0 --output data/robotwin/twp
```

The packer stores each episode's images, actions, and joint states in a single
`TWP3` file. TPU workers reuse fetched episodes across training windows to reduce
GCS requests and repeated transfers. CUDA workers read the same files from disk.

Collect the instructions and precompute their text embeddings:

```bash
uv run --extra convert python tools/prepare_text_sidetable.py collect \
  --l3 data/robotwin/twp --out data/robotwin/text
uv run --extra convert python tools/prepare_text_sidetable.py encode \
  --dir data/robotwin/text --device cpu
```

Download the Wan2.2 video weights and use them to initialize the action expert:

```bash
hf download "$WAN_REPO" --revision "$WAN_REV" \
  --include 'video_dit/**' --local-dir checkpoints/wan
uv run python tools/convert_weights.py build-action-init \
  checkpoints/wan/video_dit checkpoints/wan/action_init \
  --config configs/training/full.local.example.yaml
```

### Train on Cloud TPU

The released checkpoint used an earlier pixel-packed dataset. The recipe below
uses TWP3 with RGB images and aligned action and state rows.

Upload the training inputs to a bucket in the TPU region. Copy the example
recipe, then edit its `paths` and `tpu` sections to match your setup. The launcher
uses the active `gcloud` project.

```bash
cp configs/training/full.example.yaml configs/training/full.tpu.yaml
# Set paths and TPU settings in full.tpu.yaml.
uv run bash scripts/launch_training.sh configs/training/full.tpu.yaml robotwin-full
```

### Train on NVIDIA GPUs

We have verified that the implementation can train on NVIDIA GPUs, but have not
performed a full GPU training run. The released checkpoint was trained on TPU.

Set `training.micro_batch_size` to a multiple of the GPU count and a divisor of
the global batch size, then launch training with the local recipe:

```bash
JAX_PLATFORMS=cuda,cpu uv run --extra cuda python -m fastwam_jax.training.train \
  configs/training/full.local.example.yaml tpu.run_name=robotwin-full
```

## Citation

If you find our repo helpful, please consider citing FastWAM-JAX and the original Fast-WAM paper.

```bibtex
@misc{fastwamjax2026,
  author={{FastWAM-JAX Contributors}},
  title={{FastWAM-JAX}},
  year={2026},
  howpublished={\url{https://github.com/iPPLab/FastWAM-JAX}}
}

@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```

## License

FastWAM-JAX's own code is available under the [MIT License](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party terms.

## Acknowledgments

This project was supported in part with Cloud TPUs from Google's TPU Research Cloud (TRC).
