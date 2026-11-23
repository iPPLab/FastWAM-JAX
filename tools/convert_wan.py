#!/usr/bin/env python3
"""Convert the original Wan2.2-TI2V-5B DiT or VAE to this project's Orbax layout."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fastwam_jax import load_config
from tools import convert_weights as weights


def video_parameters(source: Path, config, *, check_only: bool) -> dict:
    from safetensors import safe_open

    shapes = weights.video_dit_param_shapes(config)
    expected = {
        weights.jax_to_torch_key(key, "video"): (key, shape) for key, shape in shapes.items()
    }
    index_file = source / "diffusion_pytorch_model.safetensors.index.json"
    toc = json.loads(index_file.read_text())["weight_map"] if index_file.is_file() else None
    if toc is None:
        shards = sorted(source.glob("diffusion_pytorch_model*.safetensors"))
    else:
        shards = [source / name for name in sorted(set(toc.values()))]
    if not shards:
        raise weights.ConversionError(f"no Wan DiT safetensors shards in {source}")

    parameters = {}
    files_by_key = {}
    for path in shards:
        with safe_open(path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in files_by_key or key not in expected:
                    raise weights.ConversionError(f"{path.name}: unexpected or duplicate tensor {key}")
                files_by_key[key] = path.name
                target, shape = expected[key]
                native_shape = tuple(shard.get_slice(key).get_shape())
                if native_shape != weights._torch_shape(target, shape):
                    raise weights.ConversionError(f"{key}: incompatible shape {native_shape}")
                if not check_only:
                    array = shard.get_tensor(key).float().numpy()
                    parameters[target] = np.ascontiguousarray(weights.from_torch_layout(target, array))
    if set(files_by_key) != set(expected):
        missing = sorted(set(expected) - set(files_by_key))
        raise weights.ConversionError(f"missing {len(missing)} native Wan video tensors: {missing[:8]}")
    if toc is not None and files_by_key != toc:
        raise weights.ConversionError("the DiT shard contents disagree with the safetensors index")
    print(f"Checked {len(expected)} Wan video tensor keys and shapes", flush=True)
    return parameters


def native_vae_key(key: str) -> str:
    *parts, leaf = key.split(".")
    source_leaf = "weight" if leaf == "kernel" else leaf
    if parts[0] in ("quant_conv", "post_quant_conv"):
        return f"{'conv1' if parts[0] == 'quant_conv' else 'conv2'}.{source_leaf}"

    tower, block, *tail = parts
    root = f"{tower}."
    if block in ("conv_in", "norm_out", "conv_out"):
        stems = {"conv_in": "conv1", "norm_out": "head.0", "conv_out": "head.2"}
        return f"{root}{stems[block]}.{source_leaf}"

    if block == "mid_block":
        group, index, *tail = tail
        if group == "attentions":
            return f"{root}middle.1.{'.'.join(tail)}.{source_leaf}"
        root += f"middle.{int(index) * 2}."
    else:
        stage, group, *tail = tail
        encoder = tower == "encoder"
        group_name = "downsamples" if encoder else "upsamples"
        root += f"{group_name}.{stage}.{group_name}."
        if group in ("downsampler", "upsampler"):
            index = 2 if encoder else 3
            stem = "resample.1" if tail[0] == "resample" else "time_conv"
            return f"{root}{index}.{stem}.{source_leaf}"
        index, *tail = tail
        root += f"{index}."

    residual = {
        "norm1": "residual.0", "conv1": "residual.2",
        "norm2": "residual.3", "conv2": "residual.6", "conv_shortcut": "shortcut",
    }
    return f"{root}{residual[tail[0]]}.{source_leaf}"


def vae_parameters(source: Path, *, check_only: bool) -> tuple[dict, dict]:
    import torch
    from flax import nnx

    from fastwam_jax.models.wan.vae import WAN22_VAE_GEOMETRY, build_vae_from_sidecar

    sidecar = asdict(WAN22_VAE_GEOMETRY)
    abstract = build_vae_from_sidecar(sidecar)
    shape_map = {
        weights._dotted(path): tuple(value.shape)
        for path, value in nnx.to_flat_state(nnx.state(abstract, nnx.Param))
    }
    native = torch.load(source, weights_only=True, map_location="cpu", mmap=True)
    keys = {key: native_vae_key(key) for key in shape_map}
    if len(set(keys.values())) != len(keys):
        raise weights.ConversionError("VAE key mapping must be one-to-one")
    weights.compare_param_shapes(
        {key: () for key in keys.values()}, {key: () for key in native}, what="native Wan VAE keys",
    )
    parameters = {}
    for target, shape in shape_map.items():
        tensor = native[keys[target]]
        if target.endswith(".gamma"):
            tensor = tensor.reshape(-1)
        elif target.endswith(".kernel"):
            tensor = tensor.permute(*range(2, tensor.ndim), 1, 0)
        if tuple(tensor.shape) != shape:
            raise weights.ConversionError(f"{keys[target]}: converted shape {tensor.shape} != {shape}")
        if not check_only:
            parameters[target] = np.ascontiguousarray(tensor.float().numpy())
    print(f"Checked {len(keys)} Wan VAE tensor keys and shapes", flush=True)
    return parameters, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("video", "vae"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", default="training/full.local.example.yaml")
    args = parser.parse_args()
    check_only = args.output is None
    if args.component == "video":
        cfg = load_config(args.config)
        parameters = video_parameters(args.source, cfg.model.video_dit_config, check_only=check_only)
        sidecar = None
    else:
        parameters, sidecar = vae_parameters(args.source, check_only=check_only)
    if args.output is not None:
        weights.save_params(parameters, str(args.output))
        if sidecar is not None:
            (args.output / "vae_config.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"Saved {len(parameters)} parameters to {args.output}")


if __name__ == "__main__":
    main()
