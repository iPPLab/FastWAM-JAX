"""Immutable evaluation contract written next to every new training run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastwam_jax.config import build_config, load_raw

EVAL_SPEC_SCHEMA = 1
EVAL_SPEC_NAME = "EVAL_SPEC.json"


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_eval_spec(
    training_config: str | Path,
    *,
    run_name: str,
    code_package_md5: str,
    code_package_uri: str,
    overrides=(),
    configs_dir: str | Path | None = None,
) -> dict:
    implicit_overrides = [
        f"tpu.run_name={json.dumps(str(run_name))}",
        "tpu.resume=true",
    ]
    raw = load_raw(
        training_config,
        configs_dir=configs_dir,
        overrides=[*implicit_overrides, *overrides],
    )
    cfg = build_config(raw)
    spec = {
        "schema_version": EVAL_SPEC_SCHEMA,
        "run_name": str(run_name),
        "source": {
            "code_package_md5": str(code_package_md5),
            "code_package_uri": str(code_package_uri),
            "training_config": str(training_config),
            "overrides": list(overrides),
            "implicit_overrides": implicit_overrides,
        },
        # This is the fully composed, pre-dataclass mapping. It can be passed straight
        # back to build_config without depending on a mutable YAML path.
        "resolved_config": raw,
        "assets": {
            "checkpoint_layout": "training_params_bf16_v1",
            "vae": cfg.model.weights.vae_orbax,
            "stats": cfg.data.l3.stats_path,
            "stats_sha256": cfg.data.l3.stats_sha256,
            "training_data": cfg.data.l3.prefix,
            "text_embeddings": cfg.data.l3.text_dir,
        },
        "embodiment": {
            "action_dim": cfg.model.action_dit_config.action_dim,
            "proprio_dim": cfg.model.proprio_dim,
            "num_output_cameras": cfg.data.train.processor.num_output_cameras,
            "video_size": list(cfg.data.train.video_size),
            "canvas_layout": cfg.data.train.concat_multi_camera,
        },
        "data_contract": {
            "text_mode": cfg.data.l3.text_mode,
            "action_shift": cfg.data.l3.action_shift,
        },
    }
    spec["contract_sha256"] = _canonical_sha256(spec)
    return spec


def validate_eval_spec(spec: dict) -> dict:
    if spec.get("schema_version") != EVAL_SPEC_SCHEMA:
        raise ValueError(
            f"unsupported EVAL_SPEC schema {spec.get('schema_version')!r}; expected {EVAL_SPEC_SCHEMA}"
        )
    expected = spec.get("contract_sha256")
    unsigned = {key: value for key, value in spec.items() if key != "contract_sha256"}
    actual = _canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError(f"EVAL_SPEC contract hash mismatch: recorded {expected}, computed {actual}")
    build_config(spec["resolved_config"])
    return spec
