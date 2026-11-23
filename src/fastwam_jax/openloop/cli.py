"""One-command Gate 1 evaluation for ``<GCS run URI> + <step>``."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath

import yaml

from fastwam_jax.config import ConfigError, build_config

from .probe import ProbeError, build_probe_set
from .spec import EVAL_SPEC_NAME, build_eval_spec, validate_eval_spec
from .storage import (
    StorageError,
    cache_file,
    cache_tree,
    exists,
    read_json,
    resolve_checkpoint,
)


class OpenLoopError(RuntimeError):
    pass


def _default_cache() -> Path:
    if value := os.environ.get("FASTWAM_OPENLOOP_CACHE"):
        return Path(value).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "fastwam" / "openloop"


def _legacy_recipe_member(training_config: str, names: list[str]) -> str:
    normalized = training_config.replace("\\", "/")
    marker = "/configs/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else normalized
    relative = relative.removeprefix("./").removeprefix("configs/")
    wanted = f"configs/{relative}"
    candidates = {name.removeprefix("./"): name for name in names}
    if wanted not in candidates:
        raise OpenLoopError(
            f"the archived source package has no {wanted}; cannot reconstruct the run config"
        )
    return candidates[wanted]


def _archive_yaml(bundle: tarfile.TarFile, member_name: str) -> tuple[dict, bytes]:
    candidates = {name.removeprefix("./"): name for name in bundle.getnames()}
    normalized = member_name.removeprefix("./")
    if normalized not in candidates:
        raise OpenLoopError(f"the archived source package has no {normalized}")
    member = bundle.getmember(candidates[normalized])
    if not member.isfile() or member.size > 2_000_000:
        raise OpenLoopError(f"unsafe or implausible archived config member: {member.name}")
    stream = bundle.extractfile(member)
    if stream is None:
        raise OpenLoopError(f"could not read archived config member: {member.name}")
    payload = stream.read()
    value = yaml.safe_load(io.BytesIO(payload))
    if not isinstance(value, dict):
        raise OpenLoopError(f"archived config is not a YAML mapping: {member.name}")
    return value, payload


def _legacy_contract(run_uri: str, cache_root: Path) -> dict:
    """Reconstruct only the YAML from an old run's content-addressed source package.

    The archive is treated as data and never imported or executed. Current code still
    validates the resulting config, so incompatible historical schemas fail closed.
    """
    run_spec = read_json(f"{run_uri.rstrip('/')}/RUN_SPEC.json")
    package_uri = run_spec.get("code_package_uri")
    package_md5 = run_spec.get("code_package_md5")
    training_config = run_spec.get("training_config")
    if not isinstance(training_config, str) or not training_config:
        raise OpenLoopError("RUN_SPEC must name a self-contained TWP3 training_config")
    if not all(isinstance(value, str) and value for value in (package_uri, package_md5)):
        raise OpenLoopError("RUN_SPEC lacks code_package_uri or code_package_md5")
    archive = cache_file(package_uri, cache_root, label=f"source-{package_md5}.tgz")
    with tarfile.open(archive, "r:gz") as bundle:
        member_name = _legacy_recipe_member(training_config, bundle.getnames())
        _, recipe_bytes = _archive_yaml(bundle, member_name)

    contract_root = cache_root / "contracts" / package_md5
    recipe_path = contract_root / "configs" / PurePosixPath(
        member_name.removeprefix("./").removeprefix("configs/")
    )
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    # The bytes are content-addressed by the enclosing source package. Do not reformat
    # them: retaining the original makes provenance inspection straightforward.
    if not recipe_path.exists():
        recipe_path.write_bytes(recipe_bytes)

    overrides = run_spec.get("overrides") or []
    run_name = str(run_spec.get("run_name") or run_uri.rstrip("/").rsplit("/", 1)[-1])
    contract = build_eval_spec(
        recipe_path,
        run_name=run_name,
        code_package_md5=package_md5,
        code_package_uri=package_uri,
        overrides=overrides,
        configs_dir=contract_root / "configs",
    )
    contract["provenance"] = "reconstructed_from_RUN_SPEC_and_archived_yaml"
    # This reconstructed contract is not presented as the immutable run-side contract.
    contract.pop("contract_sha256", None)
    return contract


def load_contract(run_uri: str, cache_root: Path) -> dict:
    uri = f"{run_uri.rstrip('/')}/{EVAL_SPEC_NAME}"
    if exists(uri):
        try:
            return validate_eval_spec(read_json(uri))
        except ValueError as exc:
            raise OpenLoopError(f"invalid {uri}: {exc}") from exc
    return _legacy_contract(run_uri, cache_root)


def _local_checkpoint(path: str, step: int) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise OpenLoopError(f"--checkpoint-dir is not a directory: {directory}")
    marker = read_json(str(directory / "commit_success.txt"))
    if int(marker.get("step", -1)) != step:
        raise OpenLoopError(
            f"local checkpoint marker records step {marker.get('step')!r}, expected {step}: "
            f"{directory}"
        )
    return directory


def _local_vae(path: str) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir() or not (directory / "vae_config.json").is_file():
        raise OpenLoopError(
            f"--vae-dir must contain a complete VAE with vae_config.json: {directory}"
        )
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="gs://.../runs/<run-name> or local run dir")
    parser.add_argument("--step", required=True, help="integer bf16 snapshot step or latest")
    parser.add_argument("--probe", default="auto", help="probe_set.npz path, or auto")
    parser.add_argument("--probe-task", default=None, help="task_key for an automatic probe")
    parser.add_argument("--probe-episodes", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="verified local params_bf16 snapshot; avoids re-downloading",
    )
    parser.add_argument(
        "--vae-dir",
        default=None,
        help="verified local VAE directory; avoids re-downloading",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="resolve and validate metadata without downloading model assets or running JAX",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.probe_episodes < 1:
        raise OpenLoopError(f"--probe-episodes must be positive, got {args.probe_episodes}")
    if args.device is not None:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != args.device:
            raise OpenLoopError(
                "CUDA_VISIBLE_DEVICES is already set to a different value; set it before launch"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    cache_root = Path(args.cache_dir).expanduser() if args.cache_dir else _default_cache()
    output_root = Path(
        args.output_dir or os.environ.get("FASTWAM_OPENLOOP_DIR", "outputs/openloop")
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    contract = load_contract(args.run, cache_root)
    cfg = build_config(contract["resolved_config"])
    data_contract = contract.get("data_contract") or {
        "text_mode": cfg.data.l3.text_mode,
        "action_shift": cfg.data.l3.action_shift,
    }
    step, checkpoint_uri = resolve_checkpoint(args.run, args.step)
    local_checkpoint = _local_checkpoint(args.checkpoint_dir, step) if args.checkpoint_dir else None
    local_vae = _local_vae(args.vae_dir) if args.vae_dir else None
    tag = args.tag or f"{contract['run_name']}-step{step:07d}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise OpenLoopError(
            f"tag must contain only letters, digits, dot, underscore, and dash; got {tag!r}"
        )
    if args.print_plan:
        print(
            json.dumps(
                {
                    "run": args.run,
                    "step": step,
                    "checkpoint_uri": checkpoint_uri,
                    "source": contract["source"],
                    "embodiment": contract["embodiment"],
                    "data_contract": data_contract,
                    "assets": contract["assets"],
                    "local_checkpoint": str(local_checkpoint) if local_checkpoint else None,
                    "local_vae": str(local_vae) if local_vae else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    checkpoint = local_checkpoint or cache_tree(checkpoint_uri, cache_root, label="checkpoint")
    vae_uri = contract["assets"]["vae"]
    vae = local_vae or cache_tree(vae_uri, cache_root, label="vae")

    contract_id = contract.get("contract_sha256") or hashlib.sha256(
        json.dumps(contract["resolved_config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.probe == "auto":
        selected_tasks = cfg.data.l3.task_filter
        effective_task = args.probe_task or (selected_tasks[0] if selected_tasks else None)
        task_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", effective_task or "auto").strip("-.")
        task_key = task_key or "task"
        probe = cache_root / "probes" / f"{contract_id[:16]}-{task_key}-{args.probe_episodes}.npz"
        if not probe.exists():
            build_probe_set(
                cfg, probe, task=effective_task, episodes=args.probe_episodes
            )
    else:
        probe = Path(args.probe).expanduser().resolve()
        if not probe.is_file():
            raise OpenLoopError(f"no such probe set: {probe}")

    # Importing inference initializes JAX, so it happens only after --device is applied.
    import jax.numpy as jnp

    from fastwam_jax.inference.load import build_policy

    from .runner import evaluate_policy

    stats = cache_file(contract["assets"]["stats"], cache_root, label="stats.json")
    policy = build_policy(
        ckpt_root=str(checkpoint),
        vae_ckpt=str(vae),
        stats_path=str(stats),
        stats_sha256=contract["assets"].get("stats_sha256"),
        config=cfg,
        dtype=jnp.bfloat16,
    )
    metrics, predictions = evaluate_policy(policy, probe)
    report = {
        "schema_version": 1,
        "tag": tag,
        "run": args.run,
        "step": step,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_cache": str(checkpoint),
        "probe": str(probe),
        "contract_sha256": contract.get("contract_sha256"),
        "code_package_md5": contract["source"]["code_package_md5"],
        "embodiment": contract["embodiment"],
        "metrics": metrics,
    }
    report_path = output_root / f"gate1_{tag}.json"
    prediction_path = output_root / f"gate1_{tag}.npz"
    occupied = [path for path in (report_path, prediction_path) if path.exists()]
    if occupied:
        raise OpenLoopError(
            f"refusing to overwrite an existing Gate 1 result: {occupied}; choose a new --tag"
        )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    import numpy as np

    np.savez_compressed(prediction_path, **predictions)
    status = "PASS" if metrics["gate1"]["pass"] else "FAIL"
    print(
        f"Gate 1 {status}: skill={metrics['skill']:+.3f} "
        f"zero_frame={metrics['zero_frame_ratio']:.2f}x\n"
        f"report: {report_path}\npredictions: {prediction_path}",
        flush=True,
    )
    return 0 if metrics["gate1"]["pass"] else 1


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (ConfigError, OpenLoopError, ProbeError, StorageError) as exc:
        raise SystemExit(f"open-loop evaluation failed: {exc}") from exc
