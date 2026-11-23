"""Small local/GCS storage boundary used by open-loop evaluation.

Large Orbax trees are cached locally through ``gcloud storage rsync``.  This is
intentional: workstation gcloud credentials do not imply the ADC credentials that
gcsfs/Orbax require.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

MARKER = "commit_success.txt"
_STEP_RE = re.compile(r"params_bf16_step(\d+)")


class StorageError(RuntimeError):
    pass


def is_gcs(uri: str) -> bool:
    return str(uri).startswith("gs://")


def _run(args: list[str]) -> bytes:
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode:
        stderr = result.stderr.decode(errors="replace")[-4000:]
        raise StorageError(f"{' '.join(args)} failed ({result.returncode}):\n{stderr}")
    return result.stdout


def read_bytes(uri: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
    if is_gcs(uri):
        args = ["gcloud", "storage", "cat"]
        if byte_range is not None:
            start, end = byte_range
            args.extend(["--range", f"{int(start)}-{int(end)}"])
        args.append(uri)
        return _run(args)
    path = Path(uri).expanduser()
    with path.open("rb") as handle:
        if byte_range is None:
            return handle.read()
        start, end = byte_range
        handle.seek(int(start))
        return handle.read(int(end) - int(start) + 1)


def read_json(uri: str) -> dict:
    try:
        value = json.loads(read_bytes(uri))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"could not read JSON from {uri}: {exc}") from exc
    if not isinstance(value, dict):
        raise StorageError(f"{uri} must contain a JSON object")
    return value


def exists(uri: str) -> bool:
    if not is_gcs(uri):
        return Path(uri).expanduser().exists()
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri], capture_output=True, check=False
    )
    return result.returncode == 0


def finalized_steps(run_uri: str) -> list[int]:
    run_uri = run_uri.rstrip("/")
    if is_gcs(run_uri):
        result = subprocess.run(
            ["gcloud", "storage", "ls", f"{run_uri}/params_bf16_step*/{MARKER}"],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            return []
        names = result.stdout.decode(errors="replace").splitlines()
    else:
        names = [str(path) for path in Path(run_uri).expanduser().glob(f"params_bf16_step*/{MARKER}")]
    return sorted({int(match.group(1)) for name in names if (match := _STEP_RE.search(name))})


def resolve_checkpoint(run_uri: str, step: str | int) -> tuple[int, str]:
    steps = finalized_steps(run_uri)
    if not steps:
        raise StorageError(f"no finalized params_bf16_step* snapshots under {run_uri}")
    if str(step) == "latest":
        selected = steps[-1]
    else:
        try:
            selected = int(step)
        except (TypeError, ValueError) as exc:
            raise StorageError(f"step must be an integer or 'latest', got {step!r}") from exc
        if selected not in steps:
            raise StorageError(
                f"step {selected} is not a finalized evaluation snapshot; available: {steps}"
            )
    uri = f"{run_uri.rstrip('/')}/params_bf16_step{selected:07d}"
    marker = read_json(f"{uri}/{MARKER}")
    if int(marker.get("step", -1)) != selected:
        raise StorageError(f"{uri}/{MARKER} records step {marker.get('step')!r}, expected {selected}")
    return selected, uri


def cache_key(uri: str) -> str:
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


def cache_tree(uri: str, cache_root: str | Path, *, label: str) -> Path:
    """Return a complete local tree, publishing the cache directory atomically."""
    if not is_gcs(uri):
        path = Path(uri).expanduser().resolve()
        if not path.is_dir():
            raise StorageError(f"no such directory: {path}")
        return path

    root = Path(cache_root).expanduser()
    target = root / "assets" / f"{label}-{cache_key(uri)}"
    if target.is_dir() and (label != "checkpoint" or (target / MARKER).is_file()):
        return target
    if target.exists():
        raise StorageError(
            f"cache target exists but is not a complete {label}: {target}; move it aside "
            "and retry rather than trusting or overwriting it"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{os.getpid()}"
    partial.mkdir(parents=False, exist_ok=False)
    _run(["gcloud", "storage", "rsync", "--recursive", uri, str(partial)])
    if not (partial / MARKER).is_file() and label == "checkpoint":
        raise StorageError(f"downloaded checkpoint has no {MARKER}: {partial}")
    os.replace(partial, target)
    return target


def cache_file(uri: str, cache_root: str | Path, *, label: str) -> Path:
    if not is_gcs(uri):
        path = Path(uri).expanduser().resolve()
        if not path.is_file():
            raise StorageError(f"no such file: {path}")
        return path
    root = Path(cache_root).expanduser() / "files"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{label}-{cache_key(uri)}"
    if target.is_file():
        return target
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    _run(["gcloud", "storage", "cp", uri, str(partial)])
    os.replace(partial, target)
    return target
