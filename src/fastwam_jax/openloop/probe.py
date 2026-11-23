"""Build a deterministic deployment-style probe from the run's training data."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from fastwam_jax.data import twp, windows
from fastwam_jax.data.canvas import CANVAS_HW, resolve_tiles

from .storage import read_bytes


class ProbeError(ValueError):
    pass


def compose_uint8_canvas(
    streams: dict[str, np.ndarray],
    *,
    tiles,
) -> np.ndarray:
    """Compose one deployment canvas from decoded uint8 camera frames.

    This intentionally stays uint8 and uses Pillow's bilinear resize, matching the
    closed-loop adapter. Training's float resize is a different boundary.
    """
    resolved = resolve_tiles(tiles)
    missing = [role for role, _ in resolved if role not in streams]
    if missing:
        raise ProbeError(f"canvas is missing camera roles {missing}; have {sorted(streams)}")
    canvas = np.empty((*CANVAS_HW, 3), np.uint8)
    for role, (y0, x0, height, width) in resolved:
        frame = np.asarray(streams[role], np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ProbeError(f"camera {role!r} must be uint8 [H,W,3], got {frame.shape}")
        tile = np.asarray(
            Image.fromarray(frame, "RGB").resize((width, height), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
        canvas[y0 : y0 + height, x0 : x0 + width] = tile
    return np.ascontiguousarray(canvas)


def _pick_entries(manifest: dict, *, task: str | None, episodes: int) -> tuple[str, list[dict]]:
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ProbeError("MANIFEST.json has no non-empty episodes list")
    entries = [
        item
        for item in entries
        if not item.get("bad_jpeg", 0) and not item.get("nonfinite_actions", 0)
    ]
    if not entries:
        raise ProbeError("MANIFEST.json has no usable episodes after integrity filtering")
    available = sorted({str(item.get("task_key", item.get("task", ""))) for item in entries})
    available = [name for name in available if name]
    selected_task = task or (available[0] if available else None)
    if not selected_task:
        raise ProbeError("MANIFEST.json entries have no task_key/task")
    chosen = [
        item
        for item in entries
        if str(item.get("task_key", item.get("task", ""))) == selected_task
    ][: int(episodes)]
    if not chosen:
        raise ProbeError(f"task {selected_task!r} is absent; available tasks: {available}")
    return selected_task, chosen


def probe_identity(cfg, dataset_spec: dict, manifest: dict, *, task: str | None, episodes: int) -> str:
    identity = {
        "schema": 1,
        "dataset_fingerprint": dataset_spec.get("index_fingerprint")
        or manifest.get("fingerprint"),
        "training_data": cfg.data.l3.prefix,
        "task": task,
        "episodes": int(episodes),
        "action_shift": cfg.data.l3.action_shift,
        "camera_roles": dataset_spec.get("camera_roles"),
        "canvas_tiles": dataset_spec.get("canvas_tiles"),
        "source_color_order": cfg.data.l3.source_color_order,
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_probe_set(
    cfg,
    output: str | Path,
    *,
    task: str | None = None,
    episodes: int = 3,
) -> Path:
    prefix = cfg.data.l3.prefix.rstrip("/")
    text_dir = cfg.data.l3.text_dir
    if not text_dir:
        raise ProbeError("automatic Gate 1 probes require a configured text side table")
    dataset_spec = json.loads(read_bytes(f"{prefix}/spec.json"))
    manifest = json.loads(read_bytes(f"{prefix}/MANIFEST.json"))
    selected_task, chosen = _pick_entries(manifest, task=task, episodes=episodes)

    declared_action_dim = dataset_spec.get("action_dim")
    declared_state_dim = dataset_spec.get("state_dim", declared_action_dim)
    expected_action_dim = cfg.model.action_dit_config.action_dim
    expected_state_dim = cfg.model.proprio_dim
    if declared_action_dim is not None and int(declared_action_dim) != expected_action_dim:
        raise ProbeError(
            f"dataset action_dim={declared_action_dim}, config action_dim={expected_action_dim}"
        )
    if declared_state_dim is not None and int(declared_state_dim) != expected_state_dim:
        raise ProbeError(
            f"dataset state_dim={declared_state_dim}, config proprio_dim={expected_state_dim}"
        )
    try:
        twp.validate_alignment(dataset_spec)
    except ValueError as exc:
        raise ProbeError(str(exc)) from exc

    roles = dataset_spec.get("camera_roles") or dataset_spec.get("cameras")
    if not isinstance(roles, list) or not roles:
        raise ProbeError("dataset spec has no camera_roles/cameras")
    tiles = resolve_tiles(dataset_spec.get("canvas_tiles"))
    if len(tiles) != cfg.data.train.processor.num_output_cameras:
        raise ProbeError(
            f"dataset canvas has {len(tiles)} cameras, config declares "
            f"{cfg.data.train.processor.num_output_cameras}"
        )
    needed_roles = [role for role, _ in tiles]
    missing = [role for role in needed_roles if role not in roles]
    if missing:
        raise ProbeError(f"canvas needs roles {missing}, dataset declares {roles}")
    columns = {role: roles.index(role) for role in needed_roles}

    meta = json.loads(read_bytes(f"{text_dir}/meta.json"))
    rows = np.frombuffer(read_bytes(f"{text_dir}/rows.i32"), np.int32)
    texts = json.loads(read_bytes(f"{text_dir}/texts.json"))
    max_len, d_model = int(meta["max_len"]), int(meta["d_model"])
    row_bytes = max_len * d_model * 2

    samples = []
    for entry in chosen:
        episode_id = int(entry.get("episode_id", entry.get("id")))
        blob = read_bytes(f"{prefix}/episode_{episode_id:06d}.twp")
        episode = twp.Episode(blob, expect_id=episode_id)
        if episode.action_dim != cfg.model.action_dit_config.action_dim:
            raise ProbeError(
                f"episode {episode_id} is {episode.action_dim}-D but config action_dim is "
                f"{cfg.model.action_dit_config.action_dim}"
            )
        if episode.states.shape[1] != cfg.model.proprio_dim:
            raise ProbeError(
                f"episode {episode_id} state is {episode.states.shape[1]}-D but proprio_dim is "
                f"{cfg.model.proprio_dim}; current TWP3 files use one shared width"
            )
        row = int(rows[episode_id])
        context_blob = read_bytes(
            f"{text_dir}/emb.f16",
            byte_range=(row * row_bytes, (row + 1) * row_bytes - 1),
        )
        context = np.frombuffer(context_blob, np.float16).reshape(max_len, d_model).astype(np.float32)
        n_windows = windows.num_windows(episode.n_frames)
        for fraction in (0.1, 0.35, 0.6, 0.85):
            start = int(n_windows * fraction)
            action_idx, action_pad = windows.action_indices(
                start,
                episode.n_frames,
                action_horizon=cfg.shapes.action_tokens,
            )
            streams = {
                role: np.asarray(
                    Image.open(io.BytesIO(bytes(episode.jpeg(start, column)))).convert("RGB"),
                    np.uint8,
                )
                for role, column in columns.items()
            }
            canvas = compose_uint8_canvas(streams, tiles=tiles)
            samples.append(
                {
                    "episode_id": episode_id,
                    "start": start,
                    "canvas": canvas,
                    "qpos": np.asarray(episode.states[start], np.float32),
                    "ground_truth": np.asarray(episode.actions[action_idx], np.float32),
                    "action_is_pad": action_pad,
                    "context": context,
                    "text": texts[row],
                }
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = probe_identity(cfg, dataset_spec, manifest, task=selected_task, episodes=episodes)
    partial = output.with_name(f".{output.stem}.partial-{os.getpid()}.npz")
    np.savez_compressed(
        partial,
        canvas=np.stack([item["canvas"] for item in samples]),
        qpos=np.stack([item["qpos"] for item in samples]),
        ground_truth=np.stack([item["ground_truth"] for item in samples]),
        action_is_pad=np.stack([item["action_is_pad"] for item in samples]),
        context=np.stack([item["context"] for item in samples]),
        meta=json.dumps(
            {
                "schema_version": 1,
                "fingerprint": identity,
                "task": selected_task,
                "episodes": [int(item.get("episode_id", item.get("id"))) for item in chosen],
                "windows": [
                    {"episode_id": item["episode_id"], "start": item["start"], "text": item["text"]}
                    for item in samples
                ],
            },
            sort_keys=True,
        ),
    )
    os.replace(partial, output)
    return output
