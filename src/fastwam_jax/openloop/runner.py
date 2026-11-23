"""Policy-independent Gate 1 execution and metric calculation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SKILL_THRESHOLD = 0.7
ZERO_FRAME_RATIO_THRESHOLD = 3.0


def _l2(a, b) -> float:
    return float(np.sqrt(np.square(np.asarray(a) - np.asarray(b)).mean()))


def score_predictions(*, clean, zero_frame, ground_truth, qpos) -> dict:
    hold = np.repeat(qpos[:, None, :], ground_truth.shape[1], axis=1)
    clean_l2 = _l2(clean, ground_truth)
    hold_l2 = _l2(hold, ground_truth)
    zero_l2 = _l2(zero_frame, ground_truth)
    skill = 1.0 - clean_l2 / hold_l2 if hold_l2 else float("-inf")
    zero_ratio = zero_l2 / clean_l2 if clean_l2 else float("inf")
    skill_pass = skill >= SKILL_THRESHOLD
    visual_pass = zero_ratio >= ZERO_FRAME_RATIO_THRESHOLD
    return {
        "clean_l2": clean_l2,
        "hold_l2": hold_l2,
        "skill": skill,
        "zero_frame_l2": zero_l2,
        "zero_frame_ratio": zero_ratio,
        "thresholds": {
            "skill": SKILL_THRESHOLD,
            "zero_frame_ratio": ZERO_FRAME_RATIO_THRESHOLD,
        },
        "gate1": {
            "skill_pass": skill_pass,
            "visual_sensitivity_pass": visual_pass,
            "pass": skill_pass and visual_pass,
        },
    }


def evaluate_policy(policy, probe_path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(probe_path, allow_pickle=True) as probe:
        canvas = probe["canvas"]
        qpos = probe["qpos"]
        context = probe["context"] if "context" in probe else probe["train_ctx"]
        ground_truth = probe["ground_truth"] if "ground_truth" in probe else probe["gt_shift1"]
        meta = json.loads(str(probe["meta"]))
    mask = np.ones(context.shape[1], bool)

    predictions = {}
    for arm in ("clean", "zero_frame"):
        rows = []
        for index in range(len(canvas)):
            policy.reset()
            frame = canvas[index] if arm == "clean" else np.full_like(canvas[index], 128)
            rows.append(policy.infer_action(frame, qpos[index], context[index], mask))
        predictions[arm] = np.stack(rows)
    metrics = score_predictions(
        clean=predictions["clean"],
        zero_frame=predictions["zero_frame"],
        ground_truth=ground_truth,
        qpos=qpos,
    )
    metrics["probe"] = meta
    return metrics, {**predictions, "ground_truth": ground_truth, "qpos": qpos}
