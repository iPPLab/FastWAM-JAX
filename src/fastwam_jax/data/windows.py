"""Map TWP3 windows onto the aligned LeRobot action, state, and image rows.

The packer reserves the final storage row, so T stored frames give T - 1
sample starts. Indices clamp at row T - 2, and the padding mask flags steps
that would exceed that final sample row. Action and state values come from
separate streams indexed by the same row.
"""

import numpy as np

__all__ = [
    "action_indices",
    "num_windows",
    "proprio_indices",
    "video_indices",
]

def num_windows(n_frames: int) -> int:
    """Sample starts in an episode of `n_frames` stored frames: the LeRobot row count."""
    if n_frames < 2:
        raise ValueError(f"an episode needs at least 2 frames to yield a window, got {n_frames}")
    return int(n_frames) - 1


def _clamped(start: int, deltas: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """`(rows, is_pad)` for `start + deltas`, LeRobot's clamp-and-flag (`:660-673`)."""
    raw = int(start) + deltas
    rows = np.clip(raw, 0, num_windows(n_frames) - 1)
    return rows, raw >= num_windows(n_frames)


def video_indices(
    start: int, n_frames: int, *, video_frames: int = 9, video_stride: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """`(frame_index[9], image_is_pad[9])` for the window starting at row `start`.

    The 33 observation steps are subsampled by `action_video_freq_ratio = 4`, so the
    frames are `start, start+4, ..., start+32`. A raw frame index equals its row index.
    """
    deltas = np.arange(video_frames, dtype=np.int64) * int(video_stride)
    return _clamped(start, deltas, n_frames)


def action_indices(
    start: int, n_frames: int, *, action_horizon: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Action rows and their padding mask for the chunk at row `start`."""
    return _clamped(start, np.arange(action_horizon, dtype=np.int64), n_frames)


def proprio_indices(
    start: int, n_frames: int, *, action_horizon: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """`(qpos_index[32], is_pad[32])` for the proprio rows at `start`.

    The official builds 33 state rows and then throws the last one away to match the
    action length (`robot_video_dataset.py:203`), so this is `qpos[start .. start+31]`,
    clamped. Only index 0 ever reaches the model (`fastwam.py:361`); the rest is carried
    because the official tensor carries it. The official's own `proprio_is_pad` stays
    length 33 and is read by nobody, so we return the length-32 mask that lines up with
    the tensor and emit no `proprio_is_pad` at all.
    """
    return _clamped(start, np.arange(action_horizon, dtype=np.int64), n_frames)
