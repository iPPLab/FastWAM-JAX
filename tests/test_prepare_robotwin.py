import json

import numpy as np
import pytest

from fastwam_jax.data import twp, windows
from tools import prepare_robotwin


@pytest.mark.parametrize("count", [1, 3])
def test_pack_keeps_last_lerobot_row_as_a_sample_start(tmp_path, monkeypatch, count):
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    source = tmp_path / "raw"
    meta = source / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "data_path": "episode_{episode_index:06d}.parquet",
        "video_path": "{video_key}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 5, "task": "Pick up the cube."}))
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": count}))
    actions = np.arange(count * 14, dtype=np.float32).reshape(count, 14)
    states = actions + 100
    parquet.write_table(
        arrow.table({
            "episode_index": [0] * count,
            "frame_index": list(range(count)),
            "action": actions.tolist(),
            "observation.state": states.tolist(),
            "task_index": [5] * count,
        }),
        source / "episode_000000.parquet",
    )
    monkeypatch.setattr(
        prepare_robotwin, "read_jpegs",
        lambda path: (f"{path.stem}.{index}".encode() for index in range(count)),
    )

    output = tmp_path / "packed"
    manifest = prepare_robotwin.convert(source, output, {})
    episode = twp.Episode((output / "episode_000000.twp").read_bytes(), expect_id=0)

    assert episode.n_frames == manifest["episodes"][0]["n_frames"] == count + 1
    assert manifest["n_transitions"] == windows.num_windows(episode.n_frames) == count
    rows, padded = windows.action_indices(count - 1, episode.n_frames, action_horizon=2)
    np.testing.assert_array_equal(rows, [count - 1, count - 1])
    np.testing.assert_array_equal(padded, [False, True])
    np.testing.assert_array_equal(episode.actions[rows[0]], actions[-1])
    np.testing.assert_array_equal(episode.states[rows[0]], states[-1])
    assert bytes(episode.jpeg(count - 1, 0)).endswith(f".{count - 1}".encode())
