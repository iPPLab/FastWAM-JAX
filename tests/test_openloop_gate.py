import copy
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from fastwam_jax.openloop.cli import OpenLoopError, _local_checkpoint, load_contract
from fastwam_jax.openloop.probe import ProbeError, compose_uint8_canvas
from fastwam_jax.openloop.runner import score_predictions
from fastwam_jax.openloop.spec import build_eval_spec, validate_eval_spec
from fastwam_jax.openloop.storage import StorageError, finalized_steps, resolve_checkpoint


def test_public_file_entrypoint_needs_no_pythonpath():
    script = Path(__file__).parents[1] / "tools" / "openloop" / "eval_run.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--run" in result.stdout and "--step" in result.stdout
    assert "latent" not in result.stdout


def test_eval_spec_round_trips_and_detects_mutation():
    spec = build_eval_spec(
        "training/full.example.yaml",
        run_name="robotwin-test",
        code_package_md5="abc123",
        code_package_uri="gs://example-bucket/code/fastwam_jax_abc123.tgz",
    )
    assert validate_eval_spec(copy.deepcopy(spec)) == spec
    assert spec["embodiment"] == {
        "action_dim": 14,
        "proprio_dim": 14,
        "num_output_cameras": 3,
        "video_size": [384, 320],
        "canvas_layout": "robotwin",
    }
    assert spec["data_contract"]["action_shift"] == 0
    assert "compatibility" not in spec
    changed = copy.deepcopy(spec)
    changed["embodiment"]["action_dim"] = 13
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_eval_spec(changed)


def test_legacy_run_reconstructs_archived_yaml_without_executing_code(tmp_path):
    template = (
        Path(__file__).parents[1]
        / "configs"
        / "training"
        / "full.example.yaml"
    )
    source = tmp_path / "robotwin.yaml"
    source.write_text(
        template.read_text(encoding="utf-8").replace("your-fastwam-bucket", "example-bucket"),
        encoding="utf-8",
    )
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="./configs/training/robotwin.yaml")
    run = tmp_path / "legacy-run"
    run.mkdir()
    (run / "RUN_SPEC.json").write_text(
        json.dumps(
            {
                "run_name": "legacy-robotwin",
                "training_config": "configs/training/robotwin.yaml",
                "code_package_md5": "deadbeef",
                "code_package_uri": str(archive),
                "overrides": [],
            }
        )
    )
    contract = load_contract(str(run), tmp_path / "cache")
    assert contract["provenance"] == "reconstructed_from_RUN_SPEC_and_archived_yaml"
    assert contract["data_contract"]["action_shift"] == 0
    assert contract["embodiment"]["action_dim"] == 14


def test_local_checkpoint_resolution_requires_commit_markers(tmp_path):
    for step in (2500, 10000):
        directory = tmp_path / f"params_bf16_step{step:07d}"
        directory.mkdir()
        (directory / "commit_success.txt").write_text(json.dumps({"step": step}))
    incomplete = tmp_path / "params_bf16_step0012500"
    incomplete.mkdir()

    assert finalized_steps(str(tmp_path)) == [2500, 10000]
    assert resolve_checkpoint(str(tmp_path), "latest") == (
        10000,
        f"{tmp_path}/params_bf16_step0010000",
    )
    with pytest.raises(StorageError, match="available"):
        resolve_checkpoint(str(tmp_path), "12500")
    assert _local_checkpoint(str(tmp_path / "params_bf16_step0010000"), 10000).is_dir()
    with pytest.raises(OpenLoopError, match="expected 2500"):
        _local_checkpoint(str(tmp_path / "params_bf16_step0010000"), 2500)


def test_two_camera_deployment_canvas_uses_dataset_tiles():
    tiles = [
        {"role": "external", "box": [0, 0, 256, 320]},
        {"role": "wrist", "box": [256, 0, 128, 320]},
    ]
    streams = {
        "external": np.full((32, 40, 3), [10, 20, 30], np.uint8),
        "wrist": np.full((24, 36, 3), [40, 50, 60], np.uint8),
    }
    canvas = compose_uint8_canvas(streams, tiles=tiles)
    assert canvas.shape == (384, 320, 3) and canvas.dtype == np.uint8
    np.testing.assert_array_equal(canvas[10, 10], [10, 20, 30])
    np.testing.assert_array_equal(canvas[300, 10], [40, 50, 60])

    with pytest.raises(ProbeError, match="missing camera"):
        compose_uint8_canvas({"external": streams["external"]}, tiles=tiles)


def test_gate_metrics_are_dimension_agnostic():
    # Synthetic eight-dimensional motion. Clean recovers 90% of the motion while a
    # grey frame is worse than holding still.
    qpos = np.zeros((2, 8), np.float32)
    truth = np.ones((2, 4, 8), np.float32)
    clean = np.full_like(truth, 0.9)
    grey = np.full_like(truth, -0.2)
    report = score_predictions(clean=clean, zero_frame=grey, ground_truth=truth, qpos=qpos)
    assert report["skill"] == pytest.approx(0.9)
    assert report["zero_frame_ratio"] == pytest.approx(12.0)
    assert report["gate1"]["pass"] is True
