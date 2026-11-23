import numpy as np
import pytest

from fastwam_jax.training.train import _reuse_action_backbone


def trees():
    target = {
        "action_encoder": {
            "kernel": np.full((8, 4), -1.0, np.float32),
            "bias": np.full((4,), -2.0, np.float32),
        },
        "blocks": {"kernel": np.full((3, 4), -3.0, np.float32)},
        "proj_out": {
            "kernel": np.full((4, 8), -4.0, np.float32),
            "bias": np.full((8,), -5.0, np.float32),
        },
    }
    loaded = {
        "action_encoder": {
            "kernel": np.full((14, 4), 1.0, np.float32),
            "bias": np.full((4,), 2.0, np.float32),
        },
        "blocks": {"kernel": np.full((3, 4), 3.0, np.float32)},
        "proj_out": {
            "kernel": np.full((4, 14), 4.0, np.float32),
            "bias": np.full((14,), 5.0, np.float32),
        },
    }
    return target, loaded


def test_action_width_change_reuses_backbone_and_reinitializes_only_io():
    target, loaded = trees()
    merged = _reuse_action_backbone(target, loaded, "test-checkpoint")
    np.testing.assert_array_equal(merged["blocks"]["kernel"], loaded["blocks"]["kernel"])
    np.testing.assert_array_equal(merged["action_encoder"]["bias"], loaded["action_encoder"]["bias"])
    np.testing.assert_array_equal(merged["action_encoder"]["kernel"], target["action_encoder"]["kernel"])
    np.testing.assert_array_equal(merged["proj_out"]["kernel"], target["proj_out"]["kernel"])
    np.testing.assert_array_equal(merged["proj_out"]["bias"], target["proj_out"]["bias"])


def test_action_width_change_rejects_an_unexpected_backbone_mismatch():
    target, loaded = trees()
    loaded["blocks"]["kernel"] = np.zeros((7, 4), np.float32)
    with pytest.raises(ValueError, match="unexpected action-backbone shape mismatch"):
        _reuse_action_backbone(target, loaded, "test-checkpoint")
