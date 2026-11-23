"""Check the resize kernel, tile placement, and RGB channel order."""

import numpy as np
import pytest

from fastwam_jax.data import canvas as cv

# --------------------------------------------------------------------------- #
# resize kernel
# --------------------------------------------------------------------------- #
def test_downsampling_weights_are_the_hand_computed_triangle():
    """4 -> 2, worked out from torch's `_compute_weights_aa`:

        scale = 2, support = 2, invscale = 0.5
        i=0: centre 1.0, taps 0..2, w = 1 - |(tap - 1.0 + 0.5) * 0.5|
                                      = 0.75, 0.75, 0.25 -> /1.75
    """
    idx, weight = cv._aa_bilinear_taps(4, 2)
    assert idx.tolist() == [[0, 1, 2], [1, 2, 3]]
    expected = np.array([0.75, 0.75, 0.25]) / 1.75
    np.testing.assert_allclose(weight[0], expected, rtol=1e-6)
    np.testing.assert_allclose(weight[1], expected[::-1], rtol=1e-6)


def test_upsampling_reduces_to_plain_bilinear():
    """With scale < 1 the antialias support collapses to 1, so 2 -> 4 is ordinary
    align_corners=False bilinear: the two border outputs read a single source pixel."""
    idx, weight = cv._aa_bilinear_taps(2, 4)
    np.testing.assert_allclose(weight[0], [1.0, 0.0])
    np.testing.assert_allclose(weight[1], [0.75, 0.25])
    np.testing.assert_allclose(weight[2], [0.25, 0.75])
    np.testing.assert_allclose(weight[3], [1.0, 0.0])
    assert idx[1].tolist() == [0, 1] and idx[2].tolist() == [0, 1]


def test_weights_sum_to_one_everywhere():
    for in_size, out_size in [(240, 256), (240, 128), (320, 160), (320, 320), (7, 3), (3, 7)]:
        _, weight = cv._aa_bilinear_taps(in_size, out_size)
        np.testing.assert_allclose(weight.sum(axis=1), 1.0, rtol=1e-6)


def test_resize_is_identity_at_the_same_size():
    x = np.random.default_rng(0).random((2, 240, 320, 3)).astype(np.float32)
    assert cv.resize_bilinear_aa(x, 240, 320) is x


def test_resize_matches_pil_within_uint8_rounding():
    """PIL's `Image.resize(BILINEAR)` computes the same triangle-with-scaled-support
    kernel torch's antialiased bilinear does, so agreement to one grey level is the
    strongest cross-check available without torch installed. Our version stays in
    float32 throughout, which is what the official does (it resizes after ToTensor)."""
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(7)
    for out_h, out_w in [(256, 320), (128, 160)]:
        src = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        mine = cv.resize_bilinear_aa(src.astype(np.float32) / 255.0, out_h, out_w) * 255.0
        pil = np.asarray(Image.fromarray(src).resize((out_w, out_h), Image.BILINEAR), np.float32)
        assert np.abs(mine - pil).max() <= 1.0


def test_resize_rejects_non_float_input():
    with pytest.raises(ValueError, match="float32"):
        cv.resize_bilinear_aa(np.zeros((2, 4, 4, 3), np.uint8), 2, 2)


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def test_tiles_are_the_official_board():
    """`concat_multi_camera="robotwin"` (`robot_video_dataset.py:159-178`): head to
    [256,320] on top; the two wrists to [128,160] each, concatenated on W in left-then-
    right order, then stacked under the head on H."""
    assert cv.CANVAS_HW == (384, 320)
    assert dict(cv.TILES) == {
        "head": (0, 0, 256, 320),
        "left_wrist": (256, 0, 128, 160),
        "right_wrist": (256, 160, 128, 160),
    }


def test_layout_covers_the_canvas_on_the_patch_grid():
    cv._check_layout()  # raises on overlap, gap, or an edge off the 16px grid
    for _, (y0, x0, h, w) in cv.TILES:
        assert (y0 % 16, x0 % 16, h % 16, w % 16) == (0, 0, 0, 0)


def test_assert_canvas_matches_rejects_other_geometry():
    cv.assert_canvas_matches((384, 320), 3)
    cv.assert_canvas_matches((384, 320), 2)
    with pytest.raises(ValueError, match="does not match"):
        cv.assert_canvas_matches((320, 512), 3)
    with pytest.raises(ValueError, match="at least one"):
        cv.assert_canvas_matches((384, 320), 0)


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #
def _flat(colour, h=240, w=320, t=1):
    return np.tile(np.asarray(colour, np.uint8), (t, h, w, 1))


def test_each_camera_lands_in_its_own_tile():
    """A constant colour per camera, so a swapped or transposed tile is visible as a
    wrong constant. Constant input means the resize is exactly the identity value."""
    streams = {
        "head": _flat((10, 20, 30)),
        "left_wrist": _flat((40, 50, 60)),
        "right_wrist": _flat((70, 80, 90)),
    }
    out = cv.compose_canvas(streams)
    assert out.shape == (1, 384, 320, 3) and out.dtype == np.float32

    def expect(c):
        return np.float32(2.0 * (c / 255.0) - 1.0)

    np.testing.assert_allclose(out[0, :256, :, 0], expect(10), atol=1e-6)
    np.testing.assert_allclose(out[0, 256:, :160, 0], expect(40), atol=1e-6)
    np.testing.assert_allclose(out[0, 256:, 160:, 0], expect(70), atol=1e-6)
    np.testing.assert_allclose(out[0, 256:, :160, 2], expect(60), atol=1e-6)


def test_dataset_defined_two_camera_layout_fills_the_same_canvas():
    tiles = [
        {"role": "head", "box": [0, 0, 256, 320]},
        {"role": "wrist", "box": [256, 0, 128, 320]},
    ]
    out = cv.compose_canvas(
        {"head": _flat((10, 20, 30)), "wrist": _flat((40, 50, 60))},
        tiles=tiles,
    )
    assert out.shape == (1, 384, 320, 3)
    np.testing.assert_allclose(out[0, :256, :, 0], 2 * 10 / 255 - 1, atol=1e-6)
    np.testing.assert_allclose(out[0, 256:, :, 0], 2 * 40 / 255 - 1, atol=1e-6)


def test_custom_layout_rejects_gaps_and_off_grid_edges():
    with pytest.raises(ValueError, match="do not cover"):
        cv.resolve_tiles([{"role": "head", "box": [0, 0, 256, 320]}])
    with pytest.raises(ValueError, match="patch grid"):
        cv.resolve_tiles([{"role": "head", "box": [0, 0, 384, 319]}])


def test_output_range_and_normalisation():
    streams = {r: _flat((0, 127, 255)) for r, _ in cv.TILES}
    out = cv.compose_canvas(streams)
    # The taps are normalised in float32, so a saturated pixel can land an ulp outside
    # [-1, 1]. torch's kernel does the same; clamping here would be a deviation from the
    # official path to buy nothing.
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5
    np.testing.assert_allclose(out[0, 0, 0], [-1.0, 2 * 127 / 255 - 1, 1.0], atol=1e-6)


def test_missing_camera_and_ragged_lengths_raise():
    streams = {"head": _flat((1, 2, 3)), "left_wrist": _flat((1, 2, 3))}
    with pytest.raises(ValueError, match="missing camera"):
        cv.compose_canvas(streams)
    streams["right_wrist"] = _flat((1, 2, 3), t=2)
    with pytest.raises(ValueError, match="disagree on frame count"):
        cv.compose_canvas(streams)


def test_jpeg_decoder_is_named_and_pinned():
    """simplejpeg and PIL differ by up to 2/255, so which one ran is part of the run's
    provenance rather than whatever happened to be installed."""
    assert cv.decoder_name in ("simplejpeg", "PIL")
