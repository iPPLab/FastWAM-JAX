"""Pixels: JPEG bytes -> the official RoboTwin 384x320 three-camera canvas.

This is the whole pixel path, and it is deliberately one module: training and the
closed-loop harness must produce byte-identical canvases, and the way that goes wrong
here is never an exception, it is a score that quietly collapses.

What the official does (`robot_video_dataset.py:145-197`, verified line by line):

    decode mp4 -> float [0,1] -> (x*255).to(uint8)      base_lerobot_dataset.py:148
    ToTensor                   -> float32 /255          transforms/image.py:6-13
    Resize([240,320])          -> no-op on our data     the config's `shape`
    subsample 9 of the 33 obs frames                    :63, :145
    cam_high      -> resize [256,320]  bilinear, antialias=True
    cam_left_wrist-> resize [128,160]  bilinear, antialias=True
    cam_right_wrist->resize [128,160]  bilinear, antialias=True
    bottom = cat([left, right], dim=-1)     -> [9,3,128,320]
    video  = cat([top, bottom], dim=-2)     -> [9,3,384,320]
    ResizeSmallestSideAspectPreserving([384,320])       -> identity (scale is exactly 1)
    CenterCrop([384,320])                               -> identity
    Normalize(mean=0.5, std=0.5)                        -> [0,1] to [-1,1]
    permute(1,0,2,3)                                    -> [3,9,384,320]

Three things this module has to get right that nothing downstream can check:

* **Resolution.** The official's `raw_shape: [3,480,640]` is never asserted (the only
  shape assert is on the post-transform `[33,3,240,320]`, `fastwam_processor.py:227`),
  and our L3 stores 240x320 JPEGs, so the official's `Resize([240,320])` is a no-op for
  us and we start from the same pixels it does.
* **Channel order.** The packer and simulator provide RGB frames.
* **Resize kernel.** `resize_bilinear_aa` re-implements torch's `_upsample_bilinear2d_aa`
  weights exactly (the same triangle filter with support scaled by the reduction factor
  that PIL's `Image.resize(BILINEAR)` uses), in float32. Doing it here rather than through
  PIL keeps the arithmetic in float — PIL would re-quantise to uint8 between the two
  separable passes — and makes the weights something a test can compute by hand. Measured
  against PIL on the real 240x320 -> 256x320 and -> 128x160 steps, the two agree to within
  one grey level.

numpy only. No jax, no torch: this runs on the TPU host's data threads and inside the
env harness, neither of which has an accelerator context to spare.
"""

import numpy as np

__all__ = [
    "CANVAS_HW",
    "CAMERA_ROLES",
    "TILES",
    "assert_canvas_matches",
    "compose_canvas",
    "decode_jpeg",
    "decoder_name",
    "resolve_tiles",
    "resize_bilinear_aa",
]

# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
CANVAS_HW = (384, 320)

# (role, (y0, x0, height, width)) — the official `concat_multi_camera="robotwin"` board.
# `cam_high` fills the top; the two wrists sit side by side underneath, left then right,
# because the official concatenates on W in that order before stacking on H.
TILES: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("head", (0, 0, 256, 320)),
    ("left_wrist", (256, 0, 128, 160)),
    ("right_wrist", (256, 160, 128, 160)),
)
CAMERA_ROLES: tuple[str, ...] = tuple(role for role, _ in TILES)

# Wan2.2 VAE is 16x spatial and the DiT patches 2x2 on top of that, so a tile edge off
# the 16px grid would put one token astride two cameras.
_PATCH_PX = 16

def _check_layout(tiles=TILES, canvas_hw=CANVAS_HW) -> None:
    covered = np.zeros(canvas_hw, dtype=bool)
    for role, (y0, x0, th, tw) in tiles:
        for name, v in (("y0", y0), ("x0", x0), ("h", th), ("w", tw)):
            if v % _PATCH_PX:
                raise AssertionError(f"tile {role}: {name}={v} is off the {_PATCH_PX}px patch grid")
        if y0 + th > canvas_hw[0] or x0 + tw > canvas_hw[1]:
            raise AssertionError(f"tile {role} at {(y0, x0, th, tw)} runs off the {canvas_hw} canvas")
        if covered[y0 : y0 + th, x0 : x0 + tw].any():
            raise AssertionError(f"tile {role} overlaps an earlier tile")
        covered[y0 : y0 + th, x0 : x0 + tw] = True
    if not covered.all():
        raise AssertionError("the tiles do not cover the canvas; a fill colour would be a guess")


_check_layout()


def resolve_tiles(raw=None) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Parse an optional dataset ``canvas_tiles`` list; absent means RoboTwin's board."""
    if raw is None:
        return TILES
    try:
        tiles = tuple(
            (str(item["role"]), tuple(int(v) for v in item["box"]))
            if isinstance(item, dict)
            else (str(item[0]), tuple(int(v) for v in item[1]))
            for item in raw
        )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("canvas_tiles must be [{role: str, box: [y0,x0,h,w]}, ...]") from exc
    if not tiles or any(len(box) != 4 for _, box in tiles):
        raise ValueError("canvas_tiles must contain at least one four-integer box")
    roles = [role for role, _ in tiles]
    if len(roles) != len(set(roles)):
        raise ValueError(f"canvas_tiles repeats camera roles: {roles}")
    try:
        _check_layout(tiles)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc
    return tiles


def assert_canvas_matches(video_size: tuple[int, int], layout) -> None:
    """Cross-check the config's declared geometry against the layout above.

    Called from `config.py`, so a yaml that says 320x512 or two cameras fails at load
    time rather than producing a differently-shaped batch several minutes into a run.
    """
    if tuple(video_size) != CANVAS_HW:
        raise ValueError(
            f"data video_size {tuple(video_size)} does not match the ported RoboTwin canvas "
            f"{CANVAS_HW}; only the official 3-camera board is implemented"
        )
    num_output_cameras = len(layout) if not isinstance(layout, (int, np.integer)) else int(layout)
    if num_output_cameras < 1:
        raise ValueError(
            f"num_output_cameras={num_output_cameras}; a canvas needs at least one camera"
        )


# --------------------------------------------------------------------------- #
# JPEG decode — one chokepoint
# --------------------------------------------------------------------------- #
def _pick_decoder():
    """The only JPEG decoder in the repo. A second one would be a second set of pixels.

    simplejpeg is ~2x PIL (0.27 vs 0.52 ms per 240x320 frame, measured on v4-8) and the
    two disagree by up to 2/255 per pixel because libjpeg's IDCT rounds differently, so
    they must never be mixed inside one experiment. The active name is recorded in the
    run's data spec.
    """
    try:
        import simplejpeg

        return "simplejpeg", lambda b: simplejpeg.decode_jpeg(b, colorspace="RGB")
    except ImportError:
        import io

        from PIL import Image

        return "PIL", lambda b: np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)


decoder_name, _decode = _pick_decoder()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes as uint8 RGB `[h, w, 3]`."""
    return _decode(bytes(data))


# --------------------------------------------------------------------------- #
# resize
# --------------------------------------------------------------------------- #
def _aa_bilinear_taps(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """`(index[out, k], weight[out, k])` for antialiased bilinear, torch's exact weights.

    Transcribed from ATen's `_compute_weights_aa` (`UpSampleKernel.cpp`), which is also
    what PIL's `ImagingResampleHorizontal` computes:

        scale    = in / out                       (align_corners=False, no given scale)
        support  = max(scale, 1.0)                (triangle filter, half-support 1)
        invscale = 1/scale if scale > 1 else 1
        centre   = scale * (i + 0.5)
        first    = max(int(centre - support + 0.5), 0)
        last     = min(int(centre + support + 0.5), in)
        w[j]     = tri((first + j - centre + 0.5) * invscale),  then normalised to sum 1

    Rows are right-padded to a common width by repeating the last index with weight 0,
    so the caller can run a fixed number of gathers.
    """
    scale = in_size / out_size
    support = max(scale, 1.0)
    invscale = 1.0 / scale if scale > 1.0 else 1.0
    rows = []
    for i in range(out_size):
        centre = scale * (i + 0.5)
        first = max(int(centre - support + 0.5), 0)
        last = min(int(centre + support + 0.5), in_size)
        offsets = np.arange(first, last, dtype=np.int64)
        w = 1.0 - np.abs((offsets - centre + 0.5) * invscale)
        w = np.where(w > 0.0, w, 0.0)
        total = w.sum()
        if total <= 0.0:  # unreachable for a triangle filter; a silent 0 row would not be
            raise AssertionError(f"degenerate resize weights at output row {i} ({in_size}->{out_size})")
        rows.append((offsets, w / total))
    k = max(len(o) for o, _ in rows)
    idx = np.zeros((out_size, k), np.int64)
    weight = np.zeros((out_size, k), np.float64)
    for i, (offsets, w) in enumerate(rows):
        idx[i, : len(offsets)] = offsets
        idx[i, len(offsets) :] = offsets[-1]
        weight[i, : len(w)] = w
    return idx, weight.astype(np.float32)


_TAP_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _taps(in_size: int, out_size: int):
    key = (in_size, out_size)
    if key not in _TAP_CACHE:
        _TAP_CACHE[key] = _aa_bilinear_taps(in_size, out_size)
    return _TAP_CACHE[key]


def _resize_axis(x: np.ndarray, axis: int, out_size: int) -> np.ndarray:
    if x.shape[axis] == out_size:
        return x
    idx, weight = _taps(x.shape[axis], out_size)
    shape = list(x.shape)
    shape[axis] = out_size
    out = np.zeros(shape, np.float32)
    wshape = [1] * x.ndim
    wshape[axis] = out_size
    for j in range(idx.shape[1]):
        # `np.take(..., out=)` looks like the frugal version but measures ~35% slower —
        # it drops out of the fast path — so the temporaries stay.
        out += np.take(x, idx[:, j], axis=axis) * weight[:, j].reshape(wshape)
    return out


def resize_bilinear_aa(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """float32 `[..., h, w, c]` -> `[..., out_h, out_w, c]`, torch's antialiased bilinear.

    Height then width. The filter is separable, so the order is mathematically
    irrelevant and only moves float rounding by an ulp or so; height goes first because
    gathering along it copies contiguous rows, which is several times cheaper than the
    strided gather along width, and doing it first shrinks the array the width pass has
    to walk.
    """
    if frames.dtype != np.float32:
        raise ValueError(f"resize expects float32 pixels, got {frames.dtype}")
    if frames.ndim < 3:
        raise ValueError(f"resize expects [..., h, w, c], got shape {frames.shape}")
    x = _resize_axis(frames, frames.ndim - 3, out_h)
    return _resize_axis(x, x.ndim - 2, out_w)


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #
def compose_canvas(streams: dict[str, np.ndarray], *, tiles=TILES) -> np.ndarray:
    """Compose uint8 RGB camera streams into a float32 canvas in [-1, 1]."""
    tiles = resolve_tiles(tiles)
    missing = [role for role, _ in tiles if role not in streams]
    if missing:
        raise ValueError(f"canvas is missing camera(s) {missing}; have {sorted(streams)}")

    lengths = {role: streams[role].shape[0] for role, _ in tiles}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"cameras disagree on frame count: {lengths}")
    t = next(iter(lengths.values()))

    canvas = np.empty((t, *CANVAS_HW, 3), np.float32)
    for role, (y0, x0, th, tw) in tiles:
        frames = streams[role]
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"camera {role!r}: expected uint8 [T, h, w, 3], got {frames.shape} {frames.dtype}"
            )
        # /255 before the resize, matching ToTensor -> Resize; the official never resizes
        # uint8, and a uint8 resize would re-quantise between the two separable passes.
        # One multiply rather than astype-then-divide: the frames are ~8 MB apiece and
        # this runs three times per sample on the host's data threads.
        tile = resize_bilinear_aa(np.multiply(frames, np.float32(1.0 / 255.0), dtype=np.float32), th, tw)
        canvas[:, y0 : y0 + th, x0 : x0 + tw] = tile
    # Normalize(mean=0.5, std=0.5): [0,1] -> [-1,1].
    canvas *= 2.0
    canvas -= 1.0
    return canvas
