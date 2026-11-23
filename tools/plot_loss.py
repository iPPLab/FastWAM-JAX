#!/usr/bin/env python3
"""Render FastWAM training loss curves from a run directory or metrics file."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


LOG_RECORD = re.compile(
    r"\bstep\s+(?P<step>\d+)(?:/\d+)?\s+"
    r"loss\s+(?P<loss>[0-9.eE+-]+)\s+"
    r"v\s+(?P<loss_video>[0-9.eE+-]+)\s+"
    r"a\s+(?P<loss_action>[0-9.eE+-]+)"
)
METRICS_GLOB = "logs/metrics/checkpoint_step_*.jsonl"
COLORS = {
    "loss": "#2563eb",
    "loss_video": "#f97316",
    "loss_action": "#16a34a",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _jsonl_records(text: str, source: str) -> list[dict[str, float]]:
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            record = {name: float(item[name]) for name in COLORS if name in item}
            record["step"] = int(item["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{source}:{line_no}: invalid metrics record: {error}") from error
        if "loss" in record:
            records.append(record)
    return records


def _log_records(text: str) -> list[dict[str, float]]:
    records = []
    for match in LOG_RECORD.finditer(text):
        records.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "loss_video": float(match.group("loss_video")),
                "loss_action": float(match.group("loss_action")),
            }
        )
    return records


def _read_gcs(uri: str) -> list[dict[str, float]]:
    target = uri.rstrip("/")
    if not target.endswith(".jsonl"):
        target = f"{target}/{METRICS_GLOB}"
    process = subprocess.run(
        ["gcloud", "storage", "cat", target], capture_output=True, text=True, check=False
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"failed to read {target} with gcloud: {detail}")
    return _jsonl_records(process.stdout, target)


def load_records(source: str) -> list[dict[str, float]]:
    if source.startswith("gs://"):
        records = _read_gcs(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        if path.is_file():
            text = path.read_text(errors="replace")
            records = _jsonl_records(text, str(path)) if path.suffix == ".jsonl" else _log_records(text)
        else:
            metric_files = sorted(path.glob(METRICS_GLOB))
            if not metric_files:
                metric_files = sorted(path.rglob("checkpoint_step_*.jsonl"))
            if metric_files:
                records = []
                for item in metric_files:
                    records.extend(_jsonl_records(item.read_text(), str(item)))
            else:
                log_files = sorted(path.rglob("worker_0.log"))
                records = []
                for item in log_files:
                    records.extend(_log_records(item.read_text(errors="replace")))

    # Restarts and checkpoint snapshots can repeat steps; the last record is authoritative.
    by_step = {int(record["step"]): record for record in records}
    result = [by_step[step] for step in sorted(by_step)]
    if not result:
        raise ValueError(f"no FastWAM loss records found under {source}")
    return result


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if math.isclose(low, high):
        return [low]
    rough = (high - low) / count
    power = 10 ** math.floor(math.log10(rough))
    step = min((1, 2, 5, 10), key=lambda factor: abs(factor * power - rough)) * power
    first = math.ceil(low / step) * step
    ticks, value = [], first
    while value <= high + step * 1e-6:
        ticks.append(value)
        value += step
    return ticks


def _format_value(value: float) -> str:
    return f"{value:.2e}" if value and abs(value) < 0.001 else f"{value:.4f}".rstrip("0").rstrip(".")


def _panel(
    draw: ImageDraw.ImageDraw,
    records: list[dict[str, float]],
    box: tuple[int, int, int, int],
    smoothing: int,
    title: str,
    scale: int,
    log_y: bool,
) -> None:
    left, top, right, bottom = (value * scale for value in box)
    steps = [int(item["step"]) for item in records]
    series = {key: [item[key] for item in records if key in item] for key in COLORS}
    series = {key: values for key, values in series.items() if len(values) == len(records)}
    smoothed = {key: moving_average(values, smoothing) for key, values in series.items()}
    values = [value for values in smoothed.values() for value in values]
    if log_y:
        positive = [value for value in values if value > 0]
        if not positive:
            raise ValueError("log-scale plot requires at least one positive loss value")
        low, high = min(positive) * 0.8, max(positive) * 1.15
        low_log, high_log = math.log10(low), math.log10(high)
        ticks = []
        for exponent in range(math.floor(low_log), math.ceil(high_log) + 1):
            for multiplier in (1, 2, 5):
                value = multiplier * 10**exponent
                if low <= value <= high:
                    ticks.append(value)
    else:
        low = 0.0
        high = max(values) * 1.08
        ticks = _nice_ticks(low, high)

    draw.text((left, top - 43 * scale), title, fill="#111827", font=_font(25 * scale, bold=True))
    draw.rectangle((left, top, right, bottom), fill="#ffffff", outline="#cbd5e1", width=scale)

    def value_y(value: float) -> float:
        if log_y:
            value = max(value, low)
            fraction = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
        else:
            fraction = (value - low) / (high - low)
        return bottom - fraction * (bottom - top)

    for tick in ticks:
        y = value_y(tick)
        draw.line((left, y, right, y), fill="#e2e8f0", width=scale)
        label = _format_value(tick)
        draw.text(
            (left - 16 * scale, y), label, fill="#64748b", font=_font(16 * scale), anchor="rm"
        )

    def points(values: list[float]) -> list[tuple[float, float]]:
        width = max(steps[-1] - steps[0], 1)
        return [
            (
                left + (step - steps[0]) / width * (right - left),
                value_y(value),
            )
            for step, value in zip(steps, values)
        ]

    for key, values in smoothed.items():
        draw.line(points(values), fill=COLORS[key], width=2 * scale, joint="curve")

    for index in range(6):
        fraction = index / 5
        x = left + fraction * (right - left)
        step = round(steps[0] + fraction * (steps[-1] - steps[0]))
        draw.text(
            (x, bottom + 10 * scale),
            f"{step:,}",
            fill="#64748b",
            font=_font(16 * scale),
            anchor="ma",
        )
    draw.text(
        ((left + right) / 2, bottom + 42 * scale),
        "Training step",
        fill="#334155",
        font=_font(17 * scale),
        anchor="ma",
    )

    legend_x = right
    for key in reversed(list(series)):
        label = key.removeprefix("loss_").replace("loss", "total")
        font = _font(17 * scale)
        legend_x -= draw.textlength(label, font=font) + 48 * scale
        draw.line(
            (legend_x, top - 27 * scale, legend_x + 27 * scale, top - 27 * scale),
            fill=COLORS[key],
            width=3 * scale,
        )
        draw.text(
            (legend_x + 34 * scale, top - 27 * scale),
            label,
            fill="#334155",
            font=font,
            anchor="lm",
        )


def render(records: list[dict[str, float]], output: Path, title: str, smoothing: int) -> None:
    scale = 3
    image = Image.new("RGB", (1600 * scale, 1120 * scale), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((100 * scale, 42 * scale), title, fill="#0f172a", font=_font(38 * scale, bold=True))
    subtitle = f"{len(records):,} records · steps {int(records[0]['step']):,}–{int(records[-1]['step']):,}"
    if smoothing > 1:
        subtitle += f" · moving average {smoothing}"
    draw.text((100 * scale, 92 * scale), subtitle, fill="#64748b", font=_font(19 * scale))
    _panel(draw, records, (150, 180, 1500, 550), smoothing, "Loss · linear scale", scale, False)
    _panel(draw, records, (150, 700, 1500, 1030), smoothing, "Loss · log scale", scale, True)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((1600, 1120), Image.Resampling.LANCZOS).save(output, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local run/file path or gs:// run URI")
    parser.add_argument("--output", "-o", type=Path, help="PNG path (default: outputs/<run>-loss.png)")
    parser.add_argument("--title", help="plot title (default: source basename)")
    parser.add_argument("--smooth", type=int, default=15, help="moving-average window (default: 15)")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")
    run_name = args.source.rstrip("/").rsplit("/", 1)[-1].removesuffix(".jsonl")
    output = args.output or Path("outputs") / f"{run_name}-loss.png"
    try:
        records = load_records(args.source)
        render(records, output, args.title or run_name, args.smooth)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"wrote {output} ({len(records)} records, steps {records[0]['step']}–{records[-1]['step']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
