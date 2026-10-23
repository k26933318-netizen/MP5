"""Evaluate passive anti-spoof scores on attacks/<type>/*.jpg without Flask."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_verification.config import load_settings
from face_verification.engine import FaceEngine
from face_verification.liveness import AntiSpoofEnsemble

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.05
MIN_GROUP_IMAGES = 50
KNOWN_TYPES = {"real", "print", "screen", "video", "mask"}


def evaluate_rates(
    scores_by_type: dict[str, list[float | None]],
    thresholds: list[float],
    target_apcer: float,
) -> tuple[list[dict[str, object]], float | None]:
    """Compute per-attack APCER, real BPCER, ACER, and target-APCER choice."""
    rows = []
    attack_types = sorted(name for name in scores_by_type if name != "real")
    real_scores = scores_by_type.get("real", [])
    for threshold in thresholds:
        bpcer = float(np.mean([score is None or score < threshold for score in real_scores])) if real_scores else float("nan")
        apcer = {
            kind: float(np.mean([score is not None and score >= threshold for score in scores_by_type[kind]]))
            for kind in attack_types
        }
        acer = (bpcer + float(np.mean(list(apcer.values())))) / 2 if apcer and np.isfinite(bpcer) else float("nan")
        rows.append({"threshold": threshold, "apcer": apcer, "bpcer": bpcer, "acer": acer})
    eligible = [
        row for row in rows
        if row["apcer"] and all(rate <= target_apcer for rate in row["apcer"].values()) and np.isfinite(row["bpcer"])
    ]
    selected = min(eligible, key=lambda row: (row["bpcer"], row["threshold"])) if eligible else None
    return rows, (float(selected["threshold"]) if selected else None)


def _write_plot(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 960, 600
    canvas = np.full((height, width, 3), 255, dtype="uint8")
    left, top, right, bottom = 80, 50, 900, 500
    cv2.line(canvas, (left, top), (left, bottom), (40, 40, 40), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (40, 40, 40), 2)
    for tick in range(0, 11):
        y = bottom - round(tick * (bottom - top) / 10)
        cv2.line(canvas, (left - 5, y), (right, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{tick / 10:.1f}", (25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    series: dict[str, tuple[tuple[int, int, int], list[tuple[float, float]]]] = {}
    attack_types = sorted(rows[0]["apcer"]) if rows else []
    colors = [(190, 80, 35), (35, 120, 190), (80, 150, 50), (170, 50, 160), (20, 150, 150), (80, 80, 80)]
    for index, kind in enumerate(attack_types):
        series[f"APCER {kind}"] = (colors[index % len(colors)], [(float(row["threshold"]), float(row["apcer"][kind])) for row in rows])
    series["BPCER real"] = ((30, 30, 210), [(float(row["threshold"]), float(row["bpcer"])) for row in rows])
    for name, (color, points) in series.items():
        chart_points = [
            (left + round((threshold - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN) * (right - left)),
             bottom - round(rate * (bottom - top)))
            for threshold, rate in points
        ]
        for first, second in zip(chart_points, chart_points[1:]):
            cv2.line(canvas, first, second, color, 2, cv2.LINE_AA)
        legend_y = top + 22 * (list(series).index(name) + 1)
        cv2.line(canvas, (right - 170, legend_y - 5), (right - 145, legend_y - 5), color, 3)
        cv2.putText(canvas, name, (right - 138, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(canvas, "MiniFASNet threshold evaluation", (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
    cv2.putText(canvas, "Threshold", (width // 2 - 40, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write ROC-style plot: {path}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", type=Path, default=root / "attacks")
    parser.add_argument("--target-apcer", type=float, default=float(os.getenv("TARGET_APCER", "0.01")))
    parser.add_argument("--output-json", type=Path, default=root / "database" / "antispoof_threshold.json")
    parser.add_argument("--output-png", type=Path, default=root / "database" / "antispoof_roc.png")
    args = parser.parse_args()
    if not 0.0 <= args.target_apcer <= 1.0:
        parser.error("--target-apcer must be between 0 and 1.")
    if not args.attacks.is_dir():
        parser.error(f"attack directory does not exist: {args.attacks}")

    settings = load_settings()
    antispoof = AntiSpoofEnsemble(settings)
    if not antispoof.configured:
        parser.error("Set both local anti-spoof model paths before evaluating.")
    engine = FaceEngine(settings)
    scores_by_type: dict[str, list[float | None]] = defaultdict(list)
    undetected: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for group_dir in sorted(path for path in args.attacks.iterdir() if path.is_dir()):
        kind = group_dir.name
        if kind not in KNOWN_TYPES:
            warnings.warn(f"Extra attack group '{kind}' will be treated as an attack type.", RuntimeWarning)
        for image_path in sorted(group_dir.glob("*.jpg")):
            counts[kind] += 1
            image = cv2.imread(str(image_path))
            faces = engine.analyze(image) if image is not None else []
            if not faces:
                undetected[kind] += 1
                scores_by_type[kind].append(None)
                continue
            scores_by_type[kind].append(antispoof.is_live(image, faces[0]))

    for kind in sorted(counts):
        print(f"{kind}: images={counts[kind]}, detected={counts[kind] - undetected[kind]}, undetected={undetected[kind]}")
        if counts[kind] < MIN_GROUP_IMAGES:
            warnings.warn(f"Group '{kind}' has only {counts[kind]} images; at least {MIN_GROUP_IMAGES} are recommended.", RuntimeWarning)
    if "real" not in scores_by_type or not any(kind != "real" for kind in scores_by_type):
        parser.error("At least one real group and one attack group are required.")

    thresholds = [round(value, 2) for value in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP)]
    rows, suggested = evaluate_rates(scores_by_type, thresholds, args.target_apcer)
    attack_types = sorted(kind for kind in scores_by_type if kind != "real")
    print("threshold " + " ".join(f"APCER_{kind}" for kind in attack_types) + " BPCER ACER")
    for row in rows:
        apcer_values = " ".join(f"{row['apcer'][kind]:.4f}" for kind in attack_types)
        print(f"{row['threshold']:.2f} {apcer_values} {row['bpcer']:.4f} {row['acer']:.4f}")
    selected = next((row for row in rows if row["threshold"] == suggested), None)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({
            "antispoof_threshold": suggested,
            "target_apcer": args.target_apcer,
            "bpcer": selected["bpcer"] if selected else None,
            "apcer": selected["apcer"] if selected else None,
            "sample_counts": dict(counts),
            "undetected_counts": dict(undetected),
            "thresholds": rows,
        }, indent=2),
        encoding="utf-8",
    )
    if suggested is None:
        print(f"No tested threshold meets target APCER {args.target_apcer:g}.")
    else:
        print(f"Suggested ANTISPOOF_THRESHOLD={suggested:.2f}; wrote {args.output_json}")
    if suggested is None:
        print(f"Wrote evaluation report {args.output_json}")
    _write_plot(rows, args.output_png)
    print(f"Wrote ROC-style plot {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())