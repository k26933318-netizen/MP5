"""Calibrate 1:1 face thresholds from stored embeddings; no Flask or image access."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_verification.config import load_settings
from face_verification.store import FaceStore

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.05
MIN_IMPOSTOR_PAIRS_FACTOR = 3


def collect_similarity_pairs(store: FaceStore) -> tuple[list[float], list[float]]:
    """Return genuine within-subject and impostor cross-subject cosine scores."""
    subjects = [subject for subject in store.list_subjects(limit=500000) if subject["status"] == "active"]
    vectors = {
        subject["subject_id"]: [sample.vector for sample in store.get_embeddings(subject["subject_id"])]
        for subject in subjects
    }
    genuine: list[float] = []
    impostor: list[float] = []
    subject_ids = sorted(vectors)
    for subject_id in subject_ids:
        genuine.extend(
            float(np.dot(first, second))
            for first, second in combinations(vectors[subject_id], 2)
        )
    for first_id, second_id in combinations(subject_ids, 2):
        impostor.extend(
            float(np.dot(first, second))
            for first in vectors[first_id]
            for second in vectors[second_id]
        )
    return genuine, impostor


def similarity_statistics(values: list[float]) -> dict[str, float | int] | None:
    """Summarize a score distribution with count, extrema, mean and percentiles."""
    if not values:
        return None
    scores = np.asarray(values, dtype="float64")
    percentiles = np.percentile(scores, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(scores.size),
        "mean": float(scores.mean()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def threshold_rates(
    genuine: list[float],
    impostor: list[float],
    target_far: float,
) -> tuple[list[dict[str, float]], float | None]:
    """Compute FAR/FRR rows and return the lowest threshold meeting target FAR."""
    rows = []
    for threshold in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP):
        far = float(np.mean(np.asarray(impostor) >= threshold)) if impostor else float("nan")
        frr = float(np.mean(np.asarray(genuine) < threshold)) if genuine else float("nan")
        rows.append({"threshold": round(float(threshold), 2), "far": far, "frr": frr})
    eligible = [row for row in rows if np.isfinite(row["far"]) and row["far"] <= target_far]
    return rows, (min(eligible, key=lambda row: row["threshold"])["threshold"] if eligible else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-far", type=float, default=float(os.getenv("TARGET_FAR", "1e-3")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.target_far <= 1.0:
        parser.error("--target-far must be between 0 and 1.")

    settings = load_settings()
    store = FaceStore(settings)
    try:
        genuine, impostor = collect_similarity_pairs(store)
        print("Genuine pairs:", json.dumps(similarity_statistics(genuine), sort_keys=True))
        print("Impostor pairs:", json.dumps(similarity_statistics(impostor), sort_keys=True))
        if not genuine:
            warnings.warn("No genuine pairs; enroll at least two samples for one subject.", RuntimeWarning)
        minimum_impostor_count = MIN_IMPOSTOR_PAIRS_FACTOR / max(args.target_far, 1e-12)
        if len(impostor) < minimum_impostor_count:
            warnings.warn(
                f"Only {len(impostor)} impostor pairs; at least {minimum_impostor_count:.0f} are recommended for target FAR {args.target_far:g}.",
                RuntimeWarning,
            )
        rows, suggested = threshold_rates(genuine, impostor, args.target_far)
        print("threshold FAR FRR")
        for row in rows:
            print(f"{row['threshold']:.2f} {row['far']:.6f} {row['frr']:.6f}")
        if suggested is None:
            print(f"No tested threshold meets target FAR {args.target_far:g}.")
            return 0
        selected = next(row for row in rows if row["threshold"] == suggested)
        output_path = args.output or settings.face_data_dir / "threshold.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "verify_threshold": suggested,
                    "target_far": args.target_far,
                    "false_accept_rate": selected["far"],
                    "false_reject_rate": selected["frr"],
                    "genuine_pairs": len(genuine),
                    "impostor_pairs": len(impostor),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Suggested VERIFY_THRESHOLD={suggested:.2f}; wrote {output_path}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())