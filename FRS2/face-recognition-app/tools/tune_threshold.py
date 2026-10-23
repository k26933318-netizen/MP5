import argparse
from itertools import combinations
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_verification.config import load_settings
from face_verification.engine import FaceEngine


def _statistics(label, values):
    if not values:
        print(f"{label}: no pairs available")
        return
    scores = np.asarray(values, dtype="float32")
    percentiles = np.percentile(scores, [5, 25, 50, 75, 95])
    print(
        f"{label}: mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}, "
        f"p05={percentiles[0]:.4f}, p25={percentiles[1]:.4f}, p50={percentiles[2]:.4f}, "
        f"p75={percentiles[3]:.4f}, p95={percentiles[4]:.4f}"
    )


def main():
    default_dataset = Path(__file__).resolve().parents[1] / "dataset"
    parser = argparse.ArgumentParser(description="Evaluate face-match thresholds on a labeled dataset.")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"dataset directory does not exist: {args.dataset}")

    engine = FaceEngine(load_settings())
    embeddings_by_person = {}
    for person_dir in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        person_embeddings = []
        for image_path in sorted(person_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
                continue
            try:
                faces = engine.analyze(image)
                if not faces:
                    raise ValueError("No face detected.")
                person_embeddings.append(engine.embed(faces[0], image))
            except ValueError as error:
                print(f"Skipping {image_path}: {error}", file=sys.stderr)
        if person_embeddings:
            embeddings_by_person[person_dir.name] = person_embeddings

    same_person = []
    different_person = []
    for person, embeddings in embeddings_by_person.items():
        same_person.extend(
            float(np.dot(first, second))
            for first, second in combinations(embeddings, 2)
        )
        for other_person, other_embeddings in embeddings_by_person.items():
            if person >= other_person:
                continue
            different_person.extend(
                float(np.dot(first, second))
                for first in embeddings
                for second in other_embeddings
            )

    _statistics("Same-person similarity", same_person)
    _statistics("Different-person similarity", different_person)
    if not same_person or not different_person:
        print("Threshold rates require at least one same-person and one different-person pair.")
        return 0

    print("threshold false_accept_rate false_reject_rate")
    for threshold in np.arange(0.20, 0.8001, 0.05):
        false_accept_rate = np.mean(np.asarray(different_person) > threshold)
        false_reject_rate = np.mean(np.asarray(same_person) <= threshold)
        print(f"{threshold:.2f} {false_accept_rate:.4f} {false_reject_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())