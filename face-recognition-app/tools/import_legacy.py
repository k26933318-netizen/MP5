"""Explicitly import legacy FAISS data for test use only; never run automatically."""

import argparse
import json
import os
from pathlib import Path
import sys

import faiss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_verification.config import load_settings
from face_verification.store import FaceStore


def main() -> int:
    """Import old index vectors after the operator confirms they are test data."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=root / "database")
    parser.add_argument("--face-data-dir", type=Path, default=root / "face_data")
    parser.add_argument(
        "--confirm-test-data-only",
        action="store_true",
        help="confirm legacy names and embeddings are test data, not production biometric data",
    )
    args = parser.parse_args()
    if not args.confirm_test_data_only:
        parser.error("Legacy import is test-data-only; pass --confirm-test-data-only to proceed.")

    legacy_index_path = args.legacy_dir / "faces.index"
    legacy_names_path = args.legacy_dir / "names.json"
    index = faiss.read_index(str(legacy_index_path))
    names = json.loads(legacy_names_path.read_text(encoding="utf-8"))
    if index.ntotal != len(names):
        parser.error("Legacy FAISS index and names mapping are out of sync.")

    settings = load_settings({**os.environ, "FACE_DATA_DIR": str(args.face_data_dir)})
    store = FaceStore(settings)
    grouped: dict[str, list] = {}
    for position, name in enumerate(names):
        if not isinstance(name, str):
            parser.error("Legacy name entries must be strings usable as subject IDs.")
        grouped.setdefault(name, []).append(index.reconstruct(position))
    for subject_id, vectors in grouped.items():
        store.add_subject_with_embeddings(
            subject_id,
            vectors,
            [0.0] * len(vectors),
            consent=False,
        )
    print(f"Imported {len(grouped)} test-only subjects and {index.ntotal} embeddings.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())