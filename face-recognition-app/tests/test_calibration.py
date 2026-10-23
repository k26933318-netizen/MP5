import numpy as np

from face_verification.config import Settings
from face_verification.store import FaceStore
from tools.calibrate_from_db import collect_similarity_pairs, threshold_rates
from tools.antispoof_eval import evaluate_rates


def _unit(index):
    vector = np.zeros(512, dtype="float32")
    vector[index] = 1
    return vector


def test_calibration_pairs_and_threshold_target(tmp_path):
    store = FaceStore(Settings(face_data_dir=tmp_path))
    store.add_subject_with_embeddings("a", [_unit(0), _unit(0)], [0.9, 0.9], consent=True)
    store.add_subject_with_embeddings("b", [_unit(1), _unit(1)], [0.9, 0.9], consent=True)

    genuine, impostor = collect_similarity_pairs(store)
    rows, suggested = threshold_rates(genuine, impostor, 0.0)

    assert len(genuine) == 2
    assert len(impostor) == 4
    assert rows[0]["far"] == 0.0
    assert rows[0]["frr"] == 0.0
    assert suggested == 0.2
    store.close()


def test_antispoof_rates_count_undetected_real_as_false_reject():
    rows, suggested = evaluate_rates(
        {"real": [0.9, None], "print": [0.7, 0.1], "screen": [0.7]},
        [0.5, 0.75],
        target_apcer=0.4,
    )

    assert rows[0]["bpcer"] == 0.5
    assert rows[0]["apcer"] == {"print": 0.5, "screen": 1.0}
    assert rows[1]["bpcer"] == 0.5
    assert rows[1]["apcer"] == {"print": 0.0, "screen": 0.0}
    assert suggested == 0.75