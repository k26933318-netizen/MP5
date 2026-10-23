"""Persistent FAISS inner-product index and name mapping."""

import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np


# Beyond roughly 50k faces, consider IndexHNSWFlat or IVF, plus IndexIDMap for deletions.
class VectorStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index_path = self.directory / "faces.index"
        self.mapping_path = self.directory / "names.json"
        self.index = None
        self.names = []
        self._lock = threading.RLock()
        self.load()

    def add(self, embedding, name):
        import faiss

        vector = self._normalized_vector(embedding)
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(vector.shape[1])
            self.index.add(vector)
            self.names.append(name)

    def search(self, embedding, top_k=1):
        del top_k
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                raise ValueError("The face database is empty. Register a person first.")
            scores, positions = self.index.search(self._normalized_vector(embedding), 1)
            position = int(positions[0][0])
            return self.names[position], float(scores[0][0])

    def search_with_runner_up(self, embedding, k=10):
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                raise ValueError("The face database is empty. Register a person first.")

            vector = self._normalized_vector(embedding)
            search_k = min(max(1, int(k)), self.index.ntotal)
            while True:
                scores, positions = self.index.search(vector, search_k)
                best_position = int(positions[0][0])
                best_name = self.names[best_position]
                best_score = float(scores[0][0])
                runner_up_score = next(
                    (
                        float(score)
                        for score, position in zip(scores[0], positions[0])
                        if int(position) >= 0 and self.names[int(position)] != best_name
                    ),
                    None,
                )
                if runner_up_score is not None or search_k == self.index.ntotal:
                    return best_name, best_score, runner_up_score
                search_k = min(self.index.ntotal, max(search_k + 1, search_k * 2))

    def save(self):
        with self._lock:
            if self.index is None:
                return
            import faiss

            self.directory.mkdir(parents=True, exist_ok=True)
            temporary_paths = []
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as index_file:
                    temporary_paths.append(Path(index_file.name))
                with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as mapping_file:
                    temporary_paths.append(Path(mapping_file.name))

                faiss.write_index(self.index, str(temporary_paths[0]))
                temporary_paths[1].write_text(json.dumps(self.names, indent=2), encoding="utf-8")
                os.replace(temporary_paths[0], self.index_path)
                os.replace(temporary_paths[1], self.mapping_path)
            finally:
                for temporary_path in temporary_paths:
                    temporary_path.unlink(missing_ok=True)

    def load(self):
        with self._lock:
            index_exists = self.index_path.exists()
            mapping_exists = self.mapping_path.exists()
            if index_exists != mapping_exists:
                raise ValueError("The FAISS index and name mapping must both exist or both be absent.")
            if not index_exists:
                return
            import faiss

            self.index = faiss.read_index(str(self.index_path))
            self.names = json.loads(self.mapping_path.read_text(encoding="utf-8"))
            if self.index.ntotal != len(self.names):
                raise ValueError("The FAISS index and name mapping are out of sync.")

    @staticmethod
    def _normalized_vector(embedding):
        vector = np.asarray(embedding, dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        return vector / np.maximum(norm, 1e-12)