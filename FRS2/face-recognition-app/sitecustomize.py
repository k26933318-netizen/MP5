"""Project-wide Python startup hooks for local development.

This file is imported automatically by Python during startup when the project root
is on sys.path. It keeps BLAS thread counts low so OpenBLAS does not exhaust
memory on constrained machines and loads the repo-local .env values before app
modules import heavy scientific dependencies.
"""

import os
from pathlib import Path


def _load_local_dotenv() -> None:
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _limit_blas_threads() -> None:
    if any(name in os.environ for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        return
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


_load_local_dotenv()
_limit_blas_threads()
