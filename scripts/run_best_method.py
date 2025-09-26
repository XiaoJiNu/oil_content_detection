#!/usr/bin/env python3
"""Execute the GA + PLSR pipeline on the simulated spectral dataset."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from oil_content_detection.models.plsr_best import RunConfig, run_and_print


if __name__ == "__main__":
    run_and_print(RunConfig())
