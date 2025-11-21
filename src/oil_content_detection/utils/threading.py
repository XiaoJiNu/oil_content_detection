"""Threading configuration utilities.

This module provides utilities to configure thread settings for numerical libraries
to avoid conflicts during parallel operations like cross-validation in genetic algorithms.
"""
import os


def setup_single_thread() -> None:
    """Configure numerical libraries to use single-threaded execution.

    This prevents thread-related conflicts when using multiprocessing or when
    libraries like OpenBLAS, MKL, and BLAS compete for resources during
    cross-validation in sklearn.

    Should be called before importing numpy, sklearn, or other numerical libraries
    that use BLAS/LAPACK backends.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


__all__ = ["setup_single_thread"]