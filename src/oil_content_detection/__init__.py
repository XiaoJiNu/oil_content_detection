"""Oil content detection package."""
from oil_content_detection.utils.threading import setup_single_thread

# Enforce single-threaded BLAS/OpenMP to stay compatible with restricted environments
setup_single_thread()
