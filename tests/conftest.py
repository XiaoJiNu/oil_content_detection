"""Global pytest configuration for the project."""
from oil_content_detection.utils.threading import setup_single_thread

# Avoid BLAS over-subscription during test runs
setup_single_thread()
