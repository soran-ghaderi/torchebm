import pytest
import torch

from torchebm._deprecation import _WARNED_ONCE


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """Clear the once-per-process warning dedup before every test, so
    pytest.warns assertions never depend on test order."""
    _WARNED_ONCE.clear()
    yield


def requires_cuda(func):
    """Skip test if CUDA is not available or no NVIDIA driver is found.
    
    This decorator can be used to mark tests that require CUDA and
    should be skipped when CUDA is not available or no NVIDIA driver
    is installed.
    """
    reason = "Test requires CUDA with NVIDIA driver"
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        # Create a decorator that will skip the test
        decorator = pytest.mark.skip(reason=reason)
        return decorator(func)
    
    # Even if torch.cuda.is_available() returns True, we might still encounter
    # an error when trying to use CUDA due to missing NVIDIA driver
    try:
        # Try to initialize a CUDA tensor
        torch.zeros(1, device="cuda")
        # If no exception, CUDA is working
    except RuntimeError as e:
        # Check if the error message contains the NVIDIA driver message
        if "Found no NVIDIA driver" in str(e):
            decorator = pytest.mark.skip(reason=reason)
            return decorator(func)
    
    # If we get here, CUDA is available and working
    return func 