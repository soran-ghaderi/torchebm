import warnings

import pytest
import torch
from torch import nn

from torchebm.core.base_module import TorchEBMModule, _normalize
from torchebm.core.device_mixin import DeviceMixin, normalize_device, safe_to

from tests.conftest import requires_cuda


class _Empty(TorchEBMModule):
    pass


class _Linear(TorchEBMModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = nn.Linear(4, 4)


def test_torchebm_module_default_device_is_cpu():
    m = _Empty()
    assert m.device.type == "cpu"
    assert m.dtype == torch.get_default_dtype()


def test_torchebm_module_with_dtype_override():
    m = _Empty(dtype=torch.float64)
    assert m.dtype == torch.float64


def test_torchebm_module_device_from_parameters():
    m = _Linear()
    assert m.device.type == "cpu"
    assert m.dtype == torch.float32


def test_torchebm_module_double_cast_updates_cache():
    m = _Linear()
    m.double()
    assert m.dtype == torch.float64


def test_normalize_cuda_index_zero_collapses():
    assert _normalize(torch.device("cuda:0")) == torch.device("cuda")


def test_normalize_cpu_passthrough():
    assert _normalize(torch.device("cpu")) == torch.device("cpu")


def test_setup_mixed_precision_disabled_is_noop():
    m = _Empty()
    m.setup_mixed_precision(False)
    assert not m.use_mixed_precision
    assert not m.autocast_available


def test_setup_mixed_precision_cpu_warns_and_disables():
    m = _Empty()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.setup_mixed_precision(True)
    assert any(issubclass(x.category, UserWarning) for x in w)
    assert not m.use_mixed_precision


def test_autocast_context_is_nullcontext_on_cpu():
    m = _Empty()
    m.setup_mixed_precision(True)
    ctx = m.autocast_context()
    with ctx:
        x = torch.randn(3)
    assert x.dtype == torch.float32


# device_mixin tests


def test_normalize_device_none_returns_cpu_or_cuda():
    dev = normalize_device(None)
    assert dev.type in {"cpu", "cuda"}


def test_normalize_device_str_coerced():
    assert normalize_device("cpu") == torch.device("cpu")


@requires_cuda
def test_normalize_device_cuda_current_index_collapsed():
    dev = normalize_device(torch.device(f"cuda:{torch.cuda.current_device()}"))
    assert dev.index is None


def test_safe_to_no_device_no_dtype_passthrough():
    x = torch.randn(2)
    assert safe_to(x, None, None) is x


def test_safe_to_non_movable_object_passthrough():
    obj = object()
    assert safe_to(obj, device=torch.device("cpu")) is obj


def test_safe_to_moves_dtype():
    x = torch.randn(2)
    out = safe_to(x, dtype=torch.float64)
    assert out.dtype == torch.float64


class _MixinModel(DeviceMixin, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin = nn.Linear(3, 3)


def _mk_mixin():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _MixinModel()


def test_device_mixin_emits_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _MixinModel()
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_device_mixin_dtype_falls_back_to_parameters():
    m = _mk_mixin()
    assert m.dtype == torch.float32


def test_device_mixin_dtype_setter():
    m = _mk_mixin()
    m.dtype = torch.float64
    assert m.dtype == torch.float64


def test_device_mixin_to_updates_cache():
    m = _mk_mixin()
    m.to(dtype=torch.float64)
    assert m.dtype == torch.float64
