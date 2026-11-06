import pytest
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch.autograd.function import BackwardCFunction
from torch.nn.attention import SDPBackend, sdpa_kernel

from torchebm.models.components import scaled_dot_product_attention

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _bhnd(n, head_dim, **kwargs):
    return torch.randn(2, n, 3, head_dim, **kwargs).transpose(1, 2)


def _inputs(n_q=7, n_kv=5, head_dim=8, dtype=torch.float64, device="cpu"):
    primals = tuple(_bhnd(n, head_dim, dtype=dtype, device=device) for n in (n_q, n_kv, n_kv))
    return primals, tuple(torch.randn_like(x) for x in primals)


def _math_jvp(primals, tangents):
    with sdpa_kernel(SDPBackend.MATH):
        return torch.func.jvp(F.scaled_dot_product_attention, primals, tangents)


def test_plain_call_is_sdpa():
    torch.manual_seed(0)
    primals, _ = _inputs()
    primals = tuple(x.requires_grad_() for x in primals)
    out = scaled_dot_product_attention(*primals)
    assert torch.equal(out, F.scaled_dot_product_attention(*primals))
    assert not isinstance(out.grad_fn, BackwardCFunction)


def test_jvp_matches_math_sdpa():
    torch.manual_seed(0)
    primals, tangents = _inputs()
    out, out_t = torch.func.jvp(scaled_dot_product_attention, primals, tangents)
    _, ref_t = _math_jvp(primals, tangents)
    assert torch.equal(out, F.scaled_dot_product_attention(*primals))
    assert torch.allclose(out_t, ref_t, atol=1e-12)


def test_jvp_matches_finite_differences():
    torch.manual_seed(0)
    primals, tangents = _inputs()
    _, out_t = torch.func.jvp(scaled_dot_product_attention, primals, tangents)
    eps = 1e-6
    plus = F.scaled_dot_product_attention(*(x + eps * t for x, t in zip(primals, tangents)))
    minus = F.scaled_dot_product_attention(*(x - eps * t for x, t in zip(primals, tangents)))
    assert torch.allclose(out_t, (plus - minus) / (2 * eps), atol=1e-8)


def test_forward_ad_matches_func_jvp():
    torch.manual_seed(0)
    primals, tangents = _inputs()
    with fwAD.dual_level():
        duals = (fwAD.make_dual(x, t) for x, t in zip(primals, tangents))
        out_t = fwAD.unpack_dual(scaled_dot_product_attention(*duals)).tangent
    _, ref_t = _math_jvp(primals, tangents)
    assert torch.allclose(out_t, ref_t, atol=1e-12)


def test_jvp_with_one_tangent_matches_math_sdpa():
    torch.manual_seed(0)
    primals, tangents = _inputs()
    q, k, v = primals
    _, out_t = torch.func.jvp(
        lambda q: scaled_dot_product_attention(q, k, v), (q,), (tangents[0],)
    )
    zeros = tuple(torch.zeros_like(x) for x in primals[1:])
    _, ref_t = _math_jvp(primals, (tangents[0], *zeros))
    assert torch.allclose(out_t, ref_t, atol=1e-12)


def test_backward_through_jvp_output_matches_sdpa():
    torch.manual_seed(0)
    primals, tangents = _inputs()
    primals = tuple(x.requires_grad_() for x in primals)
    out, _ = torch.func.jvp(scaled_dot_product_attention, primals, tangents)
    grads = torch.autograd.grad(out.square().sum(), primals)
    ref = torch.autograd.grad(F.scaled_dot_product_attention(*primals).square().sum(), primals)
    for grad, ref_grad in zip(grads, ref):
        assert torch.allclose(grad, ref_grad, atol=1e-12)


def test_jvp_tangent_carries_no_gradient():
    torch.manual_seed(0)
    primals, tangents = _inputs()
    primals = tuple(x.requires_grad_() for x in primals)
    tangents = tuple(t.requires_grad_() for t in tangents)
    _, out_t = torch.func.jvp(scaled_dot_product_attention, primals, tangents)
    assert not out_t.requires_grad


def test_jvp_rejects_dropout():
    primals, tangents = _inputs()
    with pytest.raises(ValueError, match="dropout_p"):
        torch.func.jvp(
            lambda q, k, v: scaled_dot_product_attention(q, k, v, dropout_p=0.1),
            primals,
            tangents,
        )


def test_jvp_under_autocast_uses_autocast_dtype():
    torch.manual_seed(0)
    primals, tangents = _inputs(dtype=torch.float32)
    with torch.autocast("cpu", torch.bfloat16):
        out, out_t = torch.func.jvp(scaled_dot_product_attention, primals, tangents)
    assert out.dtype == out_t.dtype == torch.bfloat16
    cast = tuple(x.bfloat16().double() for x in primals)
    _, ref_t = _math_jvp(cast, tuple(t.bfloat16().double() for t in tangents))
    assert (out_t.double() - ref_t).norm() / ref_t.norm() < 2e-2


def test_jvp_compiles_without_graph_breaks():
    torch.manual_seed(0)
    primals, tangents = _inputs(dtype=torch.float32)
    compiled = torch.compile(
        lambda p, t: torch.func.jvp(scaled_dot_product_attention, p, t), fullgraph=True
    )
    out, out_t = compiled(primals, tangents)
    ref, ref_t = torch.func.jvp(scaled_dot_product_attention, primals, tangents)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(out_t, ref_t, atol=1e-5)


_TRITON_CASES = [
    (dtype, rtol, n_q, n_kv, head_dim)
    for dtype, rtol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2), (torch.float16, 2e-3))
    for n_q, n_kv, head_dim in (
        (256, 256, 64), (257, 100, 72), (100, 257, 48), (16, 16, 16), (64, 64, 128), (33, 17, 20)
    )
    if head_dim * dtype.itemsize <= 256
]


@requires_cuda
@pytest.mark.usefixtures("highest_matmul_precision")
@pytest.mark.parametrize("dtype, rtol, n_q, n_kv, head_dim", _TRITON_CASES)
def test_triton_jvp_matches_float64_reference(dtype, rtol, n_q, n_kv, head_dim):
    attention = pytest.importorskip("torchebm.cuda.attention")
    torch.manual_seed(0)
    primals, tangents = _inputs(n_q, n_kv, head_dim, dtype=dtype, device="cuda")
    out = F.scaled_dot_product_attention(*primals)
    out_t = attention.attention_jvp(*primals, out, *tangents, head_dim**-0.5)
    _, ref_t = _math_jvp(
        tuple(x.double() for x in primals), tuple(t.double() for t in tangents)
    )
    assert out_t.dtype == dtype and out_t.stride() == torch.empty_like(out).stride()
    assert (out_t.double() - ref_t).norm() / ref_t.norm() < rtol
