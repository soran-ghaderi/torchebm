import math
from typing import Optional

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from torchebm.cuda.attention import attention_jvp as _attention_jvp_triton
except ImportError:
    _attention_jvp_triton = None

_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TRITON_MAX_HEAD_BYTES = 256
_COMPILED_JVP_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.MATH,
]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    r"""`F.scaled_dot_product_attention` with a forward-mode derivative.

    Without forward-mode tangents on the inputs this is
    `F.scaled_dot_product_attention`, so the forward and backward passes use
    the fused kernels (flash, memory-efficient, cuDNN). These kernels have no
    forward-mode derivative, so under `torch.func.jvp` or
    `torch.autograd.forward_ad` the output is still computed by
    `F.scaled_dot_product_attention` and its tangent by the JVP of attention
    (sCM, Lu & Song, 2025, Appendix F):

    \[
    \dot O = P \dot V + (P \odot \dot S) V
    - \mathrm{rowsum}(P \odot \dot S) \odot O,
    \qquad \dot S = s\,(\dot Q K^\top + Q \dot K^\top),
    \]

    with \(P = \mathrm{softmax}(s\, Q K^\top)\). On CUDA the tangent comes
    from a Triton kernel with an online softmax, as in the rCM
    FlashAttention-2 JVP kernel (Zheng et al., 2025), and on Hopper from a
    warp-specialized kernel with the FlashAttention-3 design (Shah et al.,
    2024), so \(P\) is never stored; elsewhere it is computed with dense
    matrix products. In eager mode the backward pass of the output
    recomputes the forward with `F.scaled_dot_product_attention`. Under `torch.compile` the output uses
    the flash, cuDNN or math kernel with its own backward; the
    memory-efficient kernel is left out there, as its backward rejects the
    logsumexp layout traced under forward-mode AD. Under autocast the inputs
    are cast to the autocast dtype first, as autocast does for
    `F.scaled_dot_product_attention`.

    The tangent is first-order and carries no gradient: sCM, rCM and MeanFlow
    use it under stop-gradient.

    Args:
        query: Queries of shape ``(B, H, N_q, D)``.
        key: Keys of shape ``(B, H, N_kv, D)``.
        value: Values of shape ``(B, H, N_kv, D)``.
        dropout_p: Dropout probability on the attention weights. Must be 0
            under forward-mode AD.
        scale: Scaling of the logits. ``None`` (default) uses
            \(1/\sqrt{D}\).

    Returns:
        torch.Tensor: Attention output of shape ``(B, H, N_q, D)``.

    Example:
        ```python
        import torch
        from torchebm.models.components import scaled_dot_product_attention

        q, k, v = torch.randn(3, 2, 4, 16, 64).unbind(0)
        q_t, k_t, v_t = torch.randn(3, 2, 4, 16, 64).unbind(0)
        out, out_t = torch.func.jvp(
            scaled_dot_product_attention, (q, k, v), (q_t, k_t, v_t)
        )
        ```
    """
    if all(fwAD.unpack_dual(x).tangent is None for x in (query, key, value)):
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, scale=scale
        )
    if dropout_p > 0.0:
        raise ValueError("dropout_p must be 0 under forward-mode AD")
    device_type = query.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    if not torch.compiler.is_compiling():
        return _ScaledDotProductAttention.apply(query, key, value, scale)
    (query, query_t), (key, key_t), (value, value_t) = (
        fwAD.unpack_dual(x) for x in (query, key, value)
    )
    with sdpa_kernel(_COMPILED_JVP_BACKENDS):
        output = F.scaled_dot_product_attention(query, key, value, scale=scale)
    output_t = _output_tangent(query, key, value, output, query_t, key_t, value_t, scale)
    return fwAD.make_dual(output, output_t)


class _ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(query, key, value, scale):
        return F.scaled_dot_product_attention(query, key, value, scale=scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        query, key, value, scale = inputs
        ctx.scale = scale
        ctx.save_for_backward(query, key, value)
        ctx.save_for_forward(query, key, value, output)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, key, value = (x.detach().requires_grad_() for x in ctx.saved_tensors)
        with torch.enable_grad():
            output = F.scaled_dot_product_attention(query, key, value, scale=ctx.scale)
        return *torch.autograd.grad(output, (query, key, value), grad_output), None

    @staticmethod
    def jvp(ctx, query_t, key_t, value_t, _):
        return _output_tangent(*ctx.saved_tensors, query_t, key_t, value_t, ctx.scale)


def _output_tangent(query, key, value, output, query_t, key_t, value_t, scale):
    query_t, key_t, value_t = (
        torch.zeros_like(x) if x_t is None else x_t
        for x, x_t in zip((query, key, value), (query_t, key_t, value_t))
    )
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    with torch.no_grad():
        return attention_jvp(query, key, value, output, query_t, key_t, value_t, scale)


@torch.library.custom_op("torchebm::attention_jvp", mutates_args=())
def attention_jvp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_t: torch.Tensor,
    key_t: torch.Tensor,
    value_t: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    r"""Output tangent of scaled dot-product attention.

    Runs the Triton kernel of `torchebm.cuda.attention` on CUDA for float16
    and bfloat16 with head dimension up to 128 and float32 up to 64, and
    dense matrix products otherwise.

    Args:
        query: Queries of shape ``(B, H, N_q, D)``.
        key: Keys of shape ``(B, H, N_kv, D)``.
        value: Values of shape ``(B, H, N_kv, D)``.
        output: Attention output for ``query``, ``key``, ``value``.
        query_t: Tangent of ``query``.
        key_t: Tangent of ``key``.
        value_t: Tangent of ``value``.
        scale: Scaling of the logits.

    Returns:
        torch.Tensor: Tangent of ``output``, with the layout of ``output``.
    """
    if (
        _attention_jvp_triton is not None
        and query.is_cuda
        and query.dtype in _TRITON_DTYPES
        and query.shape[-1] == value.shape[-1]
        and query.shape[-1] * query.dtype.itemsize <= _TRITON_MAX_HEAD_BYTES
        and query.shape[:2] == key.shape[:2] == value.shape[:2]
    ):
        return _attention_jvp_triton(
            query, key, value, output, query_t, key_t, value_t, scale
        )
    dtype = torch.promote_types(query.dtype, torch.float32)
    q, k, v, o, q_t, k_t, v_t = (
        x.to(dtype) for x in (query, key, value, output, query_t, key_t, value_t)
    )
    k_transposed = k.transpose(-2, -1)
    p = torch.softmax(q @ k_transposed * scale, dim=-1)
    p_ts = p * ((q_t @ k_transposed + q @ k_t.transpose(-2, -1)) * scale)
    output_t = p @ v_t + p_ts @ v - p_ts.sum(-1, keepdim=True) * o
    return torch.empty_like(output).copy_(output_t)


@attention_jvp.register_fake
def _(query, key, value, output, query_t, key_t, value_t, scale):
    return torch.empty_like(output)
