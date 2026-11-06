r"""Triton kernel for the JVP of scaled dot-product attention.

For \(O = \mathrm{softmax}(s\, Q K^\top) V\) with input tangents
\(\dot Q, \dot K, \dot V\), the output tangent is (sCM, Lu & Song, 2025,
Appendix F)

\[
\dot O = P \dot V + (P \odot \dot S) V
- \mathrm{rowsum}(P \odot \dot S) \odot O,
\qquad \dot S = s\,(\dot Q K^\top + Q \dot K^\top),
\]

with \(P = \mathrm{softmax}(s\, Q K^\top)\). The kernel streams over key
blocks with an online softmax, as in the rCM FlashAttention-2 JVP kernel
(Zheng et al., 2025), so \(P\) is never stored and memory is linear in the
sequence length. \(O\) is read from the forward pass.

When every stride is a multiple of 16 bytes, offsets carry that alignment
as a hint, so loads stay vectorized and asynchronous for head dimensions
such as 72 whose strides Triton cannot prove aligned on its own.
"""

import torch
import triton
import triton.language as tl

_LOG2E = tl.constexpr(1.4426950408889634)

_CONFIGS = [
    triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_warps=warps, num_stages=stages)
    for block_m, block_n, warps, stages in (
        (128, 32, 4, 1),
        (128, 32, 8, 2),
        (128, 64, 8, 2),
        (128, 32, 8, 3),
        (128, 128, 8, 3),
        (64, 32, 8, 1),
        (64, 32, 4, 2),
    )
]


@triton.jit
def _block_ptrs(X, b, h, offs_row, offs_d, stride_b, stride_h, stride_n, ALIGN: tl.constexpr):
    base = tl.multiple_of(b * stride_b + h * stride_h, ALIGN)
    rows = tl.multiple_of(offs_row * stride_n, ALIGN)
    return X + base + rows[:, None] + offs_d[None, :]


@triton.autotune(configs=_CONFIGS, key=["N_Q", "N_KV", "HEAD_DIM", "ALIGN"])
@triton.jit
def _attention_jvp_kernel(
    Q, K, V, O, TQ, TK, TV, TO,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    stride_tqb, stride_tqh, stride_tqn,
    stride_tkb, stride_tkh, stride_tkn,
    stride_tvb, stride_tvh, stride_tvn,
    stride_tob, stride_toh, stride_ton,
    H, N_Q, N_KV, scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALIGN: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(N_Q, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    mask_md = (offs_m < N_Q)[:, None] & mask_d[None, :]

    q_ptrs = _block_ptrs(Q, b, h, offs_m, offs_d, stride_qb, stride_qh, stride_qn, ALIGN)
    tq_ptrs = _block_ptrs(TQ, b, h, offs_m, offs_d, stride_tqb, stride_tqh, stride_tqn, ALIGN)
    q = tl.load(q_ptrs, mask=mask_md, other=0.0)
    tq = tl.load(tq_ptrs, mask=mask_md, other=0.0)

    k_ptrs = _block_ptrs(K, b, h, offs_n, offs_d, stride_kb, stride_kh, stride_kn, ALIGN)
    tk_ptrs = _block_ptrs(TK, b, h, offs_n, offs_d, stride_tkb, stride_tkh, stride_tkn, ALIGN)
    v_ptrs = _block_ptrs(V, b, h, offs_n, offs_d, stride_vb, stride_vh, stride_vn, ALIGN)
    tv_ptrs = _block_ptrs(TV, b, h, offs_n, offs_d, stride_tvb, stride_tvh, stride_tvn, ALIGN)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    r_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for start_n in range(0, N_KV, BLOCK_N):
        mask_n = start_n + offs_n < N_KV
        mask_nd = mask_n[:, None] & mask_d[None, :]
        k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_nd, other=0.0)
        tk = tl.load(tk_ptrs + start_n * stride_tkn, mask=mask_nd, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_nd, other=0.0)
        tv = tl.load(tv_ptrs + start_n * stride_tvn, mask=mask_nd, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * (scale * _LOG2E)
        s = tl.where(mask_n[None, :], s, float("-inf"))
        ts = (
            tl.dot(tq, tl.trans(k), input_precision=INPUT_PRECISION)
            + tl.dot(q, tl.trans(tk), input_precision=INPUT_PRECISION)
        ) * scale

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        p_ts = p * ts
        l_i = l_i * alpha + tl.sum(p, 1)
        r_i = r_i * alpha + tl.sum(p_ts, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tv.dtype), tv, acc, input_precision=INPUT_PRECISION)
        acc = tl.dot(p_ts.to(v.dtype), v, acc, input_precision=INPUT_PRECISION)
        m_i = m_new

    o_ptrs = _block_ptrs(O, b, h, offs_m, offs_d, stride_ob, stride_oh, stride_on, ALIGN)
    to_ptrs = _block_ptrs(TO, b, h, offs_m, offs_d, stride_tob, stride_toh, stride_ton, ALIGN)
    o = tl.load(o_ptrs, mask=mask_md, other=0.0)
    to = acc / l_i[:, None] - (r_i / l_i)[:, None] * o.to(tl.float32)
    tl.store(to_ptrs, to.to(TO.dtype.element_ty), mask=mask_md)


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

    Args:
        query: Queries of shape ``(B, H, N_q, D)``, CUDA, float16, bfloat16
            or float32.
        key: Keys of shape ``(B, H, N_kv, D)``.
        value: Values of shape ``(B, H, N_kv, D)``.
        output: Attention output for ``query``, ``key``, ``value``.
        query_t: Tangent of ``query``.
        key_t: Tangent of ``key``.
        value_t: Tangent of ``value``.
        scale: Scaling of the logits.

    Returns:
        torch.Tensor: Tangent of ``output``, with the layout of ``output``.
        Float32 products use TF32 unless
        ``torch.get_float32_matmul_precision()`` is ``"highest"``, as for
        PyTorch matmuls.
    """
    tensors = [
        x if x.stride(-1) == 1 else x.contiguous()
        for x in (query, key, value, output, query_t, key_t, value_t)
    ]
    output_t = torch.empty_like(tensors[3])
    batch, heads, n_q, head_dim = query.shape
    n_kv = key.shape[2]
    itemsize = query.element_size()
    aligned = all(stride * itemsize % 16 == 0 for x in (*tensors, output_t) for stride in x.stride()[:3])

    def grid(meta):
        return (triton.cdiv(n_q, meta["BLOCK_M"]) * batch * heads,)

    _attention_jvp_kernel[grid](
        *tensors,
        output_t,
        *(stride for x in (*tensors, output_t) for stride in x.stride()[:3]),
        heads,
        n_q,
        n_kv,
        scale,
        HEAD_DIM=head_dim,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        ALIGN=16 // itemsize if aligned else 1,
        INPUT_PRECISION="ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32",
    )
    return output_t
