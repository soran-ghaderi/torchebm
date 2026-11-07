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

A head dimension that is not a power of two is split into two power-of-two
blocks (72 = 64 + 16) instead of being padded to the next power of two.
Blocks are loaded with TMA descriptors on compute capability 9.0 and newer
when every stride is a multiple of 16 bytes, and with pointers otherwise.
Pointer offsets then carry that alignment as a hint, so loads stay
vectorized for head dimensions such as 72 whose strides Triton cannot prove
aligned on its own. On compute capability 9.0, float16 and bfloat16 inputs
with TMA-aligned strides run the warp-specialized kernel of
`torchebm.cuda.attention_sm90` instead.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

try:
    from torchebm.cuda.attention_sm90 import attention_jvp_sm90 as _attention_jvp_sm90
except ImportError:
    _attention_jvp_sm90 = None

_LOG2E = tl.constexpr(1.4426950408889634)
_SM90_DTYPES = (torch.float16, torch.bfloat16)
_QUERY_ROWS = ("Q", "O", "TQ", "TO", "Q2", "O2", "TQ2", "TO2")
_KEY_ROWS = ("K", "V", "TK", "TV", "K2", "V2", "TK2", "TV2")


def _set_block_shapes(nargs):
    if isinstance(nargs["Q"], TensorDescriptor):
        for names, rows in ((_QUERY_ROWS, nargs["BLOCK_M"]), (_KEY_ROWS, nargs["BLOCK_N"])):
            for name in names:
                nargs[name].block_shape = [1, 1, rows, nargs[name].block_shape[-1]]


_CONFIGS = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n},
        num_warps=warps,
        num_stages=stages,
        pre_hook=_set_block_shapes,
    )
    for block_m, block_n, warps, stages in (
        (128, 32, 4, 1),
        (128, 32, 4, 2),
        (128, 32, 4, 3),
        (128, 32, 8, 2),
        (128, 32, 8, 3),
        (128, 64, 8, 2),
        (128, 64, 8, 3),
        (64, 32, 4, 2),
        (64, 64, 4, 3),
    )
]


@triton.jit
def _tile_ptrs(X, b, h, rows, cols, stride_b, stride_h, stride_n, ALIGN: tl.constexpr):
    base = tl.multiple_of(b.to(tl.int64) * stride_b + h.to(tl.int64) * stride_h, ALIGN)
    return X + base + tl.multiple_of(rows * stride_n, ALIGN)[:, None] + cols[None, :]


@triton.jit
def _load_tile(
    X, b, h, row, col, n_rows, stride_b, stride_h, stride_n,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    ALIGN: tl.constexpr,
    TMA: tl.constexpr,
):
    if TMA:
        tile = X.load([b, h, row, col]).reshape(ROWS, COLS)
    else:
        rows = row + tl.arange(0, ROWS)
        cols = col + tl.arange(0, COLS)
        ptrs = _tile_ptrs(X, b, h, rows, cols, stride_b, stride_h, stride_n, ALIGN)
        if EVEN_ROWS and EVEN_COLS:
            tile = tl.load(ptrs)
        elif EVEN_ROWS:
            tile = tl.load(ptrs, mask=cols[None, :] < HEAD_DIM, other=0.0)
        elif EVEN_COLS:
            tile = tl.load(ptrs, mask=rows[:, None] < n_rows, other=0.0)
        else:
            tile = tl.load(ptrs, mask=(rows[:, None] < n_rows) & (cols[None, :] < HEAD_DIM), other=0.0)
    return tile


@triton.jit
def _store_tile(
    X, tile, b, h, row, col, n_rows, stride_b, stride_h, stride_n,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    ALIGN: tl.constexpr,
    TMA: tl.constexpr,
):
    if TMA:
        X.store([b, h, row, col], tile.to(X.dtype).reshape(1, 1, ROWS, COLS))
    else:
        rows = row + tl.arange(0, ROWS)
        cols = col + tl.arange(0, COLS)
        ptrs = _tile_ptrs(X, b, h, rows, cols, stride_b, stride_h, stride_n, ALIGN)
        tile = tile.to(X.dtype.element_ty)
        if EVEN_ROWS and EVEN_COLS:
            tl.store(ptrs, tile)
        elif EVEN_ROWS:
            tl.store(ptrs, tile, mask=cols[None, :] < HEAD_DIM)
        elif EVEN_COLS:
            tl.store(ptrs, tile, mask=rows[:, None] < n_rows)
        else:
            tl.store(ptrs, tile, mask=(rows[:, None] < n_rows) & (cols[None, :] < HEAD_DIM))


@triton.autotune(configs=_CONFIGS, key=["N_Q", "N_KV", "HEAD_DIM", "ALIGN", "TMA"])
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["N_Q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N_KV"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attention_jvp_kernel(
    Q, K, V, O, TQ, TK, TV, TO,
    Q2, K2, V2, O2, TQ2, TK2, TV2, TO2,
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
    BLOCK_D2: tl.constexpr,
    ALIGN: tl.constexpr,
    TMA: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    SPLIT_D: tl.constexpr = BLOCK_D2 > 0
    EVEN_D: tl.constexpr = BLOCK_D <= HEAD_DIM
    EVEN_D2: tl.constexpr = BLOCK_D + BLOCK_D2 == HEAD_DIM
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(N_Q, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    b = pid_bh // H
    h = pid_bh % H
    start_m = pid_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)

    q = _load_tile(Q, b, h, start_m, 0, N_Q, stride_qb, stride_qh, stride_qn,
                   BLOCK_M, BLOCK_D, HEAD_DIM, EVEN_M, EVEN_D, ALIGN, TMA)
    tq = _load_tile(TQ, b, h, start_m, 0, N_Q, stride_tqb, stride_tqh, stride_tqn,
                    BLOCK_M, BLOCK_D, HEAD_DIM, EVEN_M, EVEN_D, ALIGN, TMA)
    if SPLIT_D:
        q2 = _load_tile(Q2, b, h, start_m, BLOCK_D, N_Q, stride_qb, stride_qh, stride_qn,
                        BLOCK_M, BLOCK_D2, HEAD_DIM, EVEN_M, EVEN_D2, ALIGN, TMA)
        tq2 = _load_tile(TQ2, b, h, start_m, BLOCK_D, N_Q, stride_tqb, stride_tqh, stride_tqn,
                         BLOCK_M, BLOCK_D2, HEAD_DIM, EVEN_M, EVEN_D2, ALIGN, TMA)
        acc2 = tl.zeros([BLOCK_M, BLOCK_D2], tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    r_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    qk_scale = scale * _LOG2E

    for start_n in range(0, N_KV, BLOCK_N):
        k = _load_tile(K, b, h, start_n, 0, N_KV, stride_kb, stride_kh, stride_kn,
                       BLOCK_N, BLOCK_D, HEAD_DIM, EVEN_N, EVEN_D, ALIGN, TMA)
        tk = _load_tile(TK, b, h, start_n, 0, N_KV, stride_tkb, stride_tkh, stride_tkn,
                        BLOCK_N, BLOCK_D, HEAD_DIM, EVEN_N, EVEN_D, ALIGN, TMA)
        s = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        ts = tl.dot(tq, tl.trans(k), input_precision=INPUT_PRECISION)
        ts = tl.dot(q, tl.trans(tk), ts, input_precision=INPUT_PRECISION)
        if SPLIT_D:
            k2 = _load_tile(K2, b, h, start_n, BLOCK_D, N_KV, stride_kb, stride_kh, stride_kn,
                            BLOCK_N, BLOCK_D2, HEAD_DIM, EVEN_N, EVEN_D2, ALIGN, TMA)
            tk2 = _load_tile(TK2, b, h, start_n, BLOCK_D, N_KV, stride_tkb, stride_tkh, stride_tkn,
                             BLOCK_N, BLOCK_D2, HEAD_DIM, EVEN_N, EVEN_D2, ALIGN, TMA)
            s = tl.dot(q2, tl.trans(k2), s, input_precision=INPUT_PRECISION)
            ts = tl.dot(tq2, tl.trans(k2), ts, input_precision=INPUT_PRECISION)
            ts = tl.dot(q2, tl.trans(tk2), ts, input_precision=INPUT_PRECISION)

        s = s * qk_scale
        if not EVEN_N:
            s = tl.where((start_n + offs_n < N_KV)[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        p_ts = p * (ts * scale)
        l_i = l_i * alpha + tl.sum(p, 1)
        r_i = r_i * alpha + tl.sum(p_ts, 1)

        v = _load_tile(V, b, h, start_n, 0, N_KV, stride_vb, stride_vh, stride_vn,
                       BLOCK_N, BLOCK_D, HEAD_DIM, EVEN_N, EVEN_D, ALIGN, TMA)
        tv = _load_tile(TV, b, h, start_n, 0, N_KV, stride_tvb, stride_tvh, stride_tvn,
                        BLOCK_N, BLOCK_D, HEAD_DIM, EVEN_N, EVEN_D, ALIGN, TMA)
        p = p.to(v.dtype)
        p_ts = p_ts.to(v.dtype)
        acc = tl.dot(p, tv, acc * alpha[:, None], input_precision=INPUT_PRECISION)
        acc = tl.dot(p_ts, v, acc, input_precision=INPUT_PRECISION)
        if SPLIT_D:
            v2 = _load_tile(V2, b, h, start_n, BLOCK_D, N_KV, stride_vb, stride_vh, stride_vn,
                            BLOCK_N, BLOCK_D2, HEAD_DIM, EVEN_N, EVEN_D2, ALIGN, TMA)
            tv2 = _load_tile(TV2, b, h, start_n, BLOCK_D, N_KV, stride_tvb, stride_tvh, stride_tvn,
                             BLOCK_N, BLOCK_D2, HEAD_DIM, EVEN_N, EVEN_D2, ALIGN, TMA)
            acc2 = tl.dot(p, tv2, acc2 * alpha[:, None], input_precision=INPUT_PRECISION)
            acc2 = tl.dot(p_ts, v2, acc2, input_precision=INPUT_PRECISION)
        m_i = m_new

    inv_l = 1.0 / l_i
    mu = r_i * inv_l
    o = _load_tile(O, b, h, start_m, 0, N_Q, stride_ob, stride_oh, stride_on,
                   BLOCK_M, BLOCK_D, HEAD_DIM, EVEN_M, EVEN_D, ALIGN, TMA)
    _store_tile(TO, acc * inv_l[:, None] - mu[:, None] * o.to(tl.float32), b, h, start_m, 0, N_Q,
                stride_tob, stride_toh, stride_ton, BLOCK_M, BLOCK_D, HEAD_DIM, EVEN_M, EVEN_D, ALIGN, TMA)
    if SPLIT_D:
        o2 = _load_tile(O2, b, h, start_m, BLOCK_D, N_Q, stride_ob, stride_oh, stride_on,
                        BLOCK_M, BLOCK_D2, HEAD_DIM, EVEN_M, EVEN_D2, ALIGN, TMA)
        _store_tile(TO2, acc2 * inv_l[:, None] - mu[:, None] * o2.to(tl.float32), b, h, start_m, BLOCK_D,
                    N_Q, stride_tob, stride_toh, stride_ton, BLOCK_M, BLOCK_D2, HEAD_DIM, EVEN_M, EVEN_D2,
                    ALIGN, TMA)


@functools.cache
def _supports_tma(device_index):
    return torch.cuda.get_device_capability(device_index)[0] >= 9


@functools.cache
def _supports_sm90(device_index):
    return torch.cuda.get_device_capability(device_index) == (9, 0)


def _head_dim_blocks(head_dim):
    full = max(16, triton.next_power_of_2(head_dim))
    low = full // 2
    if 16 <= low < head_dim:
        high = max(16, triton.next_power_of_2(head_dim - low))
        if low + high < full:
            return low, high
    return full, 0


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
    tensors.append(torch.empty_like(tensors[3]))
    batch, heads, n_q, head_dim = query.shape
    n_kv = key.shape[2]
    block_d, block_d2 = _head_dim_blocks(head_dim)
    itemsize = query.element_size()
    aligned = all(stride * itemsize % 16 == 0 for x in tensors for stride in x.stride()[:3])
    tma = (
        aligned
        and _supports_tma(query.device.index)
        and all(x.data_ptr() % 16 == 0 for x in tensors)
    )
    if (
        tma
        and _attention_jvp_sm90 is not None
        and query.dtype in _SM90_DTYPES
        and block_d + block_d2 <= 128
        and _supports_sm90(query.device.index)
    ):
        return _attention_jvp_sm90(*tensors, scale, block_d, block_d2)
    if tma:
        first = [TensorDescriptor.from_tensor(x, [1, 1, 1, block_d]) for x in tensors]
        second = [TensorDescriptor.from_tensor(x, [1, 1, 1, block_d2]) for x in tensors] if block_d2 else first
    else:
        first = second = tensors

    def grid(meta):
        return (triton.cdiv(n_q, meta["BLOCK_M"]) * batch * heads,)

    _attention_jvp_kernel[grid](
        *first,
        *second,
        *(stride for x in tensors for stride in x.stride()[:3]),
        heads,
        n_q,
        n_kv,
        scale,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_D2=block_d2,
        ALIGN=16 // itemsize if aligned else 1,
        TMA=tma,
        INPUT_PRECISION="ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32",
    )
    return tensors[-1]
