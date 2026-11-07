r"""Gluon kernel for the JVP of scaled dot-product attention on Hopper (sm_90).

Computes the same output tangent as `torchebm.cuda.attention`,

\[
\dot O = P \dot V + (P \odot \dot S) V
- \mathrm{rowsum}(P \odot \dot S) \odot O,
\qquad \dot S = s\,(\dot Q K^\top + Q \dot K^\top),
\]

with the FlashAttention-3 design (Shah et al., 2024). Each CTA is a
persistent program with three warp-specialized partitions: a producer warp
that issues TMA loads of \(Q, \dot Q, K, \dot K, V, \dot V, O\) into
shared-memory rings guarded by mbarriers, and two consumer warpgroups of 64
query rows each that run the wgmma products and the online softmax.
Consumers get 240 registers per thread and the producer 24 (`setmaxnreg`).
Within a consumer, the score products of the next key block are issued
before the value products of the current block, so the softmax overlaps
with the value products (intra-warpgroup overlap), and the two consumers
take turns issuing wgmma (ping-pong). A head dimension that is not a power
of two is split into two power-of-two blocks (72 = 64 + 16), as in
`torchebm.cuda.attention`. The output tangent is written through shared
memory with a TMA store while the next tile starts.
"""

import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

if tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 8):
    raise ImportError("the sm_90 attention JVP kernel needs Triton 3.8 or newer")

_LOG2E = 1.4426950408889634
_ROWS = gl.constexpr(64)
_BLOCK_N = 64
_MAX_REGISTERS = 168
_DTYPES = {torch.float16: gl.float16, torch.bfloat16: gl.bfloat16}


@gluon.jit
def _online_softmax(s, ts, m_i, l_i, r_i, n0, n_kv, scale, qk_scale, dtype: gl.constexpr,
                    p_layout: gl.constexpr, EVEN_N: gl.constexpr):
    if not EVEN_N:
        cols = n0 + gl.arange(0, s.shape[1], layout=gl.SliceLayout(0, s.type.layout))
        s = gl.where(gl.expand_dims(cols < n_kv, 0), s, float("-inf"))
    m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale)
    alpha = gl.exp2(m_i - m_new)
    p = gl.exp2(s * qk_scale - gl.expand_dims(m_new, 1))
    p_ts = p * (ts * scale)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    r_i = r_i * alpha + gl.sum(p_ts, axis=1)
    p = gl.convert_layout(p.to(dtype), p_layout)
    p_ts = gl.convert_layout(p_ts.to(dtype), p_layout)
    return p, p_ts, m_new, l_i, r_i, alpha


@gluon.jit
def _tile(smem, index):
    return smem.index(index).reshape([smem.shape[3], smem.shape[4]])


@gluon.jit
def _tile_coords(t, n_m, heads):
    bh = t // n_m
    return (t % n_m) * 2 * _ROWS, bh % heads, bh // heads


@gluon.jit
def _issue_scores(bufs, wg, stage, s, ts, SPLIT: gl.constexpr):
    q = _tile(bufs[0], wg)
    k = _tile(bufs[2], stage).permute((1, 0))
    s = warpgroup_mma(q, k, s, use_acc=False, is_async=True)
    if SPLIT:
        q2 = _tile(bufs[6], wg)
        k2 = _tile(bufs[8], stage).permute((1, 0))
        s = warpgroup_mma(q2, k2, s, is_async=True)
    ts = warpgroup_mma(_tile(bufs[1], wg), k, ts, use_acc=False, is_async=True)
    ts = warpgroup_mma(q, _tile(bufs[3], stage).permute((1, 0)), ts, is_async=True)
    if SPLIT:
        ts = warpgroup_mma(_tile(bufs[7], wg), k2, ts, is_async=True)
        ts = warpgroup_mma(q2, _tile(bufs[9], stage).permute((1, 0)), ts, is_async=True)
    return s, ts


@gluon.jit
def _issue_values(bufs, stage, p, p_ts, acc, acc2, SPLIT: gl.constexpr):
    acc = warpgroup_mma(p, _tile(bufs[5], stage), acc, is_async=True)
    acc = warpgroup_mma(p_ts, _tile(bufs[4], stage), acc, is_async=True)
    if SPLIT:
        acc2 = warpgroup_mma(p, _tile(bufs[11], stage), acc2, is_async=True)
        acc2 = warpgroup_mma(p_ts, _tile(bufs[10], stage), acc2, is_async=True)
    return acc, acc2


@gluon.jit
def _wait_values(acc, acc2, p, p_ts, SPLIT: gl.constexpr):
    if SPLIT:
        acc, acc2, p, p_ts = warpgroup_mma_wait(0, deps=(acc, acc2, p, p_ts))
    else:
        acc, p, p_ts = warpgroup_mma_wait(0, deps=(acc, p, p_ts))
    return acc, acc2


@gluon.jit
def _rescale(acc, acc2, alpha, SPLIT: gl.constexpr):
    acc = acc * gl.expand_dims(gl.convert_layout(alpha, gl.SliceLayout(1, acc.type.layout)), 1)
    if SPLIT:
        acc2 = acc2 * gl.expand_dims(gl.convert_layout(alpha, gl.SliceLayout(1, acc2.type.layout)), 1)
    return acc, acc2


@gluon.jit
def _write_tangent(acc, inv_l, mu, o_smem, wg):
    o_tile = _tile(o_smem, wg)
    o = o_tile.load(acc.type.layout).to(gl.float32)
    row: gl.constexpr = gl.SliceLayout(1, acc.type.layout)
    inv_l = gl.expand_dims(gl.convert_layout(inv_l, row), 1)
    mu = gl.expand_dims(gl.convert_layout(mu, row), 1)
    o_tile.store((acc * inv_l - mu * o).to(o_smem.dtype))


@gluon.jit
def _consumer(bufs, o_bufs, o_descs, bars, n_tiles, n_m, heads, n_kv, scale, qk_scale, WG: gl.constexpr,
              BLOCK_N: gl.constexpr, NUM_STAGES: gl.constexpr, SPLIT: gl.constexpr, EVEN_N: gl.constexpr):
    q_ready, q_empty, k_ready, k_empty, v_ready, v_empty, o_ready, o_empty, turn = bars
    BLOCK_D: gl.constexpr = bufs[0].shape[4]
    BLOCK_D2: gl.constexpr = bufs[6].shape[4]
    VALUE_GROUPS: gl.constexpr = 4 if SPLIT else 2
    dtype: gl.constexpr = bufs[0].dtype
    s_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                       instr_shape=[16, BLOCK_N, 16])
    o_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                       instr_shape=[16, BLOCK_D, 16])
    o2_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                        instr_shape=[16, BLOCK_D2, 16])
    p_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=o_layout, k_width=2)
    row_s: gl.constexpr = gl.SliceLayout(1, s_layout)
    n_blocks = gl.cdiv(n_kv, BLOCK_N)
    s = gl.zeros([_ROWS, BLOCK_N], gl.float32, s_layout)
    ts = gl.zeros([_ROWS, BLOCK_N], gl.float32, s_layout)

    i = 0
    for t in range(gl.program_id(0), n_tiles, gl.num_programs(0)):
        m0, h, b = _tile_coords(t, n_m, heads)
        c = i * n_blocks
        m_i = gl.full([_ROWS], float("-inf"), gl.float32, row_s)
        l_i = gl.zeros([_ROWS], gl.float32, row_s)
        r_i = gl.zeros([_ROWS], gl.float32, row_s)
        acc = gl.zeros([_ROWS, BLOCK_D], gl.float32, o_layout)
        acc2 = gl.zeros([_ROWS, BLOCK_D2], gl.float32, o2_layout)

        mbarrier.wait(q_ready, i & 1)
        mbarrier.wait(k_ready.index(c % NUM_STAGES), (c // NUM_STAGES) & 1)
        s, ts = _issue_scores(bufs, WG, c % NUM_STAGES, s, ts, SPLIT)
        s, ts = warpgroup_mma_wait(0, deps=(s, ts))
        mbarrier.arrive(k_empty.index(c % NUM_STAGES))
        mbarrier.arrive(q_empty, pred=n_blocks == 1)
        tma.store_wait(0)
        mbarrier.arrive(o_empty.index(WG), pred=i > 0)
        p, p_ts, m_i, l_i, r_i, alpha = _online_softmax(s, ts, m_i, l_i, r_i, 0, n_kv, scale, qk_scale, dtype,
                                                        p_layout, EVEN_N)

        for j in range(1, n_blocks):
            stage = (c + j) % NUM_STAGES
            prev = (c + j - 1) % NUM_STAGES
            mbarrier.wait(k_ready.index(stage), ((c + j) // NUM_STAGES) & 1)
            mbarrier.wait(turn.index(WG), ((i * (n_blocks - 1) + j - 1) & 1) ^ (1 - WG))
            s, ts = _issue_scores(bufs, WG, stage, s, ts, SPLIT)
            acc, acc2 = _rescale(acc, acc2, alpha, SPLIT)
            mbarrier.wait(v_ready.index(prev), ((c + j - 1) // NUM_STAGES) & 1)
            acc, acc2 = _issue_values(bufs, prev, p, p_ts, acc, acc2, SPLIT)
            mbarrier.arrive(turn.index(1 - WG))
            s, ts = warpgroup_mma_wait(VALUE_GROUPS, deps=(s, ts))
            mbarrier.arrive(k_empty.index(stage))
            mbarrier.arrive(q_empty, pred=j == n_blocks - 1)
            p_next, p_ts_next, m_i, l_i, r_i, alpha = _online_softmax(
                s, ts, m_i, l_i, r_i, j * BLOCK_N, n_kv, scale, qk_scale, dtype, p_layout, EVEN_N)
            acc, acc2 = _wait_values(acc, acc2, p, p_ts, SPLIT)
            mbarrier.arrive(v_empty.index(prev))
            p = p_next
            p_ts = p_ts_next

        last = (c + n_blocks - 1) % NUM_STAGES
        acc, acc2 = _rescale(acc, acc2, alpha, SPLIT)
        mbarrier.wait(v_ready.index(last), ((c + n_blocks - 1) // NUM_STAGES) & 1)
        acc, acc2 = _issue_values(bufs, last, p, p_ts, acc, acc2, SPLIT)
        acc, acc2 = _wait_values(acc, acc2, p, p_ts, SPLIT)
        mbarrier.arrive(v_empty.index(last))

        inv_l = 1.0 / l_i
        mu = r_i * inv_l
        mbarrier.wait(o_ready.index(WG), i & 1)
        _write_tangent(acc, inv_l, mu, o_bufs[0], WG)
        if SPLIT:
            _write_tangent(acc2, inv_l, mu, o_bufs[1], WG)
        fence_async_shared()
        tma.async_store(o_descs[2], [b, h, m0 + WG * _ROWS, 0], o_bufs[0].index(WG))
        if SPLIT:
            tma.async_store(o_descs[3], [b, h, m0 + WG * _ROWS, BLOCK_D], o_bufs[1].index(WG))
        i += 1
    tma.store_wait(0)


@gluon.jit
def _producer(descs, bufs, o_bufs, o_descs, bars, n_tiles, n_m, heads, n_kv, BLOCK_N: gl.constexpr,
              NUM_STAGES: gl.constexpr, SPLIT: gl.constexpr):
    q_ready, q_empty, k_ready, k_empty, v_ready, v_empty, o_ready, o_empty, turn = bars
    BLOCK_D: gl.constexpr = descs[0].block_type.shape[3]
    Q_BYTES: gl.constexpr = descs[0].block_type.nbytes + descs[6].block_type.nbytes * SPLIT
    KV_BYTES: gl.constexpr = descs[2].block_type.nbytes + descs[8].block_type.nbytes * SPLIT
    O_BYTES: gl.constexpr = o_descs[0].block_type.nbytes + o_descs[1].block_type.nbytes * SPLIT
    n_blocks = gl.cdiv(n_kv, BLOCK_N)

    i = 0
    for t in range(gl.program_id(0), n_tiles, gl.num_programs(0)):
        m0, h, b = _tile_coords(t, n_m, heads)
        c = i * n_blocks
        mbarrier.wait(q_empty, (i & 1) ^ 1)
        mbarrier.expect(q_ready, 4 * Q_BYTES)
        for wg in gl.static_range(2):
            row = m0 + wg * _ROWS
            for x in gl.static_range(2):
                tma.async_load(descs[x], [b, h, row, 0], q_ready, bufs[x].index(wg))
                if SPLIT:
                    tma.async_load(descs[6 + x], [b, h, row, BLOCK_D], q_ready, bufs[6 + x].index(wg))
        for j in range(n_blocks):
            stage = (c + j) % NUM_STAGES
            phase = ((c + j) // NUM_STAGES) & 1
            n0 = j * BLOCK_N
            mbarrier.wait(k_empty.index(stage), phase ^ 1)
            ready = k_ready.index(stage)
            mbarrier.expect(ready, 2 * KV_BYTES)
            for x in gl.static_range(2, 4):
                tma.async_load(descs[x], [b, h, n0, 0], ready, bufs[x].index(stage))
                if SPLIT:
                    tma.async_load(descs[6 + x], [b, h, n0, BLOCK_D], ready, bufs[6 + x].index(stage))
            mbarrier.wait(v_empty.index(stage), phase ^ 1)
            ready = v_ready.index(stage)
            mbarrier.expect(ready, 2 * KV_BYTES)
            for x in gl.static_range(4, 6):
                tma.async_load(descs[x], [b, h, n0, 0], ready, bufs[x].index(stage))
                if SPLIT:
                    tma.async_load(descs[6 + x], [b, h, n0, BLOCK_D], ready, bufs[6 + x].index(stage))
        for wg in gl.static_range(2):
            row = m0 + wg * _ROWS
            ready = o_ready.index(wg)
            mbarrier.wait(o_empty.index(wg), (i & 1) ^ 1)
            mbarrier.expect(ready, O_BYTES)
            tma.async_load(o_descs[0], [b, h, row, 0], ready, o_bufs[0].index(wg))
            if SPLIT:
                tma.async_load(o_descs[1], [b, h, row, BLOCK_D], ready, o_bufs[1].index(wg))
        i += 1


@gluon.jit
def _attention_jvp_sm90_kernel(
    q_desc, tq_desc, k_desc, tk_desc, v_desc, tv_desc,
    q2_desc, tq2_desc, k2_desc, tk2_desc, v2_desc, tv2_desc,
    o_desc, o2_desc, to_desc, to2_desc,
    n_tiles, n_m, heads, n_kv, scale, qk_scale,
    BLOCK_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    SPLIT: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    dtype: gl.constexpr = q_desc.dtype
    descs = (q_desc, tq_desc, k_desc, tk_desc, v_desc, tv_desc,
             q2_desc, tq2_desc, k2_desc, tk2_desc, v2_desc, tv2_desc)
    o_descs = (o_desc, o2_desc, to_desc, to2_desc)
    bufs = ()
    for x in gl.static_range(12):
        if x < 6 or SPLIT:
            bufs += (gl.allocate_shared_memory(
                dtype, [2 if x % 6 < 2 else NUM_STAGES] + descs[x].block_type.shape, descs[x].layout), )
        else:
            bufs += (bufs[x - 6], )
    o_bufs = (gl.allocate_shared_memory(dtype, [2] + o_desc.block_type.shape, o_desc.layout), )
    if SPLIT:
        o_bufs += (gl.allocate_shared_memory(dtype, [2] + o2_desc.block_type.shape, o2_desc.layout), )
    else:
        o_bufs += (o_bufs[0], )
    q_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    k_ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    k_empty = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    v_ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    v_empty = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    o_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    turn = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(q_ready, count=1)
    mbarrier.init(q_empty, count=2)
    for x in gl.static_range(NUM_STAGES):
        mbarrier.init(k_ready.index(x), count=1)
        mbarrier.init(k_empty.index(x), count=2)
        mbarrier.init(v_ready.index(x), count=1)
        mbarrier.init(v_empty.index(x), count=2)
    for x in gl.static_range(2):
        mbarrier.init(o_ready.index(x), count=1)
        mbarrier.init(o_empty.index(x), count=1)
        mbarrier.init(turn.index(x), count=1)

    bars = (q_ready, q_empty, k_ready, k_empty, v_ready, v_empty, o_ready, o_empty, turn)
    gl.warp_specialize([
        (_consumer, (bufs, o_bufs, o_descs, bars, n_tiles, n_m, heads, n_kv, scale, qk_scale, 0, BLOCK_N,
                     NUM_STAGES, SPLIT, EVEN_N)),
        (_consumer, (bufs, o_bufs, o_descs, bars, n_tiles, n_m, heads, n_kv, scale, qk_scale, 1, BLOCK_N,
                     NUM_STAGES, SPLIT, EVEN_N)),
        (_producer, (descs, bufs, o_bufs, o_descs, bars, n_tiles, n_m, heads, n_kv, BLOCK_N, NUM_STAGES,
                     SPLIT)),
    ], [4, 1], [240, 24])


@functools.cache
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _shared_layout(rows, cols, dtype):
    return gl.NVMMASharedLayout.get_default_for([1, 1, rows, cols], _DTYPES[dtype])


def _descriptors(tensors, rows, cols):
    return [TensorDescriptor.from_tensor(x, [1, 1, r, cols], _shared_layout(r, cols, x.dtype))
            for x, r in zip(tensors, rows)]


def attention_jvp_sm90(
    query, key, value, output, query_t, key_t, value_t, output_t, scale, block_d, block_d2
):
    r"""Writes the output tangent of scaled dot-product attention on sm_90.

    Args:
        query: Queries of shape ``(B, H, N_q, D)``, float16 or bfloat16, with
            16-byte aligned base pointer and strides.
        key: Keys of shape ``(B, H, N_kv, D)``.
        value: Values of shape ``(B, H, N_kv, D)``.
        output: Attention output for ``query``, ``key``, ``value``.
        query_t: Tangent of ``query``.
        key_t: Tangent of ``key``.
        value_t: Tangent of ``value``.
        output_t: Buffer for the output tangent, shaped like ``output``.
        scale: Scaling of the logits.
        block_d: First power-of-two block of the head dimension.
        block_d2: Second power-of-two block of the head dimension, 0 if
            ``block_d`` covers it.

    Returns:
        torch.Tensor: ``output_t``.
    """
    batch, heads, n_q, _ = query.shape
    n_kv = key.shape[2]
    rows = (_ROWS.value, _ROWS.value, _BLOCK_N, _BLOCK_N, _BLOCK_N, _BLOCK_N, _ROWS.value, _ROWS.value)
    tensors = (query, query_t, key, key_t, value, value_t, output, output_t)
    first = _descriptors(tensors, rows, block_d)
    second = _descriptors(tensors, rows, block_d2) if block_d2 else first
    n_m = triton.cdiv(n_q, 2 * _ROWS.value)
    n_tiles = n_m * heads * batch
    _attention_jvp_sm90_kernel[(min(n_tiles, _num_sms(query.device.index)),)](
        *first[:6], *second[:6], first[6], second[6], first[7], second[7],
        n_tiles, n_m, heads, n_kv, scale, scale * _LOG2E,
        BLOCK_N=_BLOCK_N,
        NUM_STAGES=3 if block_d + block_d2 <= 80 else 2,
        SPLIT=block_d2 > 0,
        EVEN_N=n_kv % _BLOCK_N == 0,
        num_warps=4,
        maxnreg=_MAX_REGISTERS,
    )
    return output_t
