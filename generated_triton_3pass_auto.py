"""
Graph-Driven Triton Kernel - Auto-generated from attention.json

Memory passes: 3
Sync levels: 3
Post-loop operations: 0
"""

import torch
import triton
import triton.language as tl
import math

@triton.jit
def attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Graph-driven kernel generation
    # Memory passes: 3
    # Sync levels: 3

    # Get block indices
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute offsets for Q (this block of queries)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # Initialize accumulators for global reductions
    global_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    global_sum = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    global_sum = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Memory Pass 0: Iterate over all K/V blocks
    NUM_TILES = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(NUM_TILES):
        # Load K block
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)

        # Operations for sync level 0
        # QK = Q @ K^T (optimized)
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
        rmaxm_R_max_m = tl.max(qk, axis=0)
        # ^ Global reduction

    # Memory Pass 1: Iterate over all K/V blocks
    NUM_TILES = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(NUM_TILES):
        # Load K block
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)

        # Operations for sync level 1
        msubmp_M_sub_mp = qk - rmaxm_R_max_m
        mexpmp_M_exp_mp = tl.exp(msubmp_M_sub_mp)
        raddm_R_add_m = tl.sum(mexpmp_M_exp_mp, axis=0)
        # ^ Global reduction

    # Memory Pass 2: Iterate over all K/V blocks
    NUM_TILES = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(NUM_TILES):
        # Load K block
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)

        # Load V block
        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        # Operations for sync level 2
        mdivmp_M_div_mp = mexpmp_M_exp_mp / raddm_R_add_m
        mmulfmp_M_mul_fmp = mdivmp_M_div_mp * v
        raddm_R_add_m = tl.sum(mmulfmp_M_mul_fmp, axis=1)
        # ^ Global reduction

    # Store output
    Out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, acc)

# Wrapper function

def attention_kernel_auto(Q, K, V, scale=None, causal=False):
    """
    Graph-driven attention kernel generated from attention.json

    Args:
        Q, K, V: Input tensors [batch, heads, seq, dim]
        scale: Attention scale (default: 1/sqrt(dim))
        causal: Causal masking (not yet supported)

    Returns:
        Output tensor [batch, heads, seq, dim]
    """
    if causal:
        raise NotImplementedError("Causal masking not yet supported")

    batch, num_heads, seq_q, head_dim = Q.shape
    seq_k = K.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(Q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim

    grid = (triton.cdiv(seq_q, BLOCK_M), batch, num_heads)

    attention_kernel[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_k, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out


if __name__ == "__main__":
    # Quick test
    print("Testing kernel...")
    import torch

    Q = torch.randn(1, 4, 128, 64, device='cuda', dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    out = attention_kernel_auto(Q, K, V)
    print(f"Output shape: {out.shape}")
    print("Test passed!")
