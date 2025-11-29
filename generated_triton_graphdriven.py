"""
Graph-Driven Triton Attention Implementation
Generated from egglog computation graph using PassAnalyzer

This kernel is generated DIRECTLY from the computation graph structure:
- No fixed templates for 2-pass or 3-pass
- Pass structure inferred automatically from dependencies
- Supports any attention algorithm egglog produces
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def graph_driven_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Graph-driven attention kernel.

    Memory passes: 2 (automatically inferred)
    Sync levels: 3 (automatically detected)
    Post-loop ops: M_div_fp (doesn't need K/V access)
    """
    # Get block indices
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute offsets for Q (this block of queries)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q block once
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # Initialize accumulators for global reductions
    global_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    tile_max_acc = tl.zeros([BLOCK_M], dtype=tl.float32)  # Accumulator for tile maxes

    # Pass 0: Find global max across all K tiles
    NUM_TILES = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(NUM_TILES):
        # Load K block
        k_offs = tile_idx * BLOCK_N + offs_n
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        k = tl.load(K_ptr + k_offs[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=k_offs[:, None] < N, other=0.0)

        # QK = Q @ K^T (optimized)
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]

        # Local max within this tile
        tile_max = tl.max(qk, axis=1)  # [BLOCK_M]

        # Update global max
        global_max = tl.maximum(global_max, tile_max)

    # Pass 1: Compute weighted sum with correction factor
    acc_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for tile_idx in range(NUM_TILES):
        # Load K and V blocks
        k_offs = tile_idx * BLOCK_N + offs_n
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh

        k = tl.load(K_ptr + k_offs[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=k_offs[:, None] < N, other=0.0)
        v = tl.load(V_ptr + k_offs[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                    mask=k_offs[:, None] < N, other=0.0)

        # Recompute QK for this tile
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]

        # Subtract global max and exponentiate (softmax numerator)
        qk_shifted = qk - global_max[:, None]
        weights = tl.exp(qk_shifted)  # [BLOCK_M, BLOCK_N]

        # Accumulate sum for normalization
        tile_sum = tl.sum(weights, axis=1)  # [BLOCK_M]
        acc_sum += tile_sum

        # Accumulate weighted values
        weighted_v = tl.dot(weights.to(v.dtype), v)  # [BLOCK_M, BLOCK_K]
        acc_out += weighted_v

    # Post-loop: Normalize by sum (division)
    output = acc_out / acc_sum[:, None]

    # Store output
    Out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, output)


def egg_attention_graphdriven_triton(Q, K, V, scale=None, causal=False):
    """
    Graph-driven Triton attention implementation.

    Generated directly from egglog computation graph structure.
    No fixed templates - kernel structure emerges from graph dependencies.

    Args:
        Q: Query tensor [batch, heads, seq_q, dim]
        K: Key tensor [batch, heads, seq_k, dim]
        V: Value tensor [batch, heads, seq_k, dim]
        scale: Attention scale (default: 1/sqrt(dim))
        causal: Whether to apply causal masking (not yet supported)

    Returns:
        Output tensor [batch, heads, seq_q, dim]
    """
    if causal:
        raise NotImplementedError("Causal masking not yet supported in graph-driven kernel")

    batch, num_heads, seq_q, head_dim = Q.shape
    seq_k = K.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Allocate output
    out = torch.empty_like(Q)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim

    # Launch kernel
    grid = (triton.cdiv(seq_q, BLOCK_M), batch, num_heads)

    graph_driven_attention_kernel[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_k, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out


# ============================================================================
# Testing Code
# ============================================================================

def test_correctness():
    """Test functional correctness against PyTorch reference."""
    print("=" * 70)
    print("TESTING GRAPH-DRIVEN TRITON KERNEL - FUNCTIONAL CORRECTNESS")
    print("=" * 70)

    torch.manual_seed(42)
    batch, heads, seq_len, dim = 2, 8, 512, 64

    # Create test inputs
    Q = torch.randn(batch, heads, seq_len, dim, device='cuda', dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    scale = 1.0 / math.sqrt(dim)

    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Scale: {scale:.6f}")

    # Reference implementation (PyTorch)
    with torch.no_grad():
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(attn_weights, V)

    # Graph-driven Triton kernel
    with torch.no_grad():
        out_triton = egg_attention_graphdriven_triton(Q, K, V, scale=scale)

    # Compare
    max_diff = (out_ref - out_triton).abs().max().item()
    mean_diff = (out_ref - out_triton).abs().mean().item()

    print(f"\nResults:")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    tolerance = 0.01
    if max_diff < tolerance:
        print(f"  ✓ PASSED (max diff < {tolerance})")
        return True
    else:
        print(f"  ✗ FAILED (max diff >= {tolerance})")
        return False


def test_performance():
    """Simple performance test."""
    print("\n" + "=" * 70)
    print("TESTING GRAPH-DRIVEN TRITON KERNEL - PERFORMANCE")
    print("=" * 70)

    torch.manual_seed(42)
    batch, heads, seq_len, dim = 1, 32, 1024, 64

    Q = torch.randn(batch, heads, seq_len, dim, device='cuda', dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    scale = 1.0 / math.sqrt(dim)

    print(f"Config: batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}")

    # Warmup
    for _ in range(10):
        _ = egg_attention_graphdriven_triton(Q, K, V, scale=scale)

    torch.cuda.synchronize()

    # Benchmark
    import time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 100
    start_event.record()
    for _ in range(iters):
        _ = egg_attention_graphdriven_triton(Q, K, V, scale=scale)
    end_event.record()

    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / iters
    print(f"\nAverage time: {avg_time:.3f} ms ({iters} iterations)")

    # Compare with SDPA
    with torch.no_grad():
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)

    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
    end_event.record()

    torch.cuda.synchronize()

    sdpa_time = start_event.elapsed_time(end_event) / iters
    print(f"SDPA time:      {sdpa_time:.3f} ms")
    print(f"Ratio:          {avg_time / sdpa_time:.2f}x {'slower' if avg_time > sdpa_time else 'faster'}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_correctness()
        test_performance()
    else:
        print("CUDA not available - skipping tests")
