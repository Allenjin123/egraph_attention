"""
Standalone Attention Kernel Benchmark

Compares attention implementations WITHOUT model overhead:
- Eager: Manual PyTorch matmuls + softmax
- SDPA: torch.nn.functional.scaled_dot_product_attention (Flash Attention backend)
- Triton: Our custom Triton implementation
- PyTorch: torch.compile'd graph-driven implementation
- TritonGD_2p: Graph-driven Triton (2-pass FuseMax, AUTO-GENERATED) ✓
- TritonGD_3p: Graph-driven Triton (3-pass, AUTO-GENERATED) ✓

Graph-Driven Generation Status:
  ✓ BOTH kernels are fully automatically generated from egglog JSON!
  ✓ 2-pass algorithm: Auto-generated from attention_2pass.json
  ✓ 3-pass algorithm: Auto-generated from attention.json
  ✓ PassAnalyzer: Automatically infers pass structure
  ✓ No hardcoded if/else for algorithm types

Usage:
    python kernel_benchmark.py --num-heads 32 --head-dim 64 --seq-lengths 256,512,1024,2048
"""

import torch
import torch.nn.functional as F
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import our Triton implementations
from triton_attention import triton_naive_attention

# Import PyTorch graph-driven implementation (torch.compile'd)
try:
    from generated_pytorch_2pass import egg_attention_graphdriven
    PYTORCH_AVAILABLE = True
    # Create compiled version
    pytorch_compiled = torch.compile(egg_attention_graphdriven, mode="max-autotune")
except ImportError:
    PYTORCH_AVAILABLE = False
    pytorch_compiled = None
    print("Warning: generated_pytorch_2pass.py not found")

# Import Triton graph-driven implementations (AUTO-GENERATED!)
# Both kernels are now automatically generated from egglog JSON
try:
    from generated_triton_2pass_auto import attention_kernel_2pass
    TRITON_GD_2PASS_AVAILABLE = True
except ImportError:
    TRITON_GD_2PASS_AVAILABLE = False
    print("Warning: generated_triton_2pass_auto.py not found")
    print("  Generate with: python generate_triton_kernel.py attention_2pass.json -o generated_triton_2pass_auto.py")

try:
    from generated_triton_3pass_auto import attention_kernel_auto as attention_kernel_3pass
    TRITON_GD_3PASS_AVAILABLE = True
except ImportError:
    TRITON_GD_3PASS_AVAILABLE = False
    print("Warning: generated_triton_3pass_auto.py not found")
    print("  Generate with: python generate_triton_kernel.py attention.json -o generated_triton_3pass_auto.py")


# ============================================================================
# Attention Implementations
# ============================================================================

def eager_attention(Q, K, V, scale, causal=True):
    """
    Manual PyTorch attention - BASELINE

    This is the standard eager attention computation:
    1. Compute QK^T and scale
    2. Apply causal mask
    3. Softmax
    4. Multiply by V
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        seq_len_q, seq_len_k = Q.shape[2], K.shape[2]
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

    # Softmax in float32 for numerical stability
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)

    return torch.matmul(attn_weights, V)


def sdpa_attention(Q, K, V, scale, causal=True):
    """
    PyTorch's Scaled Dot-Product Attention

    Under the hood, this uses Flash Attention when available,
    or falls back to efficient fused kernels.
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)


def triton_attention(Q, K, V, scale, causal=True):
    """
    Triton implementation (uses online softmax - 1 pass FlashAttention-style)
    """
    return triton_naive_attention(Q, K, V, scale=scale, causal=causal)


def pytorch_attention(Q, K, V, scale, causal=True):
    """
    PyTorch graph-driven attention (torch.compile'd with max-autotune)
    - Generated from egglog computation graph
    - Uses max-autotune mode for kernel fusion and optimization
    - First call triggers compilation (slow), subsequent calls are fast
    """
    if not PYTORCH_AVAILABLE or pytorch_compiled is None:
        raise RuntimeError("generated_pytorch_2pass.py not found or compilation failed")
    # Note: doesn't support causal masking yet
    return pytorch_compiled(Q, K, V, scale=scale)


def triton_gd_2pass(Q, K, V, scale, causal=True):
    """
    TritonGD 2-pass (AUTO-GENERATED from attention_2pass.json) ✓
    - FuseMax algorithm with tiling
    - 2 memory passes, 3 sync levels
    - 1 post-loop operation
    - Fully automatic code generation works!
    """
    if not TRITON_GD_2PASS_AVAILABLE:
        raise RuntimeError("generated_triton_2pass_auto.py not found")
    return attention_kernel_2pass(Q, K, V, scale=scale, causal=False)


def triton_gd_3pass(Q, K, V, scale, causal=True):
    """
    TritonGD 3-pass (AUTO-GENERATED from attention.json) ✓
    - Standard 3-pass softmax algorithm
    - 3 memory passes, 3 sync levels
    - 0 post-loop operations
    - Fully automatic code generation works!
    """
    if not TRITON_GD_3PASS_AVAILABLE:
        raise RuntimeError("generated_triton_3pass_auto.py not found")
    return attention_kernel_3pass(Q, K, V, scale=scale, causal=False)


# ============================================================================
# Verification
# ============================================================================

def verify_correctness(Q, K, V, scale, tolerance=0.01):
    """
    Verify all implementations produce the same output.
    All comparisons use non-causal (bidirectional) attention for fair comparison.

    Returns:
        True if all implementations match within tolerance
    """
    with torch.no_grad():
        # All non-causal for fair comparison
        out_eager = eager_attention(Q, K, V, scale, causal=False)
        out_sdpa = sdpa_attention(Q, K, V, scale, causal=False)
        out_triton = triton_attention(Q, K, V, scale, causal=False)
        out_pytorch = pytorch_attention(Q, K, V, scale, causal=False) if PYTORCH_AVAILABLE else None
        out_triton_gd_2p = triton_gd_2pass(Q, K, V, scale, causal=False) if TRITON_GD_2PASS_AVAILABLE else None
        out_triton_gd_3p = triton_gd_3pass(Q, K, V, scale, causal=False) if TRITON_GD_3PASS_AVAILABLE else None

    diff_sdpa = (out_eager - out_sdpa).abs().max().item()
    diff_triton = (out_eager - out_triton).abs().max().item()
    diff_pytorch = (out_eager - out_pytorch).abs().max().item() if out_pytorch is not None else float('inf')
    diff_triton_gd_2p = (out_eager - out_triton_gd_2p).abs().max().item() if out_triton_gd_2p is not None else float('inf')
    diff_triton_gd_3p = (out_eager - out_triton_gd_3p).abs().max().item() if out_triton_gd_3p is not None else float('inf')

    print(f"  Max diff eager vs SDPA:         {diff_sdpa:.6f}")
    print(f"  Max diff eager vs Triton:       {diff_triton:.6f}")
    if PYTORCH_AVAILABLE:
        print(f"  Max diff eager vs PyTorch:      {diff_pytorch:.6f}")
    if TRITON_GD_2PASS_AVAILABLE:
        print(f"  Max diff eager vs TritonGD_2p:  {diff_triton_gd_2p:.6f}")
    if TRITON_GD_3PASS_AVAILABLE:
        print(f"  Max diff eager vs TritonGD_3p:  {diff_triton_gd_3p:.6f}")

    passed = diff_sdpa < tolerance and diff_triton < tolerance
    if PYTORCH_AVAILABLE:
        passed = passed and diff_pytorch < tolerance
    if TRITON_GD_2PASS_AVAILABLE:
        passed = passed and diff_triton_gd_2p < tolerance
    if TRITON_GD_3PASS_AVAILABLE:
        passed = passed and diff_triton_gd_3p < tolerance
    return passed


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_kernel(fn, Q, K, V, scale, causal, warmup=10, iters=100):
    """
    Benchmark a single attention kernel.

    Args:
        fn: Attention function to benchmark
        Q, K, V: Input tensors
        scale: Scaling factor
        causal: Whether to use causal masking
        warmup: Number of warmup iterations
        iters: Number of timed iterations

    Returns:
        Average time per iteration in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(Q, K, V, scale, causal)

    torch.cuda.synchronize()

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = fn(Q, K, V, scale, causal)
    end_event.record()

    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters  # ms per iteration


# ============================================================================
# Main
# ============================================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Standalone Attention Kernel Benchmark")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument("--seq-lengths", type=str, default="256,512,1024,2048",
                       help="Comma-separated sequence lengths to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--skip-eager", action="store_true",
                       help="Skip eager attention (slow for long sequences)")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available! This benchmark requires a GPU.")
        return

    # Parse parameters
    batch_size = 1
    num_heads = args.num_heads
    head_dim = args.head_dim
    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]

    print("=" * 60)
    print("STANDALONE ATTENTION KERNEL BENCHMARK")
    print("=" * 60)
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, causal=False")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")

    # Verify correctness first
    print("\n" + "-" * 60)
    print("VERIFYING CORRECTNESS")
    print("-" * 60)

    torch.manual_seed(42)
    test_seq = min(seq_lengths)
    Q_test = torch.randn(batch_size, num_heads, test_seq, head_dim,
                        device='cuda', dtype=torch.float16)
    K_test = torch.randn_like(Q_test)
    V_test = torch.randn_like(Q_test)
    scale = 1.0 / math.sqrt(head_dim)

    if verify_correctness(Q_test, K_test, V_test, scale):
        print("  PASSED: All implementations match")
    else:
        print("  WARNING: Implementations differ - results may be unreliable")

    # Benchmark
    print("\n" + "-" * 60)
    print("BENCHMARKING")
    print("-" * 60)

    results = {
        "eager": {},
        "sdpa": {},
        "triton": {},
        "pytorch": {},
        "triton_gd_2p": {},
        "triton_gd_3p": {},
    }

    for seq_len in seq_lengths:
        print(f"\n[Sequence Length: {seq_len}]")

        # Create fresh inputs for each sequence length
        torch.manual_seed(42)
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float16)
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)

        # Eager attention
        if not args.skip_eager:
            try:
                eager_time = benchmark_kernel(
                    eager_attention, Q, K, V, scale, False,
                    warmup=args.warmup, iters=args.iters
                )
                results["eager"][seq_len] = eager_time
                print(f"  Eager:  {eager_time:.3f} ms")
            except Exception as e:
                print(f"  Eager:  FAILED ({e})")
                results["eager"][seq_len] = float('inf')
        else:
            results["eager"][seq_len] = float('inf')
            print(f"  Eager:  SKIPPED")

        # SDPA attention
        try:
            sdpa_time = benchmark_kernel(
                sdpa_attention, Q, K, V, scale, False,
                warmup=args.warmup, iters=args.iters
            )
            results["sdpa"][seq_len] = sdpa_time
            print(f"  SDPA:   {sdpa_time:.3f} ms")
        except Exception as e:
            print(f"  SDPA:   FAILED ({e})")
            results["sdpa"][seq_len] = float('inf')

        # Triton attention (online softmax)
        try:
            triton_time = benchmark_kernel(
                triton_attention, Q, K, V, scale, False,
                warmup=args.warmup, iters=args.iters
            )
            results["triton"][seq_len] = triton_time
            print(f"  Triton:      {triton_time:.3f} ms")
        except Exception as e:
            print(f"  Triton:      FAILED ({e})")
            results["triton"][seq_len] = float('inf')

        # PyTorch (torch.compile)
        if PYTORCH_AVAILABLE:
            try:
                pytorch_time = benchmark_kernel(
                    pytorch_attention, Q, K, V, scale, False,
                    warmup=args.warmup, iters=args.iters
                )
                results["pytorch"][seq_len] = pytorch_time
                print(f"  PyTorch:     {pytorch_time:.3f} ms")
            except Exception as e:
                print(f"  PyTorch:     FAILED ({e})")
                results["pytorch"][seq_len] = float('inf')
        else:
            results["pytorch"][seq_len] = float('inf')

        # TritonGD 2-pass (auto-generated from attention_2pass.json)
        if TRITON_GD_2PASS_AVAILABLE:
            try:
                triton_gd_2p_time = benchmark_kernel(
                    triton_gd_2pass, Q, K, V, scale, False,
                    warmup=args.warmup, iters=args.iters
                )
                results["triton_gd_2p"][seq_len] = triton_gd_2p_time
                print(f"  TritonGD_2p: {triton_gd_2p_time:.3f} ms")
            except Exception as e:
                print(f"  TritonGD_2p: FAILED ({e})")
                results["triton_gd_2p"][seq_len] = float('inf')
        else:
            results["triton_gd_2p"][seq_len] = float('inf')

        # TritonGD 3-pass (auto-generated from attention.json)
        if TRITON_GD_3PASS_AVAILABLE:
            try:
                triton_gd_3p_time = benchmark_kernel(
                    triton_gd_3pass, Q, K, V, scale, False,
                    warmup=args.warmup, iters=args.iters
                )
                results["triton_gd_3p"][seq_len] = triton_gd_3p_time
                print(f"  TritonGD_3p: {triton_gd_3p_time:.3f} ms")
            except Exception as e:
                print(f"  TritonGD_3p: FAILED ({e})")
                results["triton_gd_3p"][seq_len] = float('inf')
        else:
            results["triton_gd_3p"][seq_len] = float('inf')

        # Clear cache between sequence lengths
        torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    header = f"{'Seq Len':<10} {'Eager':<10} {'SDPA':<10} {'Triton':<10} {'PyTorch':<10} {'TGD_2p':<10} {'TGD_3p':<10}"
    print(header)
    print("-" * 70)

    for seq_len in seq_lengths:
        eager_t = results["eager"].get(seq_len, float('inf'))
        sdpa_t = results["sdpa"].get(seq_len, float('inf'))
        triton_t = results["triton"].get(seq_len, float('inf'))
        pytorch_t = results["pytorch"].get(seq_len, float('inf'))
        triton_gd_2p_t = results["triton_gd_2p"].get(seq_len, float('inf'))
        triton_gd_3p_t = results["triton_gd_3p"].get(seq_len, float('inf'))

        row = f"{seq_len:<10} "
        row += f"{eager_t:<10.2f} " if eager_t != float('inf') else f"{'SKIP':<10} "
        row += f"{sdpa_t:<10.2f} " if sdpa_t != float('inf') else f"{'FAIL':<10} "
        row += f"{triton_t:<10.2f} " if triton_t != float('inf') else f"{'FAIL':<10} "
        row += f"{pytorch_t:<10.2f} " if pytorch_t != float('inf') else f"{'N/A':<10} "
        row += f"{triton_gd_2p_t:<10.2f} " if triton_gd_2p_t != float('inf') else f"{'N/A':<10} "
        row += f"{triton_gd_3p_t:<10.2f}" if triton_gd_3p_t != float('inf') else f"{'N/A':<10}"

        print(row)

    # Create results directory
    results_dir = "kernel_benchmarks"
    os.makedirs(results_dir, exist_ok=True)

    # Generate plots
    print("\n" + "-" * 60)
    print("GENERATING PLOTS")
    print("-" * 60)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid_seq_lens = [s for s in seq_lengths if results["sdpa"].get(s, float('inf')) != float('inf')]

    # Plot 1: Time Comparison
    ax1 = axes[0]
    if valid_seq_lens:
        if not args.skip_eager:
            eager_times = [results["eager"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in eager_times):
                ax1.plot(valid_seq_lens, eager_times,
                        marker='o', linewidth=2, markersize=8, label='Eager', color='#3498db')

        ax1.plot(valid_seq_lens, [results["sdpa"][s] for s in valid_seq_lens],
                marker='s', linewidth=2, markersize=8, label='SDPA', color='#e74c3c')
        ax1.plot(valid_seq_lens, [results["triton"][s] for s in valid_seq_lens],
                marker='^', linewidth=2, markersize=8, label='Triton', color='#2ecc71')

        if PYTORCH_AVAILABLE:
            pytorch_times = [results["pytorch"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in pytorch_times):
                ax1.plot(valid_seq_lens, pytorch_times,
                        marker='p', linewidth=2, markersize=8, label='PyTorch', color='#9b59b6')

        if TRITON_GD_2PASS_AVAILABLE:
            triton_gd_2p_times = [results["triton_gd_2p"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in triton_gd_2p_times):
                ax1.plot(valid_seq_lens, triton_gd_2p_times,
                        marker='D', linewidth=2, markersize=8, label='TritonGD_2p', color='#16a085')

        if TRITON_GD_3PASS_AVAILABLE:
            triton_gd_3p_times = [results["triton_gd_3p"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in triton_gd_3p_times):
                ax1.plot(valid_seq_lens, triton_gd_3p_times,
                        marker='s', linewidth=2, markersize=8, label='TritonGD_3p', color='#e67e22')

    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Attention Kernel Performance\n(heads={num_heads}, dim={head_dim})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup over Eager (or comparison between SDPA and Triton)
    ax2 = axes[1]
    if valid_seq_lens:
        x = range(len(valid_seq_lens))
        width = 0.35

        if not args.skip_eager:
            # Speedup over eager
            sdpa_speedups = [results["eager"][s] / results["sdpa"][s]
                           if results["eager"].get(s, float('inf')) != float('inf') else 0
                           for s in valid_seq_lens]
            triton_speedups = [results["eager"][s] / results["triton"][s]
                             if results["eager"].get(s, float('inf')) != float('inf') else 0
                             for s in valid_seq_lens]

            bars1 = ax2.bar([i - width/2 for i in x], sdpa_speedups, width,
                           label='SDPA', color='#e74c3c', alpha=0.8)
            bars2 = ax2.bar([i + width/2 for i in x], triton_speedups, width,
                           label='Triton', color='#2ecc71', alpha=0.8)

            ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
            ax2.set_ylabel('Speedup over Eager', fontsize=12)
            ax2.set_title('Speedup over Eager Attention', fontsize=14, fontweight='bold')

            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                            f'{height:.1f}x', ha='center', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                            f'{height:.1f}x', ha='center', fontsize=9)
        else:
            # Direct comparison: Triton vs SDPA
            ratios = [results["triton"][s] / results["sdpa"][s] for s in valid_seq_lens]
            bars = ax2.bar(x, ratios, width * 2, label='Triton / SDPA', color='#9b59b6', alpha=0.8)
            ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Equal performance')
            ax2.set_ylabel('Triton / SDPA Time Ratio', fontsize=12)
            ax2.set_title('Triton vs SDPA Comparison', fontsize=14, fontweight='bold')

            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.2f}', ha='center', fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_seq_lens)
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, f'kernel_benchmark_h{num_heads}_d{head_dim}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    # Save CSV
    csv_path = os.path.join(results_dir, f'kernel_benchmark_h{num_heads}_d{head_dim}.csv')
    with open(csv_path, 'w') as f:
        f.write("Sequence_Length,Eager_ms,SDPA_ms,Triton_ms,PyTorch_ms,TritonGD_2p_ms,TritonGD_3p_ms\n")
        for seq_len in valid_seq_lens:
            eager_t = results["eager"].get(seq_len, float('inf'))
            pytorch_t = results["pytorch"].get(seq_len, float('inf'))
            triton_gd_2p_t = results["triton_gd_2p"].get(seq_len, float('inf'))
            triton_gd_3p_t = results["triton_gd_3p"].get(seq_len, float('inf'))
            f.write(f"{seq_len},{eager_t:.3f},{results['sdpa'][seq_len]:.3f},"
                    f"{results['triton'][seq_len]:.3f},{pytorch_t:.3f},"
                    f"{triton_gd_2p_t:.3f},{triton_gd_3p_t:.3f}\n")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
