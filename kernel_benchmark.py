"""
Standalone Attention Kernel Benchmark

Compares attention implementations WITHOUT model overhead:
- Eager: Manual PyTorch matmuls + softmax
- SDPA: torch.nn.functional.scaled_dot_product_attention (Flash Attention backend)
- Triton: Our custom Triton implementation

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
from triton_attention import triton_naive_attention, triton_truly_naive_attention

# Import egglog-generated attention kernels
from generated_attention import egg_attention_triton as egg_3pass_attention
from generated_2pass_attention import egg_attention_2pass_triton as egg_2pass_attention


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
    Our Triton implementation (uses online softmax - 1 pass)
    """
    return triton_naive_attention(Q, K, V, scale=scale, causal=causal)


def triton_truly_naive(Q, K, V, scale, causal=True):
    """
    TRULY NAIVE Triton implementation (3-pass, no online softmax)
    - Pass 1: Compute scores, find max
    - Pass 2: Compute exp, find sum
    - Pass 3: Compute output
    """
    return triton_truly_naive_attention(Q, K, V, scale=scale, causal=causal)


def egg_3pass(Q, K, V, scale, causal=True):
    """
    Egglog-generated 3-pass attention (no causal support yet)
    - Pass 1: Compute QK, find row max
    - Pass 2: Compute exp sum
    - Pass 3: Compute output
    """
    # Note: egglog-generated kernel doesn't support causal masking yet
    return egg_3pass_attention(Q, K, V, scale=scale)


def egg_2pass(Q, K, V, scale, causal=True):
    """
    Egglog-generated 2-pass tiled attention (FuseMax algorithm)
    - Pass 1: Find global max from local tile maxes
    - Pass 2: Apply correction factor, compute output
    """
    # Note: egglog-generated kernel doesn't support causal masking yet
    return egg_2pass_attention(Q, K, V, scale=scale)


# ============================================================================
# Verification
# ============================================================================

def verify_correctness(Q, K, V, scale, tolerance=0.01):
    """
    Verify all implementations produce the same output.

    Returns:
        True if all implementations match within tolerance
    """
    with torch.no_grad():
        # Causal versions
        out_eager = eager_attention(Q, K, V, scale, causal=True)
        out_sdpa = sdpa_attention(Q, K, V, scale, causal=True)
        out_triton = triton_attention(Q, K, V, scale, causal=True)
        out_truly_naive = triton_truly_naive(Q, K, V, scale, causal=True)

        # Non-causal reference for egglog kernels
        out_eager_nc = eager_attention(Q, K, V, scale, causal=False)
        out_egg_3pass = egg_3pass(Q, K, V, scale, causal=False)
        out_egg_2pass = egg_2pass(Q, K, V, scale, causal=False)

    diff_sdpa = (out_eager - out_sdpa).abs().max().item()
    diff_triton = (out_eager - out_triton).abs().max().item()
    diff_truly_naive = (out_eager - out_truly_naive).abs().max().item()
    diff_egg_3pass = (out_eager_nc - out_egg_3pass).abs().max().item()
    diff_egg_2pass = (out_eager_nc - out_egg_2pass).abs().max().item()

    print(f"  Max diff eager vs SDPA:         {diff_sdpa:.6f}")
    print(f"  Max diff eager vs Triton:       {diff_triton:.6f}")
    print(f"  Max diff eager vs Truly Naive:  {diff_truly_naive:.6f}")
    print(f"  Max diff eager vs Egg 3-pass:   {diff_egg_3pass:.6f} (non-causal)")
    print(f"  Max diff eager vs Egg 2-pass:   {diff_egg_2pass:.6f} (non-causal)")

    passed = (diff_sdpa < tolerance and diff_triton < tolerance and
              diff_truly_naive < tolerance and diff_egg_3pass < tolerance and
              diff_egg_2pass < tolerance)
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
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, causal=True")
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
        "truly_naive": {},
        "egg_3pass": {},
        "egg_2pass": {},
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
                    eager_attention, Q, K, V, scale, True,
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
                sdpa_attention, Q, K, V, scale, True,
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
                triton_attention, Q, K, V, scale, True,
                warmup=args.warmup, iters=args.iters
            )
            results["triton"][seq_len] = triton_time
            print(f"  Triton (online):      {triton_time:.3f} ms")
        except Exception as e:
            print(f"  Triton (online):      FAILED ({e})")
            results["triton"][seq_len] = float('inf')

        # Truly Naive Triton (3-pass, no online softmax)
        try:
            truly_naive_time = benchmark_kernel(
                triton_truly_naive, Q, K, V, scale, True,
                warmup=args.warmup, iters=args.iters
            )
            results["truly_naive"][seq_len] = truly_naive_time
            print(f"  Triton (truly naive): {truly_naive_time:.3f} ms")
        except Exception as e:
            print(f"  Triton (truly naive): FAILED ({e})")
            results["truly_naive"][seq_len] = float('inf')

        # Egglog-generated 3-pass (non-causal)
        try:
            egg_3pass_time = benchmark_kernel(
                egg_3pass, Q, K, V, scale, False,  # non-causal
                warmup=args.warmup, iters=args.iters
            )
            results["egg_3pass"][seq_len] = egg_3pass_time
            print(f"  Egg 3-pass:           {egg_3pass_time:.3f} ms (non-causal)")
        except Exception as e:
            print(f"  Egg 3-pass:           FAILED ({e})")
            results["egg_3pass"][seq_len] = float('inf')

        # Egglog-generated 2-pass (non-causal)
        try:
            egg_2pass_time = benchmark_kernel(
                egg_2pass, Q, K, V, scale, False,  # non-causal
                warmup=args.warmup, iters=args.iters
            )
            results["egg_2pass"][seq_len] = egg_2pass_time
            print(f"  Egg 2-pass:           {egg_2pass_time:.3f} ms (non-causal)")
        except Exception as e:
            print(f"  Egg 2-pass:           FAILED ({e})")
            results["egg_2pass"][seq_len] = float('inf')

        # Clear cache between sequence lengths
        torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    header = f"{'Seq Len':<10} {'Eager':<10} {'SDPA':<10} {'Triton':<10} {'TrulyNaive':<12} {'Egg3Pass':<10} {'Egg2Pass':<10}"
    print(header)
    print("-" * 72)

    for seq_len in seq_lengths:
        eager_t = results["eager"].get(seq_len, float('inf'))
        sdpa_t = results["sdpa"].get(seq_len, float('inf'))
        triton_t = results["triton"].get(seq_len, float('inf'))
        truly_naive_t = results["truly_naive"].get(seq_len, float('inf'))
        egg_3pass_t = results["egg_3pass"].get(seq_len, float('inf'))
        egg_2pass_t = results["egg_2pass"].get(seq_len, float('inf'))

        row = f"{seq_len:<10} "
        row += f"{eager_t:<10.2f} " if eager_t != float('inf') else f"{'SKIP':<10} "
        row += f"{sdpa_t:<10.2f} " if sdpa_t != float('inf') else f"{'FAIL':<10} "
        row += f"{triton_t:<10.2f} " if triton_t != float('inf') else f"{'FAIL':<10} "
        row += f"{truly_naive_t:<12.2f} " if truly_naive_t != float('inf') else f"{'FAIL':<12} "
        row += f"{egg_3pass_t:<10.2f} " if egg_3pass_t != float('inf') else f"{'FAIL':<10} "
        row += f"{egg_2pass_t:<10.2f}" if egg_2pass_t != float('inf') else f"{'FAIL':<10}"

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
                marker='^', linewidth=2, markersize=8, label='Triton (online)', color='#2ecc71')
        ax1.plot(valid_seq_lens, [results["truly_naive"][s] for s in valid_seq_lens],
                marker='d', linewidth=2, markersize=8, label='Triton (3-pass)', color='#9b59b6')
        ax1.plot(valid_seq_lens, [results["egg_3pass"][s] for s in valid_seq_lens],
                marker='x', linewidth=2, markersize=8, label='Egg 3-pass', color='#f39c12')
        ax1.plot(valid_seq_lens, [results["egg_2pass"][s] for s in valid_seq_lens],
                marker='*', linewidth=2, markersize=10, label='Egg 2-pass', color='#1abc9c')

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
        f.write("Sequence_Length,Eager_ms,SDPA_ms,Triton_online_ms,Triton_3pass_ms,Egg_3pass_ms,Egg_2pass_ms\n")
        for seq_len in valid_seq_lens:
            eager_t = results["eager"].get(seq_len, float('inf'))
            truly_naive_t = results["truly_naive"].get(seq_len, float('inf'))
            egg_3pass_t = results["egg_3pass"].get(seq_len, float('inf'))
            egg_2pass_t = results["egg_2pass"].get(seq_len, float('inf'))
            f.write(f"{seq_len},{eager_t:.3f},{results['sdpa'][seq_len]:.3f},"
                    f"{results['triton'][seq_len]:.3f},{truly_naive_t:.3f},"
                    f"{egg_3pass_t:.3f},{egg_2pass_t:.3f}\n")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
