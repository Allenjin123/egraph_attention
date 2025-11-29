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

# Import egglog-generated hybrid attention kernels (Phase 2)
try:
    from generated_hybrid_2pass import egg_attention_hybrid_2pass
    HYBRID_2PASS_AVAILABLE = True
except ImportError:
    HYBRID_2PASS_AVAILABLE = False
    print("Warning: generated_hybrid_2pass.py not found")

try:
    from generated_hybrid_3pass import egg_attention_hybrid
    HYBRID_3PASS_AVAILABLE = True
except ImportError:
    HYBRID_3PASS_AVAILABLE = False
    print("Warning: generated_hybrid_3pass.py not found")

# Import pure PyTorch graph-driven implementation
try:
    from generated_pytorch_2pass import egg_attention_graphdriven
    PYTORCH_GRAPHDRIVEN_AVAILABLE = True
    # Create compiled version
    pytorch_graphdriven_compiled = torch.compile(egg_attention_graphdriven, mode="max-autotune")
except ImportError:
    PYTORCH_GRAPHDRIVEN_AVAILABLE = False
    pytorch_graphdriven_compiled = None
    print("Warning: generated_pytorch_2pass.py not found")

# Import Triton graph-driven implementation
try:
    from generated_triton_graphdriven import egg_attention_graphdriven_triton
    TRITON_GRAPHDRIVEN_AVAILABLE = True
except ImportError:
    TRITON_GRAPHDRIVEN_AVAILABLE = False
    print("Warning: generated_triton_graphdriven.py not found")


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


def hybrid_3pass(Q, K, V, scale, causal=True):
    """
    Hybrid 3-pass attention (generated from egglog graph)
    - Pass 1: Compute QK, find row max
    - Pass 2: Compute exp sum
    - Pass 3: Compute output
    """
    if not HYBRID_3PASS_AVAILABLE:
        raise RuntimeError("generated_hybrid_3pass.py not found")
    # Note: hybrid kernel doesn't support causal masking yet
    return egg_attention_hybrid(Q, K, V, scale=scale)


def hybrid_2pass(Q, K, V, scale, causal=True):
    """
    Hybrid 2-pass tiled attention (FuseMax algorithm, generated from egglog graph)
    - Pass 1: Find global max from local tile maxes
    - Pass 2: Apply correction factor, compute output
    """
    if not HYBRID_2PASS_AVAILABLE:
        raise RuntimeError("generated_hybrid_2pass.py not found")
    # Note: hybrid kernel doesn't support causal masking yet
    return egg_attention_hybrid_2pass(Q, K, V, scale=scale)


def pytorch_graphdriven(Q, K, V, scale, causal=True):
    """
    Pure PyTorch graph-driven attention (generated from egglog computation graph)
    - Uses standard PyTorch ops with dimension-aware broadcasting
    - GPU accelerated when tensors are on CUDA
    """
    if not PYTORCH_GRAPHDRIVEN_AVAILABLE:
        raise RuntimeError("generated_pytorch_2pass.py not found")
    # Note: doesn't support causal masking yet
    return egg_attention_graphdriven(Q, K, V, scale=scale)


def pytorch_compiled(Q, K, V, scale, causal=True):
    """
    torch.compile'd version of PyTorch graph-driven attention
    - Uses max-autotune mode for kernel fusion and optimization
    - First call triggers compilation (slow), subsequent calls are fast
    """
    if not PYTORCH_GRAPHDRIVEN_AVAILABLE or pytorch_graphdriven_compiled is None:
        raise RuntimeError("generated_pytorch_2pass.py not found or compilation failed")
    # Note: doesn't support causal masking yet
    return pytorch_graphdriven_compiled(Q, K, V, scale=scale)


def triton_graphdriven(Q, K, V, scale, causal=True):
    """
    Triton graph-driven attention (generated from egglog computation graph)
    - Kernel structure emerges from graph dependencies
    - No fixed templates for 2-pass or 3-pass
    - Automatically adapts to any attention algorithm
    """
    if not TRITON_GRAPHDRIVEN_AVAILABLE:
        raise RuntimeError("generated_triton_graphdriven.py not found")
    # Note: doesn't support causal masking yet
    return egg_attention_graphdriven_triton(Q, K, V, scale=scale, causal=causal)


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
        out_truly_naive = triton_truly_naive(Q, K, V, scale, causal=False)
        out_hybrid_3pass = hybrid_3pass(Q, K, V, scale, causal=False) if HYBRID_3PASS_AVAILABLE else None
        out_hybrid_2pass = hybrid_2pass(Q, K, V, scale, causal=False) if HYBRID_2PASS_AVAILABLE else None
        out_pytorch_gd = pytorch_graphdriven(Q, K, V, scale, causal=False) if PYTORCH_GRAPHDRIVEN_AVAILABLE else None
        out_pytorch_compiled = pytorch_compiled(Q, K, V, scale, causal=False) if PYTORCH_GRAPHDRIVEN_AVAILABLE else None
        out_triton_gd = triton_graphdriven(Q, K, V, scale, causal=False) if TRITON_GRAPHDRIVEN_AVAILABLE else None

    diff_sdpa = (out_eager - out_sdpa).abs().max().item()
    diff_triton = (out_eager - out_triton).abs().max().item()
    diff_truly_naive = (out_eager - out_truly_naive).abs().max().item()
    diff_hybrid_3pass = (out_eager - out_hybrid_3pass).abs().max().item() if out_hybrid_3pass is not None else float('inf')
    diff_hybrid_2pass = (out_eager - out_hybrid_2pass).abs().max().item() if out_hybrid_2pass is not None else float('inf')
    diff_pytorch_gd = (out_eager - out_pytorch_gd).abs().max().item() if out_pytorch_gd is not None else float('inf')
    diff_pytorch_compiled = (out_eager - out_pytorch_compiled).abs().max().item() if out_pytorch_compiled is not None else float('inf')
    diff_triton_gd = (out_eager - out_triton_gd).abs().max().item() if out_triton_gd is not None else float('inf')

    print(f"  Max diff eager vs SDPA:           {diff_sdpa:.6f}")
    print(f"  Max diff eager vs Triton:         {diff_triton:.6f}")
    print(f"  Max diff eager vs Truly Naive:    {diff_truly_naive:.6f}")
    if HYBRID_3PASS_AVAILABLE:
        print(f"  Max diff eager vs Hybrid 3-pass:  {diff_hybrid_3pass:.6f}")
    if HYBRID_2PASS_AVAILABLE:
        print(f"  Max diff eager vs Hybrid 2-pass:  {diff_hybrid_2pass:.6f}")
    if PYTORCH_GRAPHDRIVEN_AVAILABLE:
        print(f"  Max diff eager vs PyTorch GD:     {diff_pytorch_gd:.6f}")
        print(f"  Max diff eager vs PyTorch Compile:{diff_pytorch_compiled:.6f}")
    if TRITON_GRAPHDRIVEN_AVAILABLE:
        print(f"  Max diff eager vs Triton GD:      {diff_triton_gd:.6f}")

    passed = (diff_sdpa < tolerance and diff_triton < tolerance and
              diff_truly_naive < tolerance)
    if HYBRID_3PASS_AVAILABLE:
        passed = passed and diff_hybrid_3pass < tolerance
    if HYBRID_2PASS_AVAILABLE:
        passed = passed and diff_hybrid_2pass < tolerance
    if PYTORCH_GRAPHDRIVEN_AVAILABLE:
        passed = passed and diff_pytorch_gd < tolerance and diff_pytorch_compiled < tolerance
    if TRITON_GRAPHDRIVEN_AVAILABLE:
        passed = passed and diff_triton_gd < tolerance
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
        "truly_naive": {},
        "hybrid_3pass": {},
        "hybrid_2pass": {},
        "pytorch_gd": {},
        "pytorch_compiled": {},
        "triton_gd": {},
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
            print(f"  Triton (online):      {triton_time:.3f} ms")
        except Exception as e:
            print(f"  Triton (online):      FAILED ({e})")
            results["triton"][seq_len] = float('inf')

        # Truly Naive Triton (3-pass, no online softmax)
        try:
            truly_naive_time = benchmark_kernel(
                triton_truly_naive, Q, K, V, scale, False,
                warmup=args.warmup, iters=args.iters
            )
            results["truly_naive"][seq_len] = truly_naive_time
            print(f"  Triton (truly naive): {truly_naive_time:.3f} ms")
        except Exception as e:
            print(f"  Triton (truly naive): FAILED ({e})")
            results["truly_naive"][seq_len] = float('inf')

        # Hybrid 3-pass (non-causal)
        if HYBRID_3PASS_AVAILABLE:
            try:
                hybrid_3pass_time = benchmark_kernel(
                    hybrid_3pass, Q, K, V, scale, False,  # non-causal
                    warmup=args.warmup, iters=args.iters
                )
                results["hybrid_3pass"][seq_len] = hybrid_3pass_time
                print(f"  Hybrid 3-pass:        {hybrid_3pass_time:.3f} ms")
            except Exception as e:
                print(f"  Hybrid 3-pass:        FAILED ({e})")
                results["hybrid_3pass"][seq_len] = float('inf')
        else:
            results["hybrid_3pass"][seq_len] = float('inf')

        # Hybrid 2-pass (non-causal)
        if HYBRID_2PASS_AVAILABLE:
            try:
                hybrid_2pass_time = benchmark_kernel(
                    hybrid_2pass, Q, K, V, scale, False,  # non-causal
                    warmup=args.warmup, iters=args.iters
                )
                results["hybrid_2pass"][seq_len] = hybrid_2pass_time
                print(f"  Hybrid 2-pass:        {hybrid_2pass_time:.3f} ms")
            except Exception as e:
                print(f"  Hybrid 2-pass:        FAILED ({e})")
                results["hybrid_2pass"][seq_len] = float('inf')
        else:
            results["hybrid_2pass"][seq_len] = float('inf')

        # PyTorch graph-driven (non-causal)
        if PYTORCH_GRAPHDRIVEN_AVAILABLE:
            try:
                pytorch_gd_time = benchmark_kernel(
                    pytorch_graphdriven, Q, K, V, scale, False,  # non-causal
                    warmup=args.warmup, iters=args.iters
                )
                results["pytorch_gd"][seq_len] = pytorch_gd_time
                print(f"  PyTorch GD:           {pytorch_gd_time:.3f} ms")
            except Exception as e:
                print(f"  PyTorch GD:           FAILED ({e})")
                results["pytorch_gd"][seq_len] = float('inf')
        else:
            results["pytorch_gd"][seq_len] = float('inf')

        # PyTorch compiled (non-causal)
        if PYTORCH_GRAPHDRIVEN_AVAILABLE:
            try:
                pytorch_compiled_time = benchmark_kernel(
                    pytorch_compiled, Q, K, V, scale, False,  # non-causal
                    warmup=args.warmup, iters=args.iters
                )
                results["pytorch_compiled"][seq_len] = pytorch_compiled_time
                print(f"  PyTorch Compiled:     {pytorch_compiled_time:.3f} ms")
            except Exception as e:
                print(f"  PyTorch Compiled:     FAILED ({e})")
                results["pytorch_compiled"][seq_len] = float('inf')
        else:
            results["pytorch_compiled"][seq_len] = float('inf')

        # Triton graph-driven (non-causal)
        if TRITON_GRAPHDRIVEN_AVAILABLE:
            try:
                triton_gd_time = benchmark_kernel(
                    triton_graphdriven, Q, K, V, scale, False,  # non-causal
                    warmup=args.warmup, iters=args.iters
                )
                results["triton_gd"][seq_len] = triton_gd_time
                print(f"  Triton Graph-Driven:  {triton_gd_time:.3f} ms")
            except Exception as e:
                print(f"  Triton Graph-Driven:  FAILED ({e})")
                results["triton_gd"][seq_len] = float('inf')
        else:
            results["triton_gd"][seq_len] = float('inf')

        # Clear cache between sequence lengths
        torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    header = f"{'Seq Len':<10} {'Eager':<10} {'SDPA':<10} {'Triton':<10} {'TrulyNaive':<12} {'Hybrid3P':<10} {'Hybrid2P':<10} {'PyTorchGD':<12} {'Compiled':<10} {'TritonGD':<10}"
    print(header)
    print("-" * 106)

    for seq_len in seq_lengths:
        eager_t = results["eager"].get(seq_len, float('inf'))
        sdpa_t = results["sdpa"].get(seq_len, float('inf'))
        triton_t = results["triton"].get(seq_len, float('inf'))
        truly_naive_t = results["truly_naive"].get(seq_len, float('inf'))
        hybrid_3pass_t = results["hybrid_3pass"].get(seq_len, float('inf'))
        hybrid_2pass_t = results["hybrid_2pass"].get(seq_len, float('inf'))
        pytorch_gd_t = results["pytorch_gd"].get(seq_len, float('inf'))
        pytorch_compiled_t = results["pytorch_compiled"].get(seq_len, float('inf'))
        triton_gd_t = results["triton_gd"].get(seq_len, float('inf'))

        row = f"{seq_len:<10} "
        row += f"{eager_t:<10.2f} " if eager_t != float('inf') else f"{'SKIP':<10} "
        row += f"{sdpa_t:<10.2f} " if sdpa_t != float('inf') else f"{'FAIL':<10} "
        row += f"{triton_t:<10.2f} " if triton_t != float('inf') else f"{'FAIL':<10} "
        row += f"{truly_naive_t:<12.2f} " if truly_naive_t != float('inf') else f"{'FAIL':<12} "
        row += f"{hybrid_3pass_t:<10.2f} " if hybrid_3pass_t != float('inf') else f"{'N/A':<10} "
        row += f"{hybrid_2pass_t:<10.2f} " if hybrid_2pass_t != float('inf') else f"{'N/A':<10} "
        row += f"{pytorch_gd_t:<12.2f} " if pytorch_gd_t != float('inf') else f"{'N/A':<12} "
        row += f"{pytorch_compiled_t:<10.2f} " if pytorch_compiled_t != float('inf') else f"{'N/A':<10} "
        row += f"{triton_gd_t:<10.2f}" if triton_gd_t != float('inf') else f"{'N/A':<10}"

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
        if HYBRID_3PASS_AVAILABLE:
            ax1.plot(valid_seq_lens, [results["hybrid_3pass"][s] for s in valid_seq_lens],
                    marker='x', linewidth=2, markersize=8, label='Hybrid 3-pass', color='#f39c12')
        if HYBRID_2PASS_AVAILABLE:
            ax1.plot(valid_seq_lens, [results["hybrid_2pass"][s] for s in valid_seq_lens],
                    marker='*', linewidth=2, markersize=10, label='Hybrid 2-pass', color='#1abc9c')
        if PYTORCH_GRAPHDRIVEN_AVAILABLE:
            pytorch_gd_times = [results["pytorch_gd"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in pytorch_gd_times):
                ax1.plot(valid_seq_lens, pytorch_gd_times,
                        marker='p', linewidth=2, markersize=8, label='PyTorch GD', color='#e67e22')
            pytorch_compiled_times = [results["pytorch_compiled"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in pytorch_compiled_times):
                ax1.plot(valid_seq_lens, pytorch_compiled_times,
                        marker='h', linewidth=2, markersize=8, label='PyTorch Compiled', color='#8e44ad')
        if TRITON_GRAPHDRIVEN_AVAILABLE:
            triton_gd_times = [results["triton_gd"].get(s, float('inf')) for s in valid_seq_lens]
            if not all(t == float('inf') for t in triton_gd_times):
                ax1.plot(valid_seq_lens, triton_gd_times,
                        marker='D', linewidth=2, markersize=8, label='Triton GD', color='#16a085')

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
        f.write("Sequence_Length,Eager_ms,SDPA_ms,Triton_online_ms,Triton_3pass_ms,Hybrid_3pass_ms,Hybrid_2pass_ms,PyTorch_GD_ms,PyTorch_Compiled_ms,Triton_GD_ms\n")
        for seq_len in valid_seq_lens:
            eager_t = results["eager"].get(seq_len, float('inf'))
            truly_naive_t = results["truly_naive"].get(seq_len, float('inf'))
            hybrid_3pass_t = results["hybrid_3pass"].get(seq_len, float('inf'))
            hybrid_2pass_t = results["hybrid_2pass"].get(seq_len, float('inf'))
            pytorch_gd_t = results["pytorch_gd"].get(seq_len, float('inf'))
            pytorch_compiled_t = results["pytorch_compiled"].get(seq_len, float('inf'))
            triton_gd_t = results["triton_gd"].get(seq_len, float('inf'))
            f.write(f"{seq_len},{eager_t:.3f},{results['sdpa'][seq_len]:.3f},"
                    f"{results['triton'][seq_len]:.3f},{truly_naive_t:.3f},"
                    f"{hybrid_3pass_t:.3f},{hybrid_2pass_t:.3f},"
                    f"{pytorch_gd_t:.3f},{pytorch_compiled_t:.3f},{triton_gd_t:.3f}\n")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
