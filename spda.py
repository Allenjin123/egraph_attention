import torch
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for terminal
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
import gc

# Import Triton attention
try:
    from triton_attention import inject_triton_attention, verify_triton_correctness, verify_model_correctness
    TRITON_AVAILABLE = True
    print("Triton attention available")
except ImportError as e:
    TRITON_AVAILABLE = False
    print(f"Triton attention not available: {e}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches for averaging")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Model to use")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token for gated models")
    parser.add_argument("--seq-lengths", type=str, default="256,512,1024", help="Comma-separated sequence lengths")
    parser.add_argument("--server", action="store_true", help="Use /scratch cache directory (for server environment)")
    return parser


def setup_cache(use_server: bool):
    """Set HuggingFace cache directory based on environment."""
    if use_server:
        os.environ['HF_HOME'] = '/scratch/nbleier_root/nbleier0/allenjin/hf'
        os.environ['HF_HUB_CACHE'] = '/scratch/nbleier_root/nbleier0/allenjin/hf/hub'
        print(f"Using server cache: {os.environ['HF_HOME']}")
    else:
        print(f"Using default cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

@torch.no_grad()
def warmup_and_benchmark(model, tokenizer, max_seq_len, num_batches, max_new_tokens):
    """Benchmark model generation with given sequence length"""
    model_max_length = model.config.max_position_embeddings
    if max_seq_len > model_max_length:
        print(f"    WARNING: Requested seq_len {max_seq_len} exceeds model's max {model_max_length}")
        print(f"    Clamping to {model_max_length}")
        max_seq_len = model_max_length

    text = "The quick brown fox jumps over the lazy dog. " * (max_seq_len * 2)
    temp_inputs = tokenizer(text, return_tensors="pt")
    input_ids = temp_inputs.input_ids[:, :max_seq_len]
    attention_mask = torch.ones_like(input_ids)

    inputs = {
        'input_ids': input_ids.to("cuda"),
        'attention_mask': attention_mask.to("cuda")
    }

    print(f"    Actual input shape: {input_ids.shape}, requested: {max_seq_len}")
    
    # Warmup
    print(f"    Warming up...")
    _ = model.generate(
        **inputs,
        max_new_tokens=min(20, max_new_tokens),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    
    # Benchmark
    print(f"    Benchmarking {num_batches} batches...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    start_event.record()
    for i in range(num_batches):
        _ = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if i < num_batches - 1:
            torch.cuda.empty_cache()
    end_event.record()
    torch.cuda.synchronize()
    
    total_time = start_event.elapsed_time(end_event) * 1.0e-3
    return total_time / num_batches

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup cache directory before loading any models
    setup_cache(args.server)

    model_id = args.model
    token = args.token
    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]
    
    print(f"Using model: {model_id}")
    print(f"Sequence lengths: {seq_lengths}")
    
    if not torch.cuda.is_available():
        print("CUDA not available! This script requires a GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Verify Triton correctness before benchmarking
    global TRITON_AVAILABLE
    if TRITON_AVAILABLE:
        print("\n" + "="*60)
        print("VERIFYING TRITON CORRECTNESS (Kernel)")
        print("="*60)
        if verify_triton_correctness():
            print("Kernel verification PASSED")
        else:
            print("WARNING: Kernel verification FAILED - skipping Triton benchmark")
            TRITON_AVAILABLE = False

    # Model-level verification (more thorough)
    if TRITON_AVAILABLE:
        print("\n" + "="*60)
        print("VERIFYING TRITON CORRECTNESS (Model)")
        print("="*60)
        if verify_model_correctness(model_id, seq_len=256, token=token):
            print("Model verification PASSED - proceeding with benchmark")
        else:
            print("WARNING: Model verification FAILED - skipping Triton benchmark")
            TRITON_AVAILABLE = False

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Results storage
    native_times = {}
    sdpa_times = {}
    triton_times = {}
    speedups_sdpa = {}
    speedups_triton = {}
    
    print("\n" + "="*60)
    print("STARTING BENCHMARK")
    print("="*60)
    
    for seq_len in seq_lengths:
        print(f"\n[Sequence Length: {seq_len}]")

        # Test Native Attention
        print("  Testing Native Attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            token=token,
        )
        print(f"  Model max position embeddings: {model.config.max_position_embeddings}")
        
        try:
            native_time = warmup_and_benchmark(
                model, tokenizer, seq_len, args.num_batches, args.max_new_tokens
            )
            native_times[seq_len] = native_time
            print(f"    Native time: {native_time:.3f}s")
        except Exception as e:
            print(f"    Error with native attention: {e}")
            native_times[seq_len] = float('inf')
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Test SDPA
        print("  Testing SDPA (Optimized Attention)...")
        model_sdpa = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            token=token,
        )
        
        try:
            sdpa_time = warmup_and_benchmark(
                model_sdpa, tokenizer, seq_len, args.num_batches, args.max_new_tokens
            )
            sdpa_times[seq_len] = sdpa_time
            print(f"    SDPA time: {sdpa_time:.3f}s")

            speedup = native_times[seq_len] / sdpa_time if sdpa_time > 0 else 0
            speedups_sdpa[seq_len] = speedup
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error with SDPA: {e}")
            sdpa_times[seq_len] = float('inf')
            speedups_sdpa[seq_len] = 0

        del model_sdpa
        gc.collect()
        torch.cuda.empty_cache()

        # Test Triton Attention
        if TRITON_AVAILABLE:
            print("  Testing Triton Attention...")
            model_triton = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager",  # Start with eager, then inject
                token=token,
            )
            inject_triton_attention(model_triton)

            try:
                triton_time = warmup_and_benchmark(
                    model_triton, tokenizer, seq_len, args.num_batches, args.max_new_tokens
                )
                triton_times[seq_len] = triton_time
                print(f"    Triton time: {triton_time:.3f}s")

                speedup = native_times[seq_len] / triton_time if triton_time > 0 else 0
                speedups_triton[seq_len] = speedup
                print(f"    Speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"    Error with Triton: {e}")
                triton_times[seq_len] = float('inf')
                speedups_triton[seq_len] = 0

            del model_triton
            gc.collect()
            torch.cuda.empty_cache()
    
    # Create results directory
    results_dir = "attention_benchmarks"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plotting
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid_seq_lens = [s for s in seq_lengths if s in native_times and native_times[s] != float('inf')]

    # Plot 1: Time Comparison
    ax1 = axes[0]
    if valid_seq_lens:
        ax1.plot(valid_seq_lens, [native_times[s] for s in valid_seq_lens],
                marker='o', linewidth=2, markersize=8, label='Native Attention', color='#3498db')
        ax1.plot(valid_seq_lens, [sdpa_times[s] for s in valid_seq_lens],
                marker='s', linewidth=2, markersize=8, label='SDPA', color='#e74c3c')
        if TRITON_AVAILABLE and triton_times:
            ax1.plot(valid_seq_lens, [triton_times.get(s, float('inf')) for s in valid_seq_lens],
                    marker='^', linewidth=2, markersize=8, label='Triton Naive', color='#2ecc71')

    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Average Time (seconds)', fontsize=12)
    ax1.set_title('Attention Mechanism Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup (grouped bar chart)
    ax2 = axes[1]
    if valid_seq_lens:
        x = range(len(valid_seq_lens))
        width = 0.35 if not (TRITON_AVAILABLE and triton_times) else 0.25

        # SDPA bars
        sdpa_speedups = [speedups_sdpa.get(s, 0) for s in valid_seq_lens]
        bars1 = ax2.bar([i - width/2 for i in x] if TRITON_AVAILABLE and triton_times else x,
                        sdpa_speedups, width, label='SDPA', color='#e74c3c', alpha=0.8)

        # Triton bars (if available)
        if TRITON_AVAILABLE and triton_times:
            triton_speedups = [speedups_triton.get(s, 0) for s in valid_seq_lens]
            bars2 = ax2.bar([i + width/2 for i in x], triton_speedups, width,
                           label='Triton Naive', color='#2ecc71', alpha=0.8)

        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup (1x)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_seq_lens)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.2f}x', ha='center', fontsize=9)
        if TRITON_AVAILABLE and triton_times:
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.2f}x', ha='center', fontsize=9)

    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('Speedup over Native Attention', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot (no plt.show() since we're in terminal)
    safe_model_name = model_id.replace("/", "_")
    plot_path = os.path.join(results_dir, f'{safe_model_name}_benchmark.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    print(f"Plot saved to: {plot_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    if TRITON_AVAILABLE and triton_times:
        print(f"{'Seq Len':<10} {'Native (s)':<12} {'SDPA (s)':<12} {'Triton (s)':<12} {'SDPA Spd':<10} {'Triton Spd':<10}")
        print("-" * 66)
        for seq_len in valid_seq_lens:
            triton_t = triton_times.get(seq_len, float('inf'))
            triton_spd = speedups_triton.get(seq_len, 0)
            print(f"{seq_len:<10} {native_times[seq_len]:<12.3f} {sdpa_times[seq_len]:<12.3f} "
                  f"{triton_t:<12.3f} {speedups_sdpa[seq_len]:<10.2f}x {triton_spd:<10.2f}x")
    else:
        print(f"{'Seq Length':<12} {'Native (s)':<12} {'SDPA (s)':<12} {'Speedup':<10}")
        print("-" * 46)
        for seq_len in valid_seq_lens:
            print(f"{seq_len:<12} {native_times[seq_len]:<12.3f} {sdpa_times[seq_len]:<12.3f} {speedups_sdpa[seq_len]:<10.2f}x")

    # Save results to CSV
    csv_path = os.path.join(results_dir, f'{safe_model_name}_results.csv')
    with open(csv_path, 'w') as f:
        if TRITON_AVAILABLE and triton_times:
            f.write("Sequence_Length,Native_Time,SDPA_Time,Triton_Time,SDPA_Speedup,Triton_Speedup\n")
            for seq_len in valid_seq_lens:
                triton_t = triton_times.get(seq_len, float('inf'))
                triton_spd = speedups_triton.get(seq_len, 0)
                f.write(f"{seq_len},{native_times[seq_len]:.3f},{sdpa_times[seq_len]:.3f},"
                        f"{triton_t:.3f},{speedups_sdpa[seq_len]:.2f},{triton_spd:.2f}\n")
        else:
            f.write("Sequence_Length,Native_Time,SDPA_Time,Speedup\n")
            for seq_len in valid_seq_lens:
                f.write(f"{seq_len},{native_times[seq_len]:.3f},{sdpa_times[seq_len]:.3f},{speedups_sdpa[seq_len]:.2f}\n")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
