#!/usr/bin/env python3
"""
Generate Triton Graph-Driven Kernel from Egglog JSON

Usage:
    python generate_triton_kernel.py attention.json --output generated_triton_3pass.py
    python generate_triton_kernel.py attention_2pass.json --output generated_triton_2pass.py
"""

import argparse
from egg_parser import EggParser
from operation_emitter import HoleEmitter
from kernel_scaffolds import TwoPassScaffold

def main():
    parser = argparse.ArgumentParser(description="Generate Triton kernel from egglog JSON")
    parser.add_argument("json_file", help="Path to egglog JSON file (e.g., attention.json)")
    parser.add_argument("--output", "-o", default="generated_triton_auto.py",
                       help="Output Python file (default: generated_triton_auto.py)")
    args = parser.parse_args()

    print(f"Parsing: {args.json_file}")

    # Parse the computation graph
    egg_parser = EggParser(args.json_file)
    graph = egg_parser.parse()

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Unique ops: {len(graph.unique_ops)}")

    # Generate kernel using graph-driven approach
    emitter = HoleEmitter(graph, TwoPassScaffold())  # Scaffold is just placeholder
    kernel_code = emitter.generate_kernel_from_graph()

    # Analyze passes
    from pass_analyzer import PassAnalyzer
    analyzer = PassAnalyzer(graph)
    passes = analyzer.analyze()
    mem_info = analyzer.get_memory_pass_info()

    print(f"\nPass Analysis:")
    print(f"  Memory passes: {mem_info.num_memory_passes}")
    print(f"  Sync levels: {mem_info.sync_levels}")
    print(f"  Post-loop ops: {len(mem_info.post_loop_ops)}")

    # Write to file
    with open(args.output, 'w') as f:
        f.write('"""\n')
        f.write(f'Graph-Driven Triton Kernel - Auto-generated from {args.json_file}\n')
        f.write(f'\nMemory passes: {mem_info.num_memory_passes}\n')
        f.write(f'Sync levels: {mem_info.sync_levels}\n')
        f.write(f'Post-loop operations: {len(mem_info.post_loop_ops)}\n')
        f.write('"""\n\n')
        f.write('import torch\n')
        f.write('import triton\n')
        f.write('import triton.language as tl\n')
        f.write('import math\n\n')
        f.write(kernel_code)
        f.write('\n\n# Wrapper function\n')
        f.write(generate_wrapper(args.json_file))

    print(f"\nKernel written to: {args.output}")
    print("\nYou can now import and use it:")
    print(f"  from {args.output.replace('.py', '')} import attention_kernel")

def generate_wrapper(json_filename):
    """Generate wrapper function for the kernel."""
    basename = json_filename.replace('.json', '').replace('attention_', '').replace('attention', 'auto')

    return f'''
def attention_kernel_{basename}(Q, K, V, scale=None, causal=False):
    """
    Graph-driven attention kernel generated from {json_filename}

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

    out = attention_kernel_{basename}(Q, K, V)
    print(f"Output shape: {{out.shape}}")
    print("Test passed!")
'''

if __name__ == "__main__":
    main()
