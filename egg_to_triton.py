#!/usr/bin/env python3
"""
Egglog to Triton Compiler

Translates egglog attention representations (JSON) into executable Triton kernels.

Usage:
    python egg_to_triton.py attention.json -o generated_attention.py
    python egg_to_triton.py attention.json --strategy composable
    python egg_to_triton.py attention.json --strategy fused

Pipeline:
    attention.egg -> egglog --to-json -> attention.json -> egg_to_triton.py -> generated_attention.py
"""

import argparse
import sys
import os

from egg_parser import EggParser, print_graph
from triton_codegen import generate_code


def main():
    parser = argparse.ArgumentParser(
        description='Compile egglog attention JSON to Triton code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Generate fused Triton kernel (default)
    python egg_to_triton.py attention.json

    # Generate composable PyTorch operations
    python egg_to_triton.py attention.json --strategy composable

    # Custom output file
    python egg_to_triton.py attention.json -o my_attention.py

    # Show computation graph without generating code
    python egg_to_triton.py attention.json --print-graph
        '''
    )

    parser.add_argument(
        'json_file',
        help='Path to egglog JSON file (from `egglog --to-json`)'
    )
    parser.add_argument(
        '-o', '--output',
        default='generated_attention.py',
        help='Output Python file (default: generated_attention.py)'
    )
    parser.add_argument(
        '--strategy',
        choices=['composable', 'fused'],
        default='fused',
        help='Code generation strategy (default: fused)'
    )
    parser.add_argument(
        '--print-graph',
        action='store_true',
        help='Print computation graph and exit'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON
    if args.verbose:
        print(f"Parsing {args.json_file}...")

    egg_parser = EggParser(args.json_file)
    graph = egg_parser.parse()

    if args.verbose:
        print(f"  Found {len(graph.nodes)} nodes")
        print(f"  Inputs: {list(graph.inputs.keys())}")
        print(f"  Unique ops: {graph.unique_ops}")

    # Print graph and exit if requested
    if args.print_graph:
        print_graph(graph)
        return

    # Generate code
    if args.verbose:
        print(f"Generating {args.strategy} code...")

    code = generate_code(graph, strategy=args.strategy)

    # Write output
    with open(args.output, 'w') as f:
        f.write(code)

    print(f"Generated: {args.output}")
    print(f"Strategy: {args.strategy}")
    print(f"Operations: {len(graph.unique_ops)}")

    if args.verbose:
        print("\nTo test the generated code:")
        print(f"  python {args.output}")


if __name__ == '__main__':
    main()
