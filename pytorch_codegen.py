"""
Graph-Driven PyTorch Code Generator

Generates PyTorch code from the parsed computation graph using proper
dimension-aware broadcasting. Uses DimResolver to compute correct unsqueeze
positions rather than hardcoded indices.

This serves as:
1. Correctness reference for Triton kernels
2. Proof that graph-driven generation works
3. Foundation for the dimension resolution logic
"""

import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from egg_parser import ComputationGraph, IRNode
from egg_ops import OP_REGISTRY, DimSpec, get_op_spec
from dim_resolver import (
    DimResolver,
    compute_unsqueeze_sequence,
    compute_reduce_axis,
    get_resolver,
)


# ============================================================================
# Shape Computer
# ============================================================================

class ShapeComputer:
    """
    Generates PyTorch reshape/unsqueeze/reduce code based on dimension specs.
    """

    def __init__(self):
        self.resolver = get_resolver()

    def generate_broadcast(
        self,
        var_name: str,
        input_dims: DimSpec,
        output_dims: DimSpec,
    ) -> Tuple[str, str]:
        """
        Generate code to broadcast a tensor to target dimensions.

        Args:
            var_name: Variable name of the input tensor
            input_dims: Current dimensions of the tensor
            output_dims: Target dimensions after broadcasting

        Returns:
            Tuple of (code, result_var_name)

        Example:
            generate_broadcast("Q", DimSpec(('e','p')), DimSpec(('e','m','p')))
            returns ("Q_bc = Q.unsqueeze(1)", "Q_bc")
        """
        unsqueeze_positions = self.resolver.compute_unsqueeze_sequence(
            input_dims, output_dims
        )

        if not unsqueeze_positions:
            # No unsqueeze needed
            return ("", var_name)

        # Generate chain of unsqueeze operations
        result_var = f"{var_name}_bc"
        expr = var_name

        for pos in unsqueeze_positions:
            expr = f"{expr}.unsqueeze({pos})"

        code = f"{result_var} = {expr}"
        return (code, result_var)

    def generate_reduce(
        self,
        var_name: str,
        input_dims: DimSpec,
        reduce_dim: str,
        reduce_op: str = 'sum',
    ) -> Tuple[str, str, DimSpec]:
        """
        Generate code for a reduction operation.

        Args:
            var_name: Variable name of the input tensor
            input_dims: Current dimensions of the tensor
            reduce_dim: Dimension to reduce over
            reduce_op: 'sum' or 'max'

        Returns:
            Tuple of (code, result_var_name, output_dims)

        Example:
            generate_reduce("x", DimSpec(('e','m','p')), 'e', 'sum')
            returns ("x_reduced = x.sum(dim=0)", "x_reduced", DimSpec(('m','p')))
        """
        axis = self.resolver.compute_reduce_axis(input_dims, reduce_dim)
        output_dims = input_dims.without(reduce_dim)

        result_var = f"{var_name}_red"

        if reduce_op == 'sum':
            code = f"{result_var} = {var_name}.sum(dim={axis})"
        elif reduce_op == 'max':
            code = f"{result_var} = {var_name}.max(dim={axis})[0]"
        else:
            raise ValueError(f"Unknown reduce op: {reduce_op}")

        return (code, result_var, output_dims)


# ============================================================================
# Operation Code Generator
# ============================================================================

@dataclass
class OpResult:
    """Result of generating code for an operation."""
    code_lines: List[str]
    result_var: str
    output_dims: DimSpec


class OpCodeGenerator:
    """
    Generates PyTorch code for individual operations using graph information.

    Unlike the hardcoded approach, this uses the actual dimension specs
    from the graph to compute correct broadcast positions.
    """

    def __init__(self, graph: ComputationGraph, batch_dim_offset: int = 1):
        self.graph = graph
        self.shape_computer = ShapeComputer()
        self.resolver = get_resolver()
        # Offset for batch dimension (tensors have [batch, ...] prepended)
        self.batch_dim_offset = batch_dim_offset

        # Track intermediate variables
        self.var_counter = 0

        # Track if we need to apply scale (after QK dot product)
        # The QK dot product is: R_add_e following M_mul_emp
        self._qk_reduction_node = self._find_qk_reduction()

    def _find_qk_reduction(self) -> Optional[str]:
        """Find the node ID of the R_add_e that follows M_mul_emp or M_mul_em1m0p (QK dot product)."""
        for node_id, node in self.graph.nodes.items():
            if node.op == 'R_add_e':
                # Check if its child is M_mul_emp or M_mul_em1m0p (tiled version)
                if node.children:
                    child_id = node.children[0]
                    if child_id in self.graph.nodes:
                        child = self.graph.nodes[child_id]
                        if child.op in ('M_mul_emp', 'M_mul_em1m0p'):
                            return node_id
        return None

    def generate_op(self, node: IRNode) -> OpResult:
        """
        Generate code for a single operation.

        Args:
            node: IRNode from the computation graph

        Returns:
            OpResult with code lines, result variable, and output dimensions
        """
        op = node.op
        spec = get_op_spec(op)

        if spec is None:
            # Unknown op - return placeholder
            return self._generate_unknown(node)

        if spec.op_type == 'map':
            return self._generate_map_op(node, spec)
        elif spec.op_type == 'reduce':
            return self._generate_reduce_op(node, spec)
        elif spec.op_type == 'tile':
            return self._generate_tile_op(node, spec)
        else:
            return self._generate_unknown(node)

    def _get_child_info(self, child_id: str) -> Tuple[str, Optional[DimSpec]]:
        """Get variable name and dimensions for a child node."""
        if child_id in self.graph.nodes:
            child = self.graph.nodes[child_id]
            return (child.var_name, child.output_dims)
        return (child_id, None)

    def _generate_broadcast_with_offset(
        self,
        var_name: str,
        input_dims: DimSpec,
        output_dims: DimSpec,
    ) -> Tuple[str, str]:
        """
        Generate broadcast code with batch dimension offset.

        The actual tensors have a batch dimension prepended, so we need to
        offset all unsqueeze positions by batch_dim_offset.
        """
        unsqueeze_positions = self.resolver.compute_unsqueeze_sequence(
            input_dims, output_dims
        )

        if not unsqueeze_positions:
            return ("", var_name)

        # Apply batch dimension offset to all positions
        adjusted_positions = [pos + self.batch_dim_offset for pos in unsqueeze_positions]

        result_var = f"{var_name}_bc"
        expr = var_name

        for pos in adjusted_positions:
            expr = f"{expr}.unsqueeze({pos})"

        code = f"{result_var} = {expr}"
        return (code, result_var)

    def _generate_map_op(self, node: IRNode, spec) -> OpResult:
        """Generate code for a map (element-wise) operation."""
        code_lines = []
        output_dims = node.output_dims or spec.output_dims

        # Get child information
        children_info = [self._get_child_info(c) for c in node.children]

        # For binary ops, we need to broadcast both operands
        if len(children_info) == 2:
            var_a, dims_a = children_info[0]
            var_b, dims_b = children_info[1]

            # Generate broadcast code for each operand (with batch dim offset)
            bc_code_a, bc_var_a = self._generate_broadcast_with_offset(
                var_a, dims_a, output_dims
            ) if dims_a else ("", var_a)

            bc_code_b, bc_var_b = self._generate_broadcast_with_offset(
                var_b, dims_b, output_dims
            ) if dims_b else ("", var_b)

            if bc_code_a:
                code_lines.append(bc_code_a)
            if bc_code_b:
                code_lines.append(bc_code_b)

            # Generate the operation
            triton_op = spec.triton_op
            result_var = node.var_name

            if triton_op == '*':
                code_lines.append(f"{result_var} = {bc_var_a} * {bc_var_b}")
            elif triton_op == '+':
                code_lines.append(f"{result_var} = {bc_var_a} + {bc_var_b}")
            elif triton_op == '-':
                code_lines.append(f"{result_var} = {bc_var_a} - {bc_var_b}")
            elif triton_op == '/':
                code_lines.append(f"{result_var} = {bc_var_a} / {bc_var_b}")
            else:
                code_lines.append(f"{result_var} = {triton_op}({bc_var_a}, {bc_var_b})")

        # For unary ops (like exp)
        elif len(children_info) == 1:
            var_a, dims_a = children_info[0]
            result_var = node.var_name
            triton_op = spec.triton_op

            if triton_op == 'tl.exp':
                code_lines.append(f"{result_var} = torch.exp({var_a})")
            else:
                code_lines.append(f"{result_var} = {triton_op}({var_a})")

        else:
            return self._generate_unknown(node)

        return OpResult(
            code_lines=code_lines,
            result_var=node.var_name,
            output_dims=output_dims,
        )

    def _generate_reduce_op(self, node: IRNode, spec) -> OpResult:
        """Generate code for a reduce operation."""
        code_lines = []

        # Get input info
        var_a, dims_a = self._get_child_info(node.children[0])

        if dims_a is None:
            return self._generate_unknown(node)

        # Compute reduce axis (with batch dim offset)
        reduce_dim = spec.reduce_dim
        axis = self.resolver.compute_reduce_axis(dims_a, reduce_dim)
        axis += self.batch_dim_offset  # Account for batch dimension

        result_var = node.var_name
        output_dims = dims_a.without(reduce_dim)

        triton_op = spec.triton_op
        if triton_op == 'tl.sum':
            code_lines.append(f"{result_var} = {var_a}.sum(dim={axis})")
        elif triton_op == 'tl.max':
            code_lines.append(f"{result_var} = {var_a}.max(dim={axis})[0]")
        else:
            code_lines.append(f"# Unknown reduce op: {triton_op}")
            code_lines.append(f"{result_var} = {var_a}.sum(dim={axis})")

        # Apply scale after QK dot product (R_add_e following M_mul_emp)
        if self._qk_reduction_node and node.id == self._qk_reduction_node:
            code_lines.append(f"{result_var} = {result_var} * scale  # Apply attention scale")

        return OpResult(
            code_lines=code_lines,
            result_var=result_var,
            output_dims=output_dims,
        )

    def _generate_tile_op(self, node: IRNode, spec) -> OpResult:
        """Generate code for tiling operations."""
        code_lines = []
        var_a, dims_a = self._get_child_info(node.children[0])
        result_var = node.var_name

        if node.op == 'T_split_m_m1m0':
            # Split m into m1 x m0: [batch, e, m] -> [batch, e, m1, m0]
            # In PyTorch: unflatten(dim, sizes)
            # Compute tile sizes dynamically based on sequence length
            code_lines.append(
                f"# Compute tile sizes: aim for 4 tiles by default"
            )
            code_lines.append(
                f"_m_size = {var_a}.shape[-1]"
            )
            code_lines.append(
                f"_num_tiles = min(4, _m_size)  # At most 4 tiles"
            )
            code_lines.append(
                f"_tile_size = _m_size // _num_tiles"
            )
            code_lines.append(
                f"{result_var} = {var_a}.unflatten(-1, (_num_tiles, _tile_size))"
            )
            output_dims = DimSpec(('e', 'm1', 'm0'))

        elif node.op == 'T_unsplit_m1m0_m':
            # Merge m1 x m0 back to m: [batch, m1, m0, p] -> [batch, m, p]
            # Find the axis positions for m1 and m0 (with batch offset)
            if dims_a:
                try:
                    m1_axis = dims_a.dims.index('m1') + self.batch_dim_offset
                    code_lines.append(
                        f"{result_var} = {var_a}.flatten({m1_axis}, {m1_axis + 1})"
                    )
                except ValueError:
                    code_lines.append(f"# Error: m1 not found in {dims_a}")
                    code_lines.append(f"{result_var} = {var_a}")
            else:
                code_lines.append(f"{result_var} = {var_a}.flatten(0, 1)")
            output_dims = spec.output_dims

        else:
            return self._generate_unknown(node)

        return OpResult(
            code_lines=code_lines,
            result_var=result_var,
            output_dims=output_dims,
        )

    def _generate_unknown(self, node: IRNode) -> OpResult:
        """Generate placeholder for unknown operations."""
        children = [self._get_child_info(c)[0] for c in node.children]
        args = ', '.join(children)
        code_lines = [f"# TODO: implement {node.op}"]
        code_lines.append(f"{node.var_name} = {node.op}({args})  # PLACEHOLDER")

        return OpResult(
            code_lines=code_lines,
            result_var=node.var_name,
            output_dims=node.output_dims or DimSpec(()),
        )


# ============================================================================
# Main PyTorch Code Generator
# ============================================================================

IMPORTS_TEMPLATE = '''"""
Graph-Driven Attention Implementation (PyTorch)
Generated by: pytorch_codegen.py

This implementation uses proper dimension-aware broadcasting computed
from the egglog computation graph.
"""

import torch
import math
from typing import Optional

'''


class PyTorchCodeGenerator:
    """
    Graph-driven PyTorch code generator.

    Generates correct PyTorch code by using the dimension specs from the
    parsed computation graph to compute broadcast positions dynamically.
    """

    def __init__(self, graph: ComputationGraph):
        self.graph = graph
        self.op_codegen = OpCodeGenerator(graph)

    def generate(self) -> str:
        """Generate complete PyTorch implementation."""
        code = [IMPORTS_TEMPLATE]
        code.append(self._generate_constants())
        code.append(self._generate_attention_function())
        code.append(self._generate_test())
        return '\n'.join(code)

    def _generate_constants(self) -> str:
        """Generate configuration constants."""
        # Check if tiling is used
        ops = self.graph.unique_ops
        uses_tiling = any(op.startswith('T_') or 'm1' in op or 'm0' in op
                         for op in ops)

        if uses_tiling:
            return '''
# Tiling configuration - will be set dynamically based on sequence length
# Default: 4 tiles with tile size = seq_len // 4
'''
        return ''

    def _generate_attention_function(self) -> str:
        """Generate the main attention function."""
        lines = []
        lines.append("def egg_attention_graphdriven(Q, K, V, scale=None):")
        lines.append('    """')
        lines.append('    Attention implementation generated from egglog computation graph.')
        lines.append('    Uses dimension-aware broadcasting computed from graph.')
        lines.append('    ')
        lines.append('    Args:')
        lines.append('        Q: Query tensor [batch, heads, seq_q, dim] or [dim, seq]')
        lines.append('        K: Key tensor [batch, heads, seq_k, dim] or [dim, seq]')
        lines.append('        V: Value tensor [batch, heads, seq_k, dim] or [dim, seq]')
        lines.append('        scale: Scaling factor (default: 1/sqrt(dim))')
        lines.append('    ')
        lines.append('    Returns:')
        lines.append('        Output tensor same shape as Q')
        lines.append('    """')
        lines.append('')

        # Input handling
        lines.append('    # ===== Input Shape Handling =====')
        lines.append('    original_shape = Q.shape')
        lines.append('    if Q.dim() == 4:')
        lines.append('        # [batch, heads, seq, dim] format')
        lines.append('        batch_size, num_heads, seq_len, head_dim = Q.shape')
        lines.append('        # Reshape to [batch*heads, dim, seq] for e/p dimensions')
        lines.append('        Q = Q.permute(0, 1, 3, 2).reshape(-1, head_dim, seq_len)')
        lines.append('        K = K.permute(0, 1, 3, 2).reshape(-1, head_dim, seq_len)')
        lines.append('        V = V.permute(0, 1, 3, 2).reshape(-1, head_dim, seq_len)')
        lines.append('        multi_head = True')
        lines.append('    elif Q.dim() == 2:')
        lines.append('        # [dim, seq] format - add batch dimension')
        lines.append('        head_dim, seq_len = Q.shape')
        lines.append('        batch_size, num_heads = 1, 1')
        lines.append('        Q = Q.unsqueeze(0)')
        lines.append('        K = K.unsqueeze(0)')
        lines.append('        V = V.unsqueeze(0)')
        lines.append('        multi_head = False')
        lines.append('    else:')
        lines.append('        raise ValueError(f"Unsupported input shape: {Q.shape}")')
        lines.append('')
        lines.append('    if scale is None:')
        lines.append('        scale = 1.0 / math.sqrt(head_dim)')
        lines.append('')

        # Generate computation graph operations
        lines.append('    # ===== Computation Graph =====')
        lines.append('    # Note: Working with [batch, e, p] layout where:')
        lines.append('    #   - e (dim 1) = embedding/head dimension')
        lines.append('    #   - p (dim 2) = query sequence position')
        lines.append('    #   - m = key/value sequence position')
        lines.append('')

        for node_id in self.graph.execution_order:
            node = self.graph.nodes[node_id]

            # Skip primitives and CreateTensor
            if node.is_primitive() or node.op == 'CreateTensor':
                continue

            # Generate code for this operation
            result = self.op_codegen.generate_op(node)

            # Add dimension info as comment
            dim_comment = f"  # output: {node.output_dims}" if node.output_dims else ""

            for i, line in enumerate(result.code_lines):
                if i == len(result.code_lines) - 1:
                    lines.append(f'    {line}{dim_comment}')
                else:
                    lines.append(f'    {line}')

        # Output handling
        if self.graph.outputs:
            out_var = self.graph.outputs[0].var_name
            lines.append('')
            lines.append('    # ===== Output Reshape =====')
            lines.append(f'    output = {out_var}')
            lines.append('    if multi_head:')
            lines.append('        # Reshape back to [batch, heads, seq, dim]')
            lines.append('        output = output.reshape(batch_size, num_heads, head_dim, seq_len)')
            lines.append('        output = output.permute(0, 1, 3, 2)')
            lines.append('    else:')
            lines.append('        output = output.squeeze(0)')
            lines.append('')
            lines.append('    return output')

        return '\n'.join(lines)

    def _generate_test(self) -> str:
        """Generate test code."""
        return '''

# ===== Test =====
if __name__ == "__main__":
    import torch

    print("Testing graph-driven PyTorch attention...")
    print("=" * 60)

    torch.manual_seed(42)

    # Test 1: Simple 2D input
    print("\\nTest 1: 2D input [dim, seq]")
    head_dim = 64
    seq_len = 128

    Q = torch.randn(head_dim, seq_len, device='cuda', dtype=torch.float32)
    K = torch.randn(head_dim, seq_len, device='cuda', dtype=torch.float32)
    V = torch.randn(head_dim, seq_len, device='cuda', dtype=torch.float32)

    out = egg_attention_graphdriven(Q, K, V)
    print(f"  Input shape: [{head_dim}, {seq_len}]")
    print(f"  Output shape: {list(out.shape)}")

    # Reference computation
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q.T, K) * scale  # [seq, seq]
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V.T).T  # [dim, seq]

    diff = (out - ref).abs().max().item()
    print(f"  Max diff vs reference: {diff:.6f}")
    print(f"  {'PASSED' if diff < 0.01 else 'FAILED'}!")

    # Test 2: 4D input (batch, heads)
    print("\\nTest 2: 4D input [batch, heads, seq, dim]")
    batch_size = 2
    num_heads = 4
    seq_len = 256
    head_dim = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float32)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    out = egg_attention_graphdriven(Q, K, V)
    print(f"  Input shape: {list(Q.shape)}")
    print(f"  Output shape: {list(out.shape)}")

    # Reference computation
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V)

    diff = (out - ref).abs().max().item()
    print(f"  Max diff vs reference: {diff:.6f}")
    print(f"  {'PASSED' if diff < 0.01 else 'FAILED'}!")

    print("\\n" + "=" * 60)
    print("All tests completed!")
'''


# ============================================================================
# Entry Point
# ============================================================================

def generate_pytorch_code(graph: ComputationGraph) -> str:
    """
    Generate PyTorch code from computation graph.

    Args:
        graph: Parsed computation graph from EggParser

    Returns:
        Generated Python code as string
    """
    generator = PyTorchCodeGenerator(graph)
    return generator.generate()


if __name__ == "__main__":
    import sys
    from egg_parser import EggParser

    if len(sys.argv) < 2:
        print("Usage: python pytorch_codegen.py <json_file> [-o output.py]")
        print("Example: python pytorch_codegen.py attention.json -o generated.py")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = 'generated_pytorch.py'

    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    # Parse and generate
    parser = EggParser(json_file)
    graph = parser.parse()

    print(f"Parsed graph with {len(graph.nodes)} nodes")
    print(f"Operations: {sorted(graph.unique_ops)}")

    code = generate_pytorch_code(graph)

    with open(output_file, 'w') as f:
        f.write(code)

    print(f"Generated: {output_file}")
