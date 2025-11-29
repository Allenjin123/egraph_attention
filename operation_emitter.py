"""
Operation Emitter for Triton Code Generation

Maps graph operations to Triton primitives and fills scaffold holes
based on graph analysis.

Key Differences from PyTorch:
- Triton works with fixed-size blocks (BLOCK_M, BLOCK_N, BLOCK_K)
- Broadcasting is implicit via block shapes
- Uses tl.* primitives (tl.exp, tl.sum, tl.max, tl.dot)
- Memory access patterns are explicit
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from egg_parser import ComputationGraph, IRNode
from egg_ops import OP_REGISTRY, DimSpec, get_op_spec
from algorithm_detector import AlgorithmDetector, AlgorithmInfo, AlgorithmType
from kernel_scaffolds import KernelScaffold, ThreePassScaffold, TwoPassScaffold


@dataclass
class TritonVar:
    """Represents a Triton variable in generated code."""
    name: str
    shape: str  # e.g., "[BLOCK_M, BLOCK_N]" or "[BLOCK_M]"
    dtype: str = "tl.float32"


class OperationEmitter:
    """
    Emits Triton code for graph operations.

    Maps egglog operations to Triton primitives, handling:
    - Block-based computation
    - Reduction operations
    - Broadcasting (via block shapes)
    """

    # Map egglog ops to Triton ops
    TRITON_OPS = {
        # Unary
        'tl.exp': 'tl.exp',

        # Binary
        '*': lambda a, b: f"{a} * {b}",
        '+': lambda a, b: f"{a} + {b}",
        '-': lambda a, b: f"{a} - {b}",
        '/': lambda a, b: f"{a} / {b}",

        # Reduce
        'tl.sum': lambda x, axis: f"tl.sum({x}, axis={axis})",
        'tl.max': lambda x, axis: f"tl.max({x}, axis={axis})",
    }

    # Map dimension to Triton block dimension
    DIM_TO_BLOCK = {
        'e': 'BLOCK_K',    # Embedding/head dimension
        'f': 'BLOCK_K',    # Output embedding (same as e)
        'm': 'BLOCK_N',    # Key/value sequence
        'p': 'BLOCK_M',    # Query sequence (computed in parallel)
        'm0': 'BLOCK_N',   # Tile size
        'm1': 'NUM_TILES', # Number of tiles
    }

    def __init__(self, graph: ComputationGraph, algorithm_info: AlgorithmInfo):
        self.graph = graph
        self.info = algorithm_info

        # Track variable mappings from graph nodes to Triton vars
        self.var_map: Dict[str, TritonVar] = {}

        # Initialize input mappings
        self._init_inputs()

    def _init_inputs(self):
        """Initialize Triton variable mappings for inputs."""
        # Q, K, V are loaded as blocks
        self.var_map['Q'] = TritonVar('q', '[BLOCK_M, BLOCK_K]')
        self.var_map['K'] = TritonVar('k', '[BLOCK_N, BLOCK_K]')
        self.var_map['V'] = TritonVar('v', '[BLOCK_N, BLOCK_K]')

    def emit_operation(self, node: IRNode) -> List[str]:
        """
        Emit Triton code for a single operation.

        Args:
            node: IRNode from the computation graph

        Returns:
            List of Triton code lines
        """
        spec = get_op_spec(node.op)
        if spec is None:
            return [f"# Unknown op: {node.op}"]

        if spec.op_type == 'map':
            return self._emit_map_op(node, spec)
        elif spec.op_type == 'reduce':
            return self._emit_reduce_op(node, spec)
        else:
            return [f"# Unsupported op type: {spec.op_type}"]

    def _emit_map_op(self, node: IRNode, spec) -> List[str]:
        """Emit code for map (element-wise) operations."""
        lines = []
        children = node.children
        result_var = self._get_triton_var_name(node)

        if len(children) == 2:
            # Binary op
            var_a = self._get_child_var(children[0])
            var_b = self._get_child_var(children[1])

            # Get output shape
            output_shape = self._dims_to_shape(node.output_dims)

            op = spec.triton_op
            if op == '*':
                lines.append(f"{result_var} = {var_a} * {var_b}")
            elif op == '+':
                lines.append(f"{result_var} = {var_a} + {var_b}")
            elif op == '-':
                lines.append(f"{result_var} = {var_a} - {var_b}")
            elif op == '/':
                lines.append(f"{result_var} = {var_a} / {var_b}")
            else:
                lines.append(f"{result_var} = {op}({var_a}, {var_b})")

        elif len(children) == 1:
            # Unary op
            var_a = self._get_child_var(children[0])

            op = spec.triton_op
            if op == 'tl.exp':
                lines.append(f"{result_var} = tl.exp({var_a})")
            else:
                lines.append(f"{result_var} = {op}({var_a})")

        # Register the output variable
        if node.output_dims:
            self.var_map[node.id] = TritonVar(
                result_var,
                self._dims_to_shape(node.output_dims)
            )

        return lines

    def _emit_reduce_op(self, node: IRNode, spec) -> List[str]:
        """Emit code for reduce operations."""
        lines = []
        var_a = self._get_child_var(node.children[0])
        result_var = self._get_triton_var_name(node)

        # Get input dimensions and reduce dimension
        child_node = self.graph.nodes.get(node.children[0])
        if child_node and child_node.output_dims:
            input_dims = child_node.output_dims
            reduce_dim = spec.reduce_dim

            # Find axis in block terms
            # In Triton blocks, we typically have [BLOCK_M, BLOCK_N] or similar
            axis = self._get_reduce_axis(input_dims, reduce_dim)

            op = spec.triton_op
            if op == 'tl.sum':
                lines.append(f"{result_var} = tl.sum({var_a}, axis={axis})")
            elif op == 'tl.max':
                lines.append(f"{result_var} = tl.max({var_a}, axis={axis})")
            else:
                lines.append(f"# Unknown reduce: {op}")
                lines.append(f"{result_var} = tl.sum({var_a}, axis={axis})")
        else:
            lines.append(f"# Could not determine reduce axis")
            lines.append(f"{result_var} = {var_a}")

        # Register output
        if node.output_dims:
            self.var_map[node.id] = TritonVar(
                result_var,
                self._dims_to_shape(node.output_dims)
            )

        return lines

    def _get_triton_var_name(self, node: IRNode) -> str:
        """Get Triton variable name for a node."""
        # Use a simplified name based on the operation
        op = node.op.lower().replace('_', '')
        return f"{op}_{node.id.split('-')[-1]}"

    def _get_child_var(self, child_id: str) -> str:
        """Get the Triton variable name for a child node."""
        if child_id in self.var_map:
            return self.var_map[child_id].name

        # Check if it's an input tensor
        if child_id in self.graph.nodes:
            child = self.graph.nodes[child_id]
            if child.op == 'CreateTensor':
                tensor_name = child.tensor_name
                if tensor_name in self.var_map:
                    return self.var_map[tensor_name].name

        return f"var_{child_id.split('-')[-1]}"

    def _dims_to_shape(self, dims: Optional[DimSpec]) -> str:
        """Convert DimSpec to Triton block shape string."""
        if dims is None:
            return "[BLOCK_M]"

        shape_parts = []
        for dim in dims.dims:
            if dim in self.DIM_TO_BLOCK:
                shape_parts.append(self.DIM_TO_BLOCK[dim])
            else:
                shape_parts.append('BLOCK_M')  # Default

        return f"[{', '.join(shape_parts)}]"

    def _get_reduce_axis(self, input_dims: DimSpec, reduce_dim: str) -> int:
        """Get the axis index for reduction in Triton block."""
        try:
            return input_dims.dims.index(reduce_dim)
        except ValueError:
            return 0  # Default to first axis


class HoleEmitter:
    """
    Fills scaffold holes with graph-derived Triton code.

    Analyzes the graph to determine which operations belong in which holes,
    then uses OperationEmitter to generate the code.
    """

    def __init__(self, graph: ComputationGraph, scaffold: KernelScaffold):
        self.graph = graph
        self.scaffold = scaffold
        self.detector = AlgorithmDetector(graph)
        self.info = self.detector.detect()
        self.op_emitter = OperationEmitter(graph, self.info)

    def fill_all_holes(self):
        """Analyze graph and fill all scaffold holes."""
        if self.info.algorithm == AlgorithmType.THREE_PASS:
            self._fill_3pass_holes()
        elif self.info.algorithm == AlgorithmType.TWO_PASS:
            self._fill_2pass_holes()

    def _fill_3pass_holes(self):
        """Fill holes for 3-pass scaffold."""
        passes = self.detector.get_pass_structure()

        # The standard 3-pass scaffold already has the core operations
        # The holes are for customization/extension

        # For now, we leave most holes empty (using scaffold defaults)
        # This can be extended to insert custom operations from the graph

        self.scaffold.fill_hole("init", [
            "# Graph-derived initialization",
        ])

        self.scaffold.fill_hole("pass1_qk", [
            "# QK computation handled by scaffold",
        ])

    def _fill_2pass_holes(self):
        """Fill holes for 2-pass scaffold."""
        passes = self.detector.get_pass_structure()

        # Similar to 3-pass, the scaffold has the core structure
        self.scaffold.fill_hole("pass1_local_max", [
            "# Local max computed by scaffold",
        ])

        self.scaffold.fill_hole("pass2_correction", [
            "# Correction factor: exp(local_max - global_max)",
        ])

    def generate_kernel(self) -> str:
        """Generate the complete kernel with filled holes."""
        self.fill_all_holes()
        return self.scaffold.generate()


# ============================================================================
# Wrapper Generator
# ============================================================================

def generate_wrapper(algorithm_type: str) -> str:
    """Generate Python wrapper function for the kernel."""
    if algorithm_type == '3pass':
        return '''
def egg_attention_hybrid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Hybrid attention implementation (scaffold + graph-derived ops).

    Args:
        q: Query tensor [batch, heads, seq_q, dim]
        k: Key tensor [batch, heads, seq_k, dim]
        v: Value tensor [batch, heads, seq_k, dim]
        scale: Scaling factor (default: 1/sqrt(dim))

    Returns:
        Output tensor [batch, heads, seq_q, dim]
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(head_dim)

    # Grid
    grid = (batch_size, num_heads, triton.cdiv(seq_len_q, BLOCK_M))

    _attention_3pass_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out
'''
    elif algorithm_type == '2pass':
        return '''
def egg_attention_hybrid_2pass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Hybrid 2-pass attention implementation (scaffold + graph-derived ops).

    Args:
        q: Query tensor [batch, heads, seq_q, dim]
        k: Key tensor [batch, heads, seq_k, dim]
        v: Value tensor [batch, heads, seq_k, dim]
        scale: Scaling factor (default: 1/sqrt(dim))

    Returns:
        Output tensor [batch, heads, seq_q, dim]
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(head_dim)
    num_tiles = triton.cdiv(seq_len_k, BLOCK_N)

    # Grid
    grid = (batch_size, num_heads, triton.cdiv(seq_len_q, BLOCK_M))

    _attention_2pass_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        scale,
        NUM_TILES=num_tiles,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out
'''
    else:
        return "# Unknown algorithm type"


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    from egg_parser import EggParser
    import sys

    if len(sys.argv) < 2:
        print("Usage: python operation_emitter.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    parser = EggParser(json_file)
    graph = parser.parse()

    # Detect algorithm
    detector = AlgorithmDetector(graph)
    info = detector.detect()

    print(f"Algorithm: {info.algorithm.value}")
    print()

    # Create emitter
    emitter = OperationEmitter(graph, info)

    # Emit code for each operation
    print("Generated Triton operations:")
    print("-" * 40)
    for node_id in graph.execution_order:
        node = graph.nodes.get(node_id)
        if not node or node.is_primitive() or node.op == 'CreateTensor':
            continue

        lines = emitter.emit_operation(node)
        print(f"# {node.op}")
        for line in lines:
            print(f"  {line}")
        print()
