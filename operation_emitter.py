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
from pass_analyzer import PassAnalyzer


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
        elif spec.op_type == 'tile':
            return self._emit_tile_op(node, spec)
        elif spec.op_type == 'create':
            return []  # Skip CreateTensor nodes
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

    def _emit_tile_op(self, node: IRNode, spec) -> List[str]:
        """
        Emit code for tile operations.

        In Triton, tiling is implicit through the loop structure.
        We iterate over K/V blocks (tiles), so split/unsplit are no-ops.
        However, we track them for variable dependencies.
        """
        lines = []
        result_var = self._get_triton_var_name(node)

        if node.op == 'T_split_m_m1m0':
            # Splitting K into tiles - handled by loop iteration
            # Just reference the child variable
            if node.children:
                child_var = self._get_child_var(node.children[0])
                lines.append(f"# Tiling: K blocks loaded per iteration")
                # Register the variable as reference to the input
                self.var_map[node.id] = TritonVar(child_var, self._dims_to_shape(node.output_dims))
        elif node.op == 'T_unsplit_m1m0_m':
            # Merging tiles back - this is the accumulated result
            if node.children:
                child_var = self._get_child_var(node.children[0])
                lines.append(f"# Untiling: accumulated across all tiles")
                # The unsplit result should reference the accumulator
                self.var_map[node.id] = TritonVar(child_var, self._dims_to_shape(node.output_dims))
        else:
            lines.append(f"# Tile operation: {node.op}")

        return lines


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
        """
        Analyze graph and fill scaffold holes with actual operations.

        Now uses PassAnalyzer to automatically determine pass structure
        and generates Triton code directly from the computation graph.
        """
        # Use PassAnalyzer to get automatic pass structure
        analyzer = PassAnalyzer(self.graph)
        passes = analyzer.analyze()
        pass_info_list = analyzer.get_pass_info()

        # Generate operations for each pass
        for pass_info in pass_info_list:
            pass_num = pass_info.pass_num
            operations = pass_info.operations

            # Collect code lines for this pass
            pass_code = []
            pass_code.append(f"# Pass {pass_num}: {len(operations)} operations")

            # Generate code for each operation
            for node_id in operations:
                if node_id in self.graph.nodes:
                    node = self.graph.nodes[node_id]

                    # Skip primitives and inputs
                    if node.is_primitive() or node.op == 'CreateTensor':
                        continue

                    # Emit operation
                    op_code = self.op_emitter.emit_operation(node)
                    pass_code.extend(op_code)

                    # Mark global reductions
                    if analyzer.is_global_reduction(node_id):
                        pass_code.append(f"# ^ Global reduction - ends pass {pass_num}")

            # Fill appropriate hole based on pass number
            hole_name = f"pass{pass_num + 1}_operations"  # +1 for 1-indexed
            self.scaffold.fill_hole(hole_name, pass_code)

    def generate_kernel(self) -> str:
        """Generate the complete kernel with filled holes."""
        self.fill_all_holes()
        return self.scaffold.generate()

    def generate_kernel_from_graph(self) -> str:
        """
        Generate complete Triton kernel directly from computation graph.

        No fixed templates - the kernel structure emerges from the graph.
        """
        analyzer = PassAnalyzer(self.graph)
        passes = analyzer.analyze()
        mem_info = analyzer.get_memory_pass_info()

        lines = []
        lines.append("@triton.jit")
        lines.append("def attention_kernel(")
        lines.append("    Q, K, V, Out,")
        lines.append("    stride_qb, stride_qh, stride_qm, stride_qk,")
        lines.append("    stride_kb, stride_kh, stride_kn, stride_kk,")
        lines.append("    stride_vb, stride_vh, stride_vn, stride_vk,")
        lines.append("    stride_ob, stride_oh, stride_om, stride_ok,")
        lines.append("    N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,")
        lines.append("):")
        lines.append("    # Graph-driven kernel generation")
        lines.append(f"    # Memory passes: {mem_info.num_memory_passes}")
        lines.append(f"    # Sync levels: {mem_info.sync_levels}")
        lines.append("")
        lines.append("    # Get block indices")
        lines.append("    pid_m = tl.program_id(0)")
        lines.append("    pid_b = tl.program_id(1)")
        lines.append("    pid_h = tl.program_id(2)")
        lines.append("")
        lines.append("    # Compute offsets for Q (this block of queries)")
        lines.append("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        lines.append("    offs_k = tl.arange(0, BLOCK_K)")
        lines.append("    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh")
        lines.append("    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)")
        lines.append("")

        # Initialize accumulators based on what global reductions we have
        lines.append("    # Initialize accumulators for global reductions")
        for pass_num in sorted(passes.keys()):
            for node_id in passes[pass_num]:
                if analyzer.is_global_reduction(node_id):
                    node = self.graph.nodes[node_id]
                    if 'max' in node.op.lower():
                        lines.append("    global_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)")
                    elif 'add' in node.op.lower() or 'sum' in node.op.lower():
                        lines.append("    global_sum = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)")
                        lines.append("    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)")
        lines.append("")

        # Generate K/V loops for each memory pass
        for pass_idx in range(mem_info.num_memory_passes):
            lines.append(f"    # Memory Pass {pass_idx}: Iterate over all K/V blocks")
            lines.append("    NUM_TILES = tl.cdiv(N, BLOCK_N)")
            lines.append("    for tile_idx in range(NUM_TILES):")
            lines.append("        # Load K block")
            lines.append("        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)")
            lines.append("        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh")
            lines.append("        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)")

            # Check if this pass needs V loading
            pass_info_list = analyzer.get_pass_info()
            needs_v = any(info.needs_v_load for info in pass_info_list if info.pass_num == pass_idx)
            if needs_v:
                lines.append("")
                lines.append("        # Load V block")
                lines.append("        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh")
                lines.append("        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)")
            lines.append("")

            # Generate operations for this pass
            if pass_idx in passes:
                lines.append(f"        # Operations for sync level {pass_idx}")

                # Detect and optimize QK computation pattern
                qk_computed = self._try_emit_qk_pattern(passes[pass_idx], lines, analyzer)

                for node_id in passes[pass_idx]:
                    if node_id in self.graph.nodes:
                        node = self.graph.nodes[node_id]
                        if node.is_primitive() or node.op == 'CreateTensor':
                            continue

                        # Skip if already handled by pattern detection
                        if qk_computed and node.op in ('M_mul_em1m0p', 'M_mul_emp', 'R_add_e'):
                            continue

                        op_code = self.op_emitter.emit_operation(node)
                        for code_line in op_code:
                            lines.append(f"        {code_line}")

                        if analyzer.is_global_reduction(node_id):
                            lines.append(f"        # ^ Global reduction")
            lines.append("")

        # Post-loop operations
        if mem_info.post_loop_ops:
            lines.append("    # Post-loop operations (don't need K/V access)")
            for node_id in mem_info.post_loop_ops:
                if node_id in self.graph.nodes:
                    node = self.graph.nodes[node_id]
                    op_code = self.op_emitter.emit_operation(node)
                    for code_line in op_code:
                        lines.append(f"    {code_line}")
            lines.append("")

        # Store output
        lines.append("    # Store output")
        lines.append("    Out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh")
        lines.append("    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, acc)")

        return "\n".join(lines)

    def _try_emit_qk_pattern(self, node_ids: List[str], lines: List[str], analyzer: PassAnalyzer) -> bool:
        """
        Try to detect and emit optimized QK computation pattern.

        Pattern: M_mul_emp/M_mul_em1m0p (Q*K) followed by R_add_e (sum over embedding)
        Optimized to: qk = tl.dot(q, tl.trans(k))

        Returns True if pattern was detected and emitted.
        """
        # Look for M_mul followed by R_add_e pattern
        mul_node = None
        add_node = None

        for node_id in node_ids:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                if node.op in ('M_mul_emp', 'M_mul_em1m0p'):
                    mul_node = node
                elif node.op == 'R_add_e' and mul_node:
                    # Check if R_add_e depends on the multiply
                    if mul_node.id in node.children:
                        add_node = node

        if mul_node and add_node:
            # Emit optimized QK computation
            lines.append("        # QK = Q @ K^T (optimized)")
            lines.append("        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]")

            # Register the variable
            self.op_emitter.var_map[add_node.id] = TritonVar('qk', '[BLOCK_M, BLOCK_N]')
            return True

        return False


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
