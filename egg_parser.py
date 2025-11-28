"""
Egglog JSON Parser for Attention Algorithms

Parses the JSON output from `egglog --to-json` into an intermediate representation
that can be used for Triton code generation.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from egg_ops import OP_REGISTRY, OpSpec, DimSpec, get_op_spec


@dataclass
class IRNode:
    """Intermediate representation node."""
    id: str
    op: str
    children: List[str]  # node IDs
    eclass: str
    cost: float = 1.0

    # Computed fields (filled during analysis)
    output_dims: Optional[DimSpec] = None
    var_name: str = ""  # Variable name in generated code
    tensor_name: str = ""  # For CreateTensor: "Q", "K", "V"

    # Metadata from JSON
    subsumed: bool = False

    def is_input(self) -> bool:
        return self.op == 'CreateTensor'

    def is_primitive(self) -> bool:
        return self.op.startswith('"') or self.op.isdigit()


@dataclass
class ComputationGraph:
    """Represents the parsed computation graph."""
    nodes: Dict[str, IRNode]
    inputs: Dict[str, IRNode]  # tensor_name -> node (Q, K, V)
    outputs: List[IRNode]      # Output nodes (result of extract)
    execution_order: List[str] # Topologically sorted node IDs
    unique_ops: Set[str]       # Unique operation types used

    def get_node(self, node_id: str) -> IRNode:
        return self.nodes[node_id]


class EggParser:
    """Parser for egglog JSON output."""

    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)
        self.nodes: Dict[str, IRNode] = {}
        self.inputs: Dict[str, IRNode] = {}
        self.outputs: List[IRNode] = []

    def parse(self) -> ComputationGraph:
        """Parse JSON into computation graph."""
        # Step 1: Build nodes
        self._build_nodes()

        # Step 2: Identify inputs (CreateTensor) and extract tensor names
        self._identify_inputs()

        # Step 3: Infer output dimensions for each node
        self._infer_dimensions()

        # Step 4: Assign variable names
        self._assign_var_names()

        # Step 5: Find output nodes (nodes with no consumers)
        self._find_outputs()

        # Step 6: Topological sort
        execution_order = self._topological_sort()

        # Step 7: Collect unique ops
        unique_ops = {n.op for n in self.nodes.values()
                      if not n.is_primitive() and n.op != 'CreateTensor'}

        return ComputationGraph(
            nodes=self.nodes,
            inputs=self.inputs,
            outputs=self.outputs,
            execution_order=execution_order,
            unique_ops=unique_ops,
        )

    def _build_nodes(self):
        """Build IRNode objects from JSON."""
        for node_id, node_data in self.data['nodes'].items():
            self.nodes[node_id] = IRNode(
                id=node_id,
                op=node_data['op'],
                children=node_data.get('children', []),
                eclass=node_data.get('eclass', ''),
                cost=node_data.get('cost', 1.0),
                subsumed=node_data.get('subsumed', False),
            )

    def _identify_inputs(self):
        """Identify input tensors (CreateTensor nodes)."""
        for node in self.nodes.values():
            if node.op == 'CreateTensor':
                # Extract tensor name from children
                # Children are: [dim1_node, dim2_node, name_node]
                name_node_id = node.children[2] if len(node.children) > 2 else None
                if name_node_id and name_node_id in self.nodes:
                    name_node = self.nodes[name_node_id]
                    # The op is like '"Q"', '"K"', '"V"'
                    tensor_name = name_node.op.strip('"')
                    node.tensor_name = tensor_name
                    self.inputs[tensor_name] = node

                    # Set dimensions based on tensor type
                    if tensor_name == 'Q':
                        node.output_dims = DimSpec(('e', 'p'))
                    elif tensor_name == 'K':
                        node.output_dims = DimSpec(('e', 'm'))
                    elif tensor_name == 'V':
                        node.output_dims = DimSpec(('f', 'm'))

    def _infer_dimensions(self):
        """Infer output dimensions for each node bottom-up."""
        # Process nodes in dependency order
        visited = set()

        def visit(node_id: str) -> Optional[DimSpec]:
            if node_id in visited:
                node = self.nodes.get(node_id)
                return node.output_dims if node else None

            node = self.nodes.get(node_id)
            if node is None:
                return None

            visited.add(node_id)

            # Primitives don't have dimensions
            if node.is_primitive():
                return None

            # CreateTensor already has dimensions set
            if node.op == 'CreateTensor':
                return node.output_dims

            # Visit children first
            child_dims = []
            for child_id in node.children:
                child_dim = visit(child_id)
                if child_dim:
                    child_dims.append(child_dim)

            # Infer dimensions from operation spec
            spec = get_op_spec(node.op)
            if spec:
                if spec.op_type == 'reduce' and child_dims:
                    # Remove reduce dimension from first input
                    node.output_dims = child_dims[0].without(spec.reduce_dim)
                else:
                    node.output_dims = spec.output_dims
            elif child_dims:
                # Fall back to first child's dimensions
                node.output_dims = child_dims[0]

            return node.output_dims

        for node_id in self.nodes:
            visit(node_id)

    def _assign_var_names(self):
        """Assign meaningful variable names to nodes."""
        # Counter for each operation type
        op_counters = defaultdict(int)

        for node_id, node in self.nodes.items():
            if node.is_primitive():
                continue

            if node.op == 'CreateTensor':
                node.var_name = node.tensor_name
            else:
                # Generate name like "M_mul_emp_0", "R_add_e_0"
                count = op_counters[node.op]
                op_counters[node.op] += 1
                # Shorten name for readability
                short_op = node.op.replace('_', '')
                node.var_name = f"{node.op}_{count}" if count > 0 else node.op

    def _find_outputs(self):
        """Find output nodes (nodes that are not consumed by other nodes)."""
        consumed = set()
        for node in self.nodes.values():
            for child_id in node.children:
                consumed.add(child_id)

        for node_id, node in self.nodes.items():
            if (node_id not in consumed and
                not node.is_primitive() and
                node.op != 'CreateTensor'):
                self.outputs.append(node)

        # If root_eclasses is provided, use that to identify outputs
        if self.data.get('root_eclasses'):
            root_eclasses = set(self.data['root_eclasses'])
            for node in self.nodes.values():
                if node.eclass in root_eclasses and node not in self.outputs:
                    self.outputs.append(node)

    def _topological_sort(self) -> List[str]:
        """Return nodes in topological order (dependencies first)."""
        visited = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            node = self.nodes.get(node_id)
            if node is None:
                return

            # Visit children first
            for child_id in node.children:
                visit(child_id)

            order.append(node_id)

        # Visit all nodes
        for node_id in self.nodes:
            visit(node_id)

        return order


def print_graph(graph: ComputationGraph):
    """Pretty print the computation graph."""
    print("=" * 70)
    print("COMPUTATION GRAPH")
    print("=" * 70)

    print("\nInputs:")
    for name, node in graph.inputs.items():
        print(f"  {name}: {node.output_dims}")

    print("\nExecution Order:")
    for i, node_id in enumerate(graph.execution_order):
        node = graph.nodes[node_id]
        if node.is_primitive():
            continue
        if node.op == 'CreateTensor':
            continue

        children = [graph.nodes[c].var_name if c in graph.nodes else c
                    for c in node.children]
        children_str = ', '.join(children)

        print(f"  {i:3d}. {node.var_name:20s} = {node.op}({children_str})")
        print(f"       dims: {node.output_dims}")

    print("\nOutputs:")
    for node in graph.outputs:
        print(f"  {node.var_name}: {node.output_dims}")

    print("\nUnique Operations:")
    for op in sorted(graph.unique_ops):
        print(f"  - {op}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python egg_parser.py <json_file>")
        print("Example: python egg_parser.py attention.json")
        sys.exit(1)

    json_file = sys.argv[1]
    parser = EggParser(json_file)
    graph = parser.parse()
    print_graph(graph)
