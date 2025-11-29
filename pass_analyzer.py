"""
Pass Analyzer for Graph-Driven Triton Code Generation

Automatically infers the pass structure from computation graph dependencies.
The number of passes and which operations belong to each pass emerges from
analyzing global reductions over the 'm' dimension (key sequence).

Key Insight:
A new pass is needed when an operation requires the COMPLETE result of a
global reduction over 'm'. Operations that can be computed per-block stay
in the same pass. Global reductions create "synchronization points" that
define pass boundaries.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

from egg_parser import ComputationGraph, IRNode


# Global reductions that create pass boundaries
# These need to see ALL K/V blocks before their result is valid
GLOBAL_REDUCE_OPS = {
    # Standard (non-tiled) reductions over 'm'
    'R_max_m',   # max over all key positions
    'R_add_m',   # sum over all key positions

    # Tiled reductions - 'm1' is the tile dimension
    'R_max_m1',  # max across tiles (needs all tiles)
    'R_add_m1',  # sum across tiles (needs all tiles)
}

# Local reductions that DON'T create pass boundaries
# These can be computed per-block
LOCAL_REDUCE_OPS = {
    'R_add_e',   # sum over embedding dimension (per-block)
    'R_max_m0',  # max within a tile (per-block)
    'R_add_m0',  # sum within a tile (per-block)
    'R_add_f',   # final output reduction
}


@dataclass
class PassInfo:
    """Information about a single pass."""
    pass_num: int
    operations: List[str]  # Node IDs in this pass
    global_reductions: List[str]  # Global reductions that complete in this pass
    needs_k_load: bool = True  # Whether this pass loads K blocks
    needs_v_load: bool = False  # Whether this pass loads V blocks
    is_post_loop: bool = False  # True if this "pass" is just post-loop computation


@dataclass
class MemoryPassInfo:
    """Information about memory access patterns (K/V loops)."""
    num_memory_passes: int  # How many K/V loops are needed
    sync_levels: int  # How many synchronization levels (dependency depth)
    post_loop_ops: List[str]  # Operations that run after all K/V loops


class PassAnalyzer:
    """
    Automatically infers pass structure from computation graph.

    Algorithm:
    1. Identify global reductions (operations over 'm' that need all blocks)
    2. Compute dependency level for each node:
       - Level 0: inputs and ops before any global reduction
       - Level N: ops that depend on N global reductions
    3. Group operations by level -> these become passes

    The number of passes = max dependency level + 1
    """

    def __init__(self, graph: ComputationGraph):
        self.graph = graph
        self.pass_levels: Dict[str, int] = {}  # node_id -> pass level
        self._global_reductions: Set[str] = set()  # node IDs of global reductions

    def analyze(self) -> Dict[int, List[str]]:
        """
        Analyze graph and return pass structure.

        Returns:
            Dict mapping pass_number (0-indexed) to list of node IDs in that pass.
            Operations are in topological order within each pass.
        """
        # Step 1: Find all global reductions
        self._global_reductions = self._find_global_reductions()

        # Step 2: Compute pass levels for all nodes
        for node_id in self.graph.execution_order:
            self._compute_level(node_id)

        # Step 3: Group by pass level
        return self._group_by_pass()

    def get_pass_info(self) -> List[PassInfo]:
        """
        Get detailed information about each pass.

        Returns:
            List of PassInfo objects, one per pass.
        """
        passes = self.analyze()
        result = []

        for pass_num in sorted(passes.keys()):
            ops = passes[pass_num]

            # Find global reductions in this pass
            global_reds = [
                nid for nid in ops
                if nid in self._global_reductions
            ]

            # Check if V is needed (for output accumulation)
            needs_v = any(
                self.graph.nodes[nid].op in ('M_mul_fmp', 'M_mul_fm1m0p')
                for nid in ops
                if nid in self.graph.nodes
            )

            result.append(PassInfo(
                pass_num=pass_num,
                operations=ops,
                global_reductions=global_reds,
                needs_k_load=True,  # All passes load K for QK computation
                needs_v_load=needs_v,
            ))

        return result

    def get_num_passes(self) -> int:
        """Get the total number of passes."""
        if not self.pass_levels:
            self.analyze()
        return max(self.pass_levels.values(), default=0) + 1

    def _find_global_reductions(self) -> Set[str]:
        """Find all global reduction operations in the graph."""
        return {
            node_id
            for node_id, node in self.graph.nodes.items()
            if node.op in GLOBAL_REDUCE_OPS
        }

    def _compute_level(self, node_id: str) -> int:
        """
        Compute the pass level for a node.

        Key insight:
        - A global reduction ENDS a pass (accumulates across all K/V blocks)
        - Operations that USE a global reduction's result must be in a LATER pass

        Rules:
        - Global reduction: level = max(children_levels)  [ends this pass]
        - Op depending on global reduction: level = that_reduction_level + 1
        - Op depending on non-global: level = max(children_levels)
        """
        # Already computed
        if node_id in self.pass_levels:
            return self.pass_levels[node_id]

        # Node doesn't exist (might be primitive)
        if node_id not in self.graph.nodes:
            return 0

        node = self.graph.nodes[node_id]

        # Base case: inputs have level 0
        if not node.children or node.op == 'CreateTensor':
            self.pass_levels[node_id] = 0
            return 0

        # Compute level based on children
        # If we depend on a global reduction, we need to be in the NEXT pass
        max_level = 0
        for child_id in node.children:
            if child_id in self.graph.nodes:
                child_level = self._compute_level(child_id)

                # If child is a global reduction, we must be in next pass
                if child_id in self._global_reductions:
                    max_level = max(max_level, child_level + 1)
                else:
                    max_level = max(max_level, child_level)

        self.pass_levels[node_id] = max_level
        return max_level

    def _group_by_pass(self) -> Dict[int, List[str]]:
        """
        Group operations by pass level.

        Maintains topological order within each pass.
        """
        passes: Dict[int, List[str]] = defaultdict(list)

        # Use execution_order to maintain topological ordering
        for node_id in self.graph.execution_order:
            if node_id in self.pass_levels:
                level = self.pass_levels[node_id]
                passes[level].append(node_id)

        return dict(passes)

    def is_global_reduction(self, node_id: str) -> bool:
        """Check if a node is a global reduction."""
        if node_id not in self.graph.nodes:
            return False
        return self.graph.nodes[node_id].op in GLOBAL_REDUCE_OPS

    def is_local_reduction(self, node_id: str) -> bool:
        """Check if a node is a local (per-block) reduction."""
        if node_id not in self.graph.nodes:
            return False
        return self.graph.nodes[node_id].op in LOCAL_REDUCE_OPS

    def is_post_loop_op(self, node_id: str) -> bool:
        """
        Check if an operation is a "post-loop" operation.

        Post-loop ops are operations that:
        1. Depend on global reductions (so they're at level > 0)
        2. But ALL their inputs are either global reductions or other post-loop ops
        3. They don't need to access K/V data (no tiled dimensions)

        These can run after all K/V loops complete, without reloading K/V.
        """
        if node_id not in self.graph.nodes:
            return False

        node = self.graph.nodes[node_id]
        level = self.pass_levels.get(node_id, 0)

        # Level 0 ops are in the first K/V loop
        if level == 0:
            return False

        # Check if ALL inputs are either:
        # - Global reductions, or
        # - Other post-loop ops (recursively)
        for child_id in node.children:
            if child_id not in self.graph.nodes:
                continue

            # If child is a global reduction, that's OK
            if child_id in self._global_reductions:
                continue

            # If child is at same level and also post-loop, that's OK
            child_level = self.pass_levels.get(child_id, 0)
            if child_level == level and self.is_post_loop_op(child_id):
                continue

            # Otherwise, this op needs in-loop data
            return False

        return True

    def get_memory_pass_info(self) -> MemoryPassInfo:
        """
        Analyze memory access patterns.

        Memory passes = how many times we iterate over ALL K/V blocks.
        This is different from synchronization levels (dependency depth).

        Post-loop operations don't require K/V access, so they don't
        count as a separate memory pass.
        """
        if not self.pass_levels:
            self.analyze()

        # Find all post-loop operations
        post_loop_ops = [
            node_id for node_id in self.graph.execution_order
            if node_id in self.pass_levels and self.is_post_loop_op(node_id)
        ]

        # Sync levels = number of synchronization points
        sync_levels = max(self.pass_levels.values(), default=0) + 1

        # Memory passes = sync levels minus post-loop "passes"
        # Post-loop ops are all at the highest level(s) but don't need K/V
        if post_loop_ops:
            # Find the minimum level of post-loop ops
            post_loop_levels = {
                self.pass_levels[nid] for nid in post_loop_ops
            }
            # Memory passes = levels before post-loop begins
            num_memory_passes = min(post_loop_levels)
        else:
            num_memory_passes = sync_levels

        return MemoryPassInfo(
            num_memory_passes=num_memory_passes,
            sync_levels=sync_levels,
            post_loop_ops=post_loop_ops,
        )

    def print_analysis(self):
        """Print a detailed analysis of the pass structure."""
        passes = self.analyze()
        num_passes = len(passes)
        mem_info = self.get_memory_pass_info()

        print(f"Pass Analysis: {num_passes} sync levels, {mem_info.num_memory_passes} memory passes")
        print("=" * 60)

        for pass_num in sorted(passes.keys()):
            ops = passes[pass_num]

            # Check if this is a post-loop level
            is_post_loop = all(
                self.is_post_loop_op(nid)
                for nid in ops
                if nid in self.graph.nodes
            )

            pass_type = "POST-LOOP" if is_post_loop else f"MEMORY PASS {pass_num + 1}"
            print(f"\n{pass_type} (sync level {pass_num}):")
            print("-" * 40)

            for node_id in ops:
                if node_id not in self.graph.nodes:
                    continue
                node = self.graph.nodes[node_id]

                # Mark global reductions and post-loop ops
                markers = []
                if node_id in self._global_reductions:
                    markers.append("GLOBAL_RED")
                if self.is_post_loop_op(node_id):
                    markers.append("POST_LOOP")

                marker_str = f" <- {', '.join(markers)}" if markers else ""
                print(f"  {node.op:<20}{marker_str}")


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_passes(graph: ComputationGraph) -> Dict[int, List[str]]:
    """Convenience function to analyze passes from a graph."""
    analyzer = PassAnalyzer(graph)
    return analyzer.analyze()


def get_num_passes(graph: ComputationGraph) -> int:
    """Get the number of passes needed for a graph."""
    analyzer = PassAnalyzer(graph)
    return analyzer.get_num_passes()


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    from egg_parser import EggParser
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pass_analyzer.py <json_file>")
        print("Example: python pass_analyzer.py attention.json")
        sys.exit(1)

    json_file = sys.argv[1]
    print(f"Analyzing: {json_file}")
    print()

    # Parse graph
    parser = EggParser(json_file)
    graph = parser.parse()

    # Analyze passes
    analyzer = PassAnalyzer(graph)
    analyzer.print_analysis()

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total operations: {len(graph.nodes)}")
    print(f"Unique operations: {len(graph.unique_ops)}")
    print(f"Synchronization levels: {analyzer.get_num_passes()}")

    # Show memory pass info
    mem_info = analyzer.get_memory_pass_info()
    print()
    print("Memory Access Analysis:")
    print(f"  Memory passes (K/V loops): {mem_info.num_memory_passes}")
    print(f"  Synchronization levels: {mem_info.sync_levels}")
    print(f"  Post-loop operations: {len(mem_info.post_loop_ops)}")
    if mem_info.post_loop_ops:
        print(f"    Post-loop ops: {[graph.nodes[nid].op for nid in mem_info.post_loop_ops if nid in graph.nodes]}")

    # Show pass info
    print()
    print("Pass Details:")
    for info in analyzer.get_pass_info():
        print(f"  Sync level {info.pass_num}: {len(info.operations)} ops, "
              f"K={info.needs_k_load}, V={info.needs_v_load}, "
              f"global_reds={len(info.global_reductions)}")
