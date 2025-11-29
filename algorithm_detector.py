"""
Algorithm Detector for Attention Computation Graphs

Detects the algorithm type (3-pass, 2-pass, online) from the parsed
computation graph based on operation patterns.

Algorithm Patterns:
- 3-pass: Standard softmax with global max/sum
  - Operations: M_mul_emp, R_add_e, R_max_m, M_sub_mp, M_exp_mp, R_add_m, M_div_mp
  - No tiling operations

- 2-pass (FuseMax): Tiled attention with local max → global max → correction
  - Operations: T_split_m_m1m0, M_mul_em1m0p, R_max_m0, R_max_m1, M_exp_m1p (correction)
  - Has tiling operations and hierarchical max

- Online (FlashAttention): Single pass with running max/sum correction
  - Operations: Similar to 2-pass but with incremental correction in single loop
  - Future: Not yet implemented
"""

from dataclasses import dataclass
from typing import Set, List, Optional, Dict
from enum import Enum

from egg_parser import ComputationGraph, IRNode
from pass_analyzer import PassAnalyzer


class AlgorithmType(Enum):
    """Types of attention algorithms."""
    THREE_PASS = "3pass"      # Standard 3-pass softmax
    TWO_PASS = "2pass"        # FuseMax tiled attention
    ONLINE = "online"         # FlashAttention-style (future)
    UNKNOWN = "unknown"


@dataclass
class AlgorithmInfo:
    """Information about the detected algorithm."""
    algorithm: AlgorithmType
    num_passes: int
    uses_tiling: bool
    uses_correction: bool  # exp(local_max - global_max) correction factor
    operations: Set[str]

    # Key nodes for code generation
    qk_reduction_node: Optional[str] = None  # R_add_e after M_mul
    global_max_node: Optional[str] = None    # Final max reduction
    global_sum_node: Optional[str] = None    # Final sum reduction
    output_node: Optional[str] = None        # Final output


class AlgorithmDetector:
    """
    Detects algorithm type from computation graph.

    Uses pattern matching on operation sets and graph structure.
    """

    # Operation sets that characterize each algorithm
    TILED_OPS = {
        'T_split_m_m1m0', 'T_unsplit_m1m0_m',
        'M_mul_em1m0p', 'M_sub_m1m0p', 'M_exp_m1m0p', 'M_div_m1m0p',
        'M_sub_m1p', 'M_exp_m1p', 'M_mul_m1m0p', 'M_mul_m1p',
        'R_max_m0', 'R_max_m1', 'R_add_m0', 'R_add_m1'
    }

    THREE_PASS_OPS = {
        'M_mul_emp', 'R_add_e', 'R_max_m', 'M_sub_mp',
        'M_exp_mp', 'R_add_m', 'M_div_mp', 'M_mul_fmp'
    }

    # Correction factor pattern: M_exp following M_sub on hierarchical max
    CORRECTION_PATTERN = {'M_exp_m1p', 'M_sub_m1p'}

    def __init__(self, graph: ComputationGraph):
        self.graph = graph
        self.ops = graph.unique_ops

    def detect(self) -> AlgorithmInfo:
        """
        Detect the algorithm type from the computation graph.

        Returns:
            AlgorithmInfo with detected algorithm details
        """
        uses_tiling = bool(self.ops & self.TILED_OPS)
        uses_correction = bool(self.ops & self.CORRECTION_PATTERN)

        if uses_tiling:
            # 2-pass tiled attention
            return AlgorithmInfo(
                algorithm=AlgorithmType.TWO_PASS,
                num_passes=2,
                uses_tiling=True,
                uses_correction=uses_correction,
                operations=self.ops,
                qk_reduction_node=self._find_qk_reduction(),
                global_max_node=self._find_global_max(),
                global_sum_node=self._find_global_sum(),
                output_node=self._find_output(),
            )
        elif self.ops & self.THREE_PASS_OPS:
            # Standard 3-pass attention
            return AlgorithmInfo(
                algorithm=AlgorithmType.THREE_PASS,
                num_passes=3,
                uses_tiling=False,
                uses_correction=False,
                operations=self.ops,
                qk_reduction_node=self._find_qk_reduction(),
                global_max_node=self._find_node_by_op('R_max_m'),
                global_sum_node=self._find_node_by_op('R_add_m'),
                output_node=self._find_output(),
            )
        else:
            return AlgorithmInfo(
                algorithm=AlgorithmType.UNKNOWN,
                num_passes=0,
                uses_tiling=False,
                uses_correction=False,
                operations=self.ops,
            )

    def _find_qk_reduction(self) -> Optional[str]:
        """Find R_add_e that follows M_mul (QK dot product)."""
        for node_id, node in self.graph.nodes.items():
            if node.op == 'R_add_e':
                if node.children:
                    child_id = node.children[0]
                    if child_id in self.graph.nodes:
                        child = self.graph.nodes[child_id]
                        if child.op in ('M_mul_emp', 'M_mul_em1m0p'):
                            return node_id
        return None

    def _find_global_max(self) -> Optional[str]:
        """Find the global max reduction node."""
        # For 2-pass: R_max_m1 (max across tiles)
        # For 3-pass: R_max_m
        for node_id, node in self.graph.nodes.items():
            if node.op in ('R_max_m1', 'R_max_m'):
                return node_id
        return None

    def _find_global_sum(self) -> Optional[str]:
        """Find the global sum reduction node."""
        # For 2-pass: R_add_m1 (sum across tiles after correction)
        # For 3-pass: R_add_m (before final multiply)
        for node_id, node in self.graph.nodes.items():
            if node.op == 'R_add_m1':
                return node_id
        # For 3-pass, find R_add_m that's used for normalization
        for node_id, node in self.graph.nodes.items():
            if node.op == 'R_add_m':
                # Check if used by M_div (normalization)
                for other_id, other in self.graph.nodes.items():
                    if other.op.startswith('M_div') and node_id in other.children:
                        return node_id
        return None

    def _find_output(self) -> Optional[str]:
        """Find the output node."""
        if self.graph.outputs:
            return self.graph.outputs[0].id
        return None

    def _find_node_by_op(self, op_name: str) -> Optional[str]:
        """Find first node with given operation."""
        for node_id, node in self.graph.nodes.items():
            if node.op == op_name:
                return node_id
        return None

    def get_pass_structure(self) -> Dict[int, List[str]]:
        """
        Analyze graph to determine which operations belong to which pass.

        Now uses PassAnalyzer to automatically infer pass structure from
        computation graph dependencies instead of hardcoded patterns.

        Returns:
            Dict mapping pass number (0-indexed) to list of operation node IDs
        """
        analyzer = PassAnalyzer(self.graph)
        return analyzer.analyze()


def detect_algorithm(graph: ComputationGraph) -> AlgorithmInfo:
    """Convenience function to detect algorithm from graph."""
    detector = AlgorithmDetector(graph)
    return detector.detect()


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    from egg_parser import EggParser
    import sys

    if len(sys.argv) < 2:
        print("Usage: python algorithm_detector.py <json_file>")
        print("Example: python algorithm_detector.py attention.json")
        sys.exit(1)

    json_file = sys.argv[1]
    parser = EggParser(json_file)
    graph = parser.parse()

    detector = AlgorithmDetector(graph)
    info = detector.detect()

    print("=" * 60)
    print("ALGORITHM DETECTION")
    print("=" * 60)
    print(f"Algorithm: {info.algorithm.value}")
    print(f"Number of passes: {info.num_passes}")
    print(f"Uses tiling: {info.uses_tiling}")
    print(f"Uses correction: {info.uses_correction}")
    print()
    print("Key nodes:")
    print(f"  QK reduction: {info.qk_reduction_node}")
    print(f"  Global max: {info.global_max_node}")
    print(f"  Global sum: {info.global_sum_node}")
    print(f"  Output: {info.output_node}")
    print()
    print("Operations:", sorted(info.operations))

    # Show pass structure
    print()
    print("Pass Structure:")
    passes = detector.get_pass_structure()
    for pass_num, nodes in passes.items():
        print(f"  Pass {pass_num}:")
        for node_id in nodes:
            node = graph.nodes[node_id]
            print(f"    - {node.var_name} ({node.op})")
