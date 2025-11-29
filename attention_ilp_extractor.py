#!/usr/bin/env python3
"""
ILP Extractor for Attention E-graph
====================================

Processes egglog JSON output and uses Integer Linear Programming to select
the optimal implementation (2-pass or 3-pass) based on a configurable cost model.

Usage:
    python attention_ilp_extractor.py --input attention_rewrite.json --cost-model 2pass --output attention_2pass_extracted.json

Reference: egraph_isa_compiler_codesign/Extractor for ILP formulation
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# Cost Models
# ============================================================================

# Cost model favoring 2-pass (tiled operations are cheap)
COST_2PASS = {
    # 3-pass operations = EXPENSIVE
    'R_max_m': 100,      # Global max reduction
    'R_add_m': 100,      # Global sum reduction
    'M_div_mp': 100,     # In-loop division
    'M_sub_mp': 50,      # Non-tiled subtraction
    'M_exp_mp': 50,      # Non-tiled exp
    'M_mul_emp': 50,     # Non-tiled multiply

    # 2-pass operations = CHEAP
    'R_max_m0': 1,       # Local max (per tile)
    'R_max_m1': 10,      # Cross-tile max
    'R_add_m0': 1,       # Local sum
    'R_add_m1': 10,      # Cross-tile sum
    'M_div_fp': 1,       # Post-loop division
    'M_div_m1m0p': 5,    # Tiled in-loop division
    'M_exp_m1m0p': 1,    # Tiled exp
    'M_sub_m1m0p': 1,    # Tiled subtraction
    'M_sub_m1p': 1,      # Correction factor subtraction
    'M_exp_m1p': 1,      # Correction factor exp
    'M_mul_m1m0p': 1,    # Correction multiply
    'M_mul_m1p': 1,      # Correction multiply
    'M_mul_em1m0p': 1,   # Tiled QK multiply
    'T_split_m_m1m0': 1, # Tiling
    'T_unsplit_m1m0_m': 1,  # Untiling

    # Neutral operations
    'M_mul_fmp': 1,      # Multiply with V
    'R_add_e': 1,        # Sum over embedding dim
    'CreateTensor': 0,   # Input creation
}

# Cost model favoring 3-pass (global operations are cheap)
COST_3PASS = {
    # 3-pass operations = CHEAP
    'R_max_m': 1,        # Global max reduction
    'R_add_m': 1,        # Global sum reduction
    'M_div_mp': 1,       # In-loop division
    'M_sub_mp': 1,       # Non-tiled subtraction
    'M_exp_mp': 1,       # Non-tiled exp
    'M_mul_emp': 1,      # Non-tiled multiply

    # 2-pass operations = EXPENSIVE
    'R_max_m0': 100,     # Local max (per tile)
    'R_max_m1': 100,     # Cross-tile max
    'R_add_m0': 100,     # Local sum
    'R_add_m1': 100,     # Cross-tile sum
    'M_div_fp': 100,     # Post-loop division
    'M_div_m1m0p': 100,  # Tiled in-loop division
    'M_exp_m1m0p': 50,   # Tiled exp
    'M_sub_m1m0p': 50,   # Tiled subtraction
    'M_sub_m1p': 50,     # Correction factor subtraction
    'M_exp_m1p': 50,     # Correction factor exp
    'M_mul_m1m0p': 50,   # Correction multiply
    'M_mul_m1p': 50,     # Correction multiply
    'M_mul_em1m0p': 50,  # Tiled QK multiply
    'T_split_m_m1m0': 100,  # Tiling
    'T_unsplit_m1m0_m': 100,  # Untiling

    # Neutral operations
    'M_mul_fmp': 1,      # Multiply with V
    'R_add_e': 1,        # Sum over embedding dim
    'CreateTensor': 0,   # Input creation
}

DEFAULT_COST = 10  # Default cost for unknown ops

# Cost model favoring 2-pass with M_div_fp avoidance
# This makes M_div_fp expensive to prefer pure 2-pass without post-loop division
COST_2PASS_NO_MDIV = {
    # 3-pass operations = EXPENSIVE
    'R_max_m': 100,
    'R_add_m': 100,
    'M_div_mp': 100,
    'M_sub_mp': 50,
    'M_exp_mp': 50,
    'M_mul_emp': 50,

    # 2-pass operations = CHEAP
    'R_max_m0': 1,
    'R_max_m1': 10,
    'R_add_m0': 1,
    'R_add_m1': 10,
    'M_div_fp': 50,        # Make this MORE expensive to avoid problematic variants
    'M_div_m1m0p': 1,      # Make in-loop division CHEAP (prefer over post-loop)
    'M_exp_m1m0p': 1,
    'M_sub_m1m0p': 1,
    'M_sub_m1p': 1,
    'M_exp_m1p': 1,
    'M_mul_m1m0p': 1,
    'M_mul_m1p': 1,
    'M_mul_em1m0p': 1,
    'T_split_m_m1m0': 1,
    'T_unsplit_m1m0_m': 1,

    # Neutral operations
    'M_mul_fmp': 1,
    'R_add_e': 1,
    'CreateTensor': 0,
}


# ============================================================================
# E-graph Data Structures
# ============================================================================

def sanitize(s: str) -> str:
    """Convert string to valid ILP variable name (only letters, numbers, underscores)."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(s))


@dataclass
class ENode:
    """Represents a node in the e-graph."""
    enode_id: str
    eclass_id: str
    op: str
    children: List[str]  # List of child eclass IDs
    cost: float

    def __repr__(self):
        return f"ENode({self.op}, eclass={self.eclass_id}, children={self.children})"


@dataclass
class EClass:
    """Represents an equivalence class in the e-graph."""
    eclass_id: str
    member_enodes: Set[str] = field(default_factory=set)
    parent_enodes: Set[str] = field(default_factory=set)

    def __repr__(self):
        return f"EClass({self.eclass_id}, members={len(self.member_enodes)})"


class EGraph:
    """E-graph representation parsed from egglog JSON."""

    def __init__(self):
        self.enodes: Dict[str, ENode] = {}
        self.eclasses: Dict[str, EClass] = {}
        self.root_eclasses: List[str] = []
        self.class_data: Dict[str, dict] = {}

    @classmethod
    def from_json_file(cls, filepath: str) -> 'EGraph':
        """Load e-graph from egglog JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_json(data)

    @classmethod
    def from_json(cls, data: dict) -> 'EGraph':
        """Parse e-graph from egglog JSON data."""
        graph = cls()
        nodes_data = data.get('nodes', {})
        graph.root_eclasses = data.get('root_eclasses', [])
        graph.class_data = data.get('class_data', {})

        # Build node ID to eclass mapping first
        node_to_eclass = {}
        for node_id, node_data in nodes_data.items():
            eclass_id = node_data.get('eclass')
            if eclass_id:
                node_to_eclass[node_id] = eclass_id

        # Parse nodes and build eclasses
        for node_id, node_data in nodes_data.items():
            eclass_id = node_data.get('eclass')
            if not eclass_id:
                continue

            op = node_data.get('op', '')
            original_cost = node_data.get('cost', 1.0)

            # Convert children from node IDs to eclass IDs
            child_node_ids = node_data.get('children', [])
            child_eclass_ids = []
            for child_node_id in child_node_ids:
                child_eclass = node_to_eclass.get(child_node_id)
                if child_eclass:
                    child_eclass_ids.append(child_eclass)

            # Create ENode
            enode = ENode(
                enode_id=node_id,
                eclass_id=eclass_id,
                op=op,
                children=child_eclass_ids,
                cost=original_cost
            )
            graph.enodes[node_id] = enode

            # Create/update EClass
            if eclass_id not in graph.eclasses:
                graph.eclasses[eclass_id] = EClass(eclass_id)
            graph.eclasses[eclass_id].member_enodes.add(node_id)

            # Update parent links
            for child_eclass_id in child_eclass_ids:
                if child_eclass_id not in graph.eclasses:
                    graph.eclasses[child_eclass_id] = EClass(child_eclass_id)
                graph.eclasses[child_eclass_id].parent_enodes.add(node_id)

        # If no root eclasses specified, find the output (AV)
        if not graph.root_eclasses:
            graph.root_eclasses = graph._find_root_eclasses()

        return graph

    def _find_root_eclasses(self) -> List[str]:
        """Find root eclasses (those with no parents or those named 'AV')."""
        roots = []

        # Look for AV in class_data
        for eclass_id, data in self.class_data.items():
            if data.get('let') == 'AV':
                roots.append(eclass_id)
                return roots

        # Fallback: find eclasses with no parent nodes
        for eclass_id, eclass in self.eclasses.items():
            if not eclass.parent_enodes:
                # Skip primitive types
                if not eclass_id.startswith('i64') and not eclass_id.startswith('String'):
                    roots.append(eclass_id)

        return roots

    def apply_cost_model(self, cost_model: Dict[str, float]):
        """Override node costs based on cost model."""
        for enode in self.enodes.values():
            if enode.op in cost_model:
                enode.cost = cost_model[enode.op]
            else:
                enode.cost = DEFAULT_COST


# ============================================================================
# ILP Generator
# ============================================================================

class ILPGenerator:
    """Generates ILP formulation for e-graph extraction."""

    def __init__(self, egraph: EGraph):
        self.egraph = egraph

        # Variable mappings
        self.class_active_vars: Dict[str, str] = {}  # eclass_id -> A_var
        self.node_vars: Dict[Tuple[str, str], str] = {}  # (eclass_id, node_id) -> N_var
        self.level_vars: Dict[str, str] = {}  # eclass_id -> L_var
        self.opposite_vars: Dict[Tuple[str, str], str] = {}  # (eclass_id, node_id) -> Opp_var
        self.op_vars: Dict[str, str] = {}  # op_name -> Op_var
        self.node_to_op: Dict[str, str] = {}  # node_id -> op_name

    def generate(self, output_path: str):
        """Generate ILP file in LP format."""
        self._create_variables()

        lp_lines = []
        lp_lines.extend(self._generate_objective())
        lp_lines.extend(self._generate_constraints())
        lp_lines.extend(self._generate_bounds())
        lp_lines.extend(self._generate_variable_declarations())
        lp_lines.append("End")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lp_lines))

        print(f"ILP file generated: {output_path}")
        print(f"  - EClasses: {len(self.class_active_vars)}")
        print(f"  - Nodes: {len(self.node_vars)}")
        print(f"  - Operators: {len(self.op_vars)}")

    def _create_variables(self):
        """Create all ILP variables."""
        all_ops = set()

        for eclass_id, eclass in self.egraph.eclasses.items():
            # Skip primitive types (i64, String, etc.)
            if eclass_id.startswith('i64') or eclass_id.startswith('String'):
                continue

            # A_<eclass_id>: eclass activation
            a_var = f"A_{sanitize(eclass_id)}"
            self.class_active_vars[eclass_id] = a_var

            # L_<eclass_id>: level for cycle prevention
            l_var = f"L_{sanitize(eclass_id)}"
            self.level_vars[eclass_id] = l_var

            for node_id in eclass.member_enodes:
                if node_id not in self.egraph.enodes:
                    continue

                # N_<eclass_id>_<node_id>: node selection
                n_var = f"N_{sanitize(eclass_id)}_{sanitize(node_id)}"
                self.node_vars[(eclass_id, node_id)] = n_var

                # Opp_<eclass_id>_<node_id>: opposite for cycle prevention
                opp_var = f"Opp_{sanitize(eclass_id)}_{sanitize(node_id)}"
                self.opposite_vars[(eclass_id, node_id)] = opp_var

                # Record operator
                op_name = self.egraph.enodes[node_id].op
                self.node_to_op[node_id] = op_name
                all_ops.add(op_name)

        # Op_<op_name>: operator usage
        for op_name in sorted(all_ops):
            op_var = f"Op_{sanitize(op_name)}"
            self.op_vars[op_name] = op_var

    def _generate_objective(self) -> List[str]:
        """Generate objective function: minimize total node cost."""
        lines = ["Minimize"]
        obj_terms = []

        # Sum of node costs
        for (eclass_id, node_id), n_var in self.node_vars.items():
            node = self.egraph.enodes[node_id]
            cost = node.cost

            if cost != 0:
                if cost == 1:
                    obj_terms.append(n_var)
                else:
                    obj_terms.append(f"{cost} {n_var}")

        if obj_terms:
            lines.append(" obj: " + " + ".join(obj_terms))
        else:
            lines.append(" obj: 0")

        lines.append("")
        return lines

    def _generate_constraints(self) -> List[str]:
        """Generate all constraints."""
        lines = ["Subject To"]

        # C1: One node per activated eclass
        lines.extend(self._constraint_one_node_per_eclass())

        # C2: Child activation
        lines.extend(self._constraint_child_activation())

        # C3: Root activation
        lines.extend(self._constraint_root_activation())

        # C4: Intersection constraint (optional optimization)
        lines.extend(self._constraint_intersection())

        # C5: Self-loop prevention
        lines.extend(self._constraint_self_loop())

        # C6: Cycle prevention (level ordering)
        lines.extend(self._constraint_cycle_prevention())

        # C7: Operator activation (optional, for operator minimization)
        lines.extend(self._constraint_operator_activation())

        # C8: M_div_fp must have distinct children (numerator != denominator)
        lines.extend(self._constraint_mdiv_distinct_children())

        lines.append("")
        return lines

    def _constraint_mdiv_distinct_children(self) -> List[str]:
        """C8: Ensure M_div_fp gets distinct numerator and denominator."""
        lines = []

        # Find all M_div_fp nodes
        for (eclass_id, node_id), n_var in self.node_vars.items():
            if node_id not in self.egraph.enodes:
                continue

            node = self.egraph.enodes[node_id]
            if node.op == 'M_div_fp' and len(node.children) == 2:
                child0_eclass = node.children[0]
                child1_eclass = node.children[1]

                # If both children from same eclass, need to select different nodes
                if child0_eclass == child1_eclass:
                    # Find all R_add_m1 nodes in this eclass
                    if child0_eclass in self.egraph.eclasses:
                        r_add_nodes = [
                            nid for nid in self.egraph.eclasses[child0_eclass].member_enodes
                            if nid in self.egraph.enodes and self.egraph.enodes[nid].op == 'R_add_m1'
                        ]

                        if len(r_add_nodes) >= 2:
                            # If M_div_fp is selected, at least 2 different R_add_m1 must be selected
                            node_vars = [self.node_vars.get((child0_eclass, r_nid)) for r_nid in r_add_nodes if (child0_eclass, r_nid) in self.node_vars]

                            if len(node_vars) >= 2:
                                # Sum of selected R_add_m1 nodes >= 2 * M_div_fp selection
                                constraint = f" MDIV_DISTINCT_{sanitize(eclass_id)}_{sanitize(node_id)}: {' + '.join(node_vars)} - 2 {n_var} >= 0"
                                lines.append(constraint)

        return lines

    def _constraint_one_node_per_eclass(self) -> List[str]:
        """C1: Sum of N vars = A var for each eclass."""
        lines = []

        for eclass_id in self.class_active_vars:
            if eclass_id not in self.egraph.eclasses:
                continue

            eclass = self.egraph.eclasses[eclass_id]
            a_var = self.class_active_vars[eclass_id]

            sum_terms = []
            for node_id in eclass.member_enodes:
                if (eclass_id, node_id) in self.node_vars:
                    sum_terms.append(self.node_vars[(eclass_id, node_id)])

            if sum_terms:
                constraint = f" C_ACT_{sanitize(eclass_id)}: {' + '.join(sum_terms)} - {a_var} = 0"
                lines.append(constraint)

        return lines

    def _constraint_child_activation(self) -> List[str]:
        """C2: If a node is selected, all its child eclasses must be activated."""
        lines = []

        for (eclass_id, node_id), n_var in self.node_vars.items():
            if node_id not in self.egraph.enodes:
                continue

            node = self.egraph.enodes[node_id]
            child_eclasses = set(node.children)

            for child_eclass_id in child_eclasses:
                if child_eclass_id in self.class_active_vars:
                    child_a_var = self.class_active_vars[child_eclass_id]
                    constraint = f" NODE_CHILD_{sanitize(eclass_id)}_{sanitize(node_id)}_{sanitize(child_eclass_id)}: {n_var} - {child_a_var} <= 0"
                    lines.append(constraint)

        return lines

    def _constraint_root_activation(self) -> List[str]:
        """C3: Root eclasses must be activated."""
        lines = []

        for root_id in self.egraph.root_eclasses:
            if root_id in self.class_active_vars:
                a_var = self.class_active_vars[root_id]
                constraint = f" ROOT_{sanitize(root_id)}: {a_var} >= 1"
                lines.append(constraint)

        return lines

    def _constraint_intersection(self) -> List[str]:
        """C4: If all nodes in eclass share a child, that child must be activated."""
        lines = []

        for eclass_id, eclass in self.egraph.eclasses.items():
            if eclass_id not in self.class_active_vars:
                continue

            node_list = [nid for nid in eclass.member_enodes if nid in self.egraph.enodes]
            if not node_list:
                continue

            # Calculate intersection of child classes
            first_node = self.egraph.enodes[node_list[0]]
            intersection = set(first_node.children)

            for node_id in node_list[1:]:
                node = self.egraph.enodes[node_id]
                intersection = intersection.intersection(set(node.children))

            # Add constraint for each common child
            for child_eclass_id in intersection:
                if child_eclass_id in self.class_active_vars:
                    constraint = f" INTERSECT_{sanitize(eclass_id)}_{sanitize(child_eclass_id)}: {self.class_active_vars[eclass_id]} - {self.class_active_vars[child_eclass_id]} <= 0"
                    lines.append(constraint)

        return lines

    def _constraint_self_loop(self) -> List[str]:
        """C5: Nodes that reference their own eclass are invalid."""
        lines = []

        for (eclass_id, node_id), n_var in self.node_vars.items():
            if node_id not in self.egraph.enodes:
                continue

            node = self.egraph.enodes[node_id]
            if eclass_id in node.children:
                constraint = f" SELF_LOOP_{sanitize(eclass_id)}_{sanitize(node_id)}: {n_var} = 0"
                lines.append(constraint)

        return lines

    def _constraint_cycle_prevention(self) -> List[str]:
        """C6: Level ordering to prevent cycles."""
        lines = []
        num_classes = len(self.class_active_vars)
        M = num_classes + 1  # Big-M constant

        # Opposite variable constraint: N + Opp = 1
        for (eclass_id, node_id), n_var in self.node_vars.items():
            opp_var = self.opposite_vars[(eclass_id, node_id)]
            constraint = f" OPP_{sanitize(eclass_id)}_{sanitize(node_id)}: {n_var} + {opp_var} = 1"
            lines.append(constraint)

        # Level constraint: L_child - L_parent + M * Opp >= 1
        for (eclass_id, node_id), n_var in self.node_vars.items():
            if eclass_id not in self.level_vars:
                continue
            if node_id not in self.egraph.enodes:
                continue

            node = self.egraph.enodes[node_id]
            opp_var = self.opposite_vars[(eclass_id, node_id)]
            level_var = self.level_vars[eclass_id]

            # For each non-self-loop child
            child_eclasses = set(node.children)
            child_eclasses.discard(eclass_id)

            for child_eclass_id in child_eclasses:
                if child_eclass_id in self.level_vars:
                    child_level_var = self.level_vars[child_eclass_id]
                    constraint = f" LEVEL_{sanitize(eclass_id)}_{sanitize(node_id)}_{sanitize(child_eclass_id)}: {child_level_var} - {level_var} + {M} {opp_var} >= 1"
                    lines.append(constraint)

        return lines

    def _constraint_operator_activation(self) -> List[str]:
        """C7: If a node is selected, its operator must be marked as used."""
        lines = []

        for (eclass_id, node_id), n_var in self.node_vars.items():
            op_name = self.node_to_op.get(node_id)
            if op_name and op_name in self.op_vars:
                op_var = self.op_vars[op_name]
                constraint = f" OP_ACT_{sanitize(eclass_id)}_{sanitize(node_id)}: {n_var} - {op_var} <= 0"
                lines.append(constraint)

        return lines

    def _generate_bounds(self) -> List[str]:
        """Generate variable bounds."""
        lines = ["Bounds"]
        num_classes = len(self.class_active_vars)

        # Level variable bounds
        for eclass_id, level_var in self.level_vars.items():
            lines.append(f" 0 <= {level_var} <= {num_classes}")

        lines.append("")
        return lines

    def _generate_variable_declarations(self) -> List[str]:
        """Generate binary and integer variable declarations."""
        lines = ["Binaries"]

        # Binary variables
        for var in self.class_active_vars.values():
            lines.append(f" {var}")
        for var in self.node_vars.values():
            lines.append(f" {var}")
        for var in self.opposite_vars.values():
            lines.append(f" {var}")
        for var in self.op_vars.values():
            lines.append(f" {var}")

        lines.append("")
        lines.append("Generals")

        # Integer variables (level)
        for var in self.level_vars.values():
            lines.append(f" {var}")

        lines.append("")
        return lines


# ============================================================================
# ILP Solver
# ============================================================================

class ILPSolver:
    """Wrapper for ILP solvers (Gurobi, CBC, PuLP)."""

    def __init__(self, solver: str = 'pulp'):
        self.solver = solver

    def solve(self, lp_file: str, sol_file: str, timeout: int = 300) -> bool:
        """Solve the ILP and return success status."""
        if self.solver == 'gurobi':
            return self._solve_gurobi(lp_file, sol_file, timeout)
        elif self.solver == 'cbc':
            return self._solve_cbc(lp_file, sol_file, timeout)
        elif self.solver == 'pulp':
            return self._solve_pulp(lp_file, sol_file, timeout)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def _solve_gurobi(self, lp_file: str, sol_file: str, timeout: int) -> bool:
        """Solve using Gurobi."""
        try:
            import gurobipy as gp

            model = gp.read(lp_file)
            model.Params.TimeLimit = timeout
            model.optimize()

            if model.Status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) and model.SolCount > 0:
                # Write solution
                with open(sol_file, 'w') as f:
                    for var in model.getVars():
                        if abs(var.X) > 0.001:
                            f.write(f"{var.VarName} {var.X}\n")
                print(f"Solution found with objective: {model.ObjVal}")
                return True
            else:
                print(f"No solution found (status: {model.Status})")
                return False

        except ImportError:
            print("Gurobi not available, trying CBC...")
            return self._solve_cbc(lp_file, sol_file, timeout)
        except Exception as e:
            print(f"Gurobi error: {e}")
            return False

    def _solve_cbc(self, lp_file: str, sol_file: str, timeout: int) -> bool:
        """Solve using CBC (COIN-OR Branch and Cut)."""
        try:
            cmd = [
                'cbc', lp_file,
                f'sec', str(timeout),
                'solve',
                'solu', sol_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)

            if os.path.exists(sol_file):
                print("CBC solution found")
                return True
            else:
                print(f"CBC failed: {result.stderr}")
                return False

        except FileNotFoundError:
            print("CBC solver not found. Install with: apt install coinor-cbc")
            return False
        except Exception as e:
            print(f"CBC error: {e}")
            return False

    def parse_solution(self, sol_file: str) -> Dict[str, float]:
        """Parse solution file."""
        variables = {}

        if not os.path.exists(sol_file):
            return variables

        with open(sol_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Optimal'):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    var_name = parts[0]
                    try:
                        # CBC format: index var_name value
                        if parts[0].isdigit():
                            var_name = parts[1]
                            value = float(parts[2])
                        else:
                            value = float(parts[1])

                        if abs(value) > 0.001:
                            variables[var_name] = value
                    except (ValueError, IndexError):
                        continue

        return variables


# ============================================================================
# Solution Extractor
# ============================================================================

class SolutionExtractor:
    """Extracts selected nodes from ILP solution."""

    def __init__(self, egraph: EGraph, ilp_gen: ILPGenerator):
        self.egraph = egraph
        self.ilp_gen = ilp_gen

    def extract(self, variables: Dict[str, float]) -> Dict[str, str]:
        """Extract selected node for each eclass.

        Returns: {eclass_id: node_id}
        """
        # Build reverse mapping: var_name -> (eclass_id, node_id)
        var_to_node = {}
        for (eclass_id, node_id), n_var in self.ilp_gen.node_vars.items():
            var_to_node[n_var] = (eclass_id, node_id)

        choices = {}
        for var_name, value in variables.items():
            if var_name.startswith("N_") and abs(value - 1.0) < 0.01:
                if var_name in var_to_node:
                    eclass_id, node_id = var_to_node[var_name]
                    choices[eclass_id] = node_id

        return choices

    def build_extracted_json(self, choices: Dict[str, str], original_data: dict) -> dict:
        """Build JSON with only selected nodes, including primitives."""
        nodes = {}
        original_nodes = original_data.get('nodes', {})

        # First pass: check if we need to add missing R_add_m1 nodes for M_div_fp
        additional_nodes = set()
        for eclass_id, node_id in choices.items():
            if node_id in original_nodes and original_nodes[node_id].get('op') == 'M_div_fp':
                # Check if children are duplicates or missing
                orig_children = original_nodes[node_id].get('children', [])
                if len(orig_children) == 2:
                    # Map children through choices
                    child0_eclass = original_nodes.get(orig_children[0], {}).get('eclass')
                    child1_eclass = original_nodes.get(orig_children[1], {}).get('eclass')

                    if child0_eclass and child1_eclass and child0_eclass == child1_eclass:
                        # Both from same eclass - need to find alternatives
                        print(f"  M_div_fp requires distinct children from eclass {child0_eclass}")

                        # Find all R_add_m1 in original graph
                        # Look 2 levels deep: R_add_m1 → R_add_m0 → M_mul_fmp/M_mul_m1p
                        r_add_candidates = []
                        for o_nid, o_node in original_nodes.items():
                            if o_node.get('op') == 'R_add_m1':
                                o_children = o_node.get('children', [])
                                if o_children and o_children[0] in original_nodes:
                                    # Level 1: typically R_add_m0 or M_mul_m1p
                                    child1 = original_nodes[o_children[0]]
                                    child1_op = child1.get('op', '')

                                    # If child is R_add_m0, look one level deeper
                                    if child1_op == 'R_add_m0' and child1.get('children'):
                                        gc_id = child1.get('children')[0]
                                        if gc_id in original_nodes:
                                            child1_op = original_nodes[gc_id].get('op', '')

                                    r_add_candidates.append((o_nid, child1_op))

                        # Filter candidates to only use nodes already in ILP-selected set
                        # This avoids pulling in mixed algorithm variants
                        selected_node_ids = set(choices.values())
                        selected_eclasses = set(choices.keys())  # For nested function access

                        # Function to check if node uses only ILP-selected operations
                        def uses_selected_ops_only(nid, depth=0):
                            if depth > 3 or nid not in original_nodes:
                                return True
                            node = original_nodes[nid]
                            # Check if this node was selected by ILP
                            # If not, it might use conflicting operations
                            for child in node.get('children', []):
                                if child not in selected_node_ids and not uses_selected_ops_only(child, depth + 1):
                                    return False
                            return True

                        # Pick two with different grandchildren, preferring ILP-selected ops
                        output_r_add = None
                        denom_r_add = None

                        print(f"    Found {len(r_add_candidates)} R_add_m1 candidates:")
                        for r_nid, gc_op in r_add_candidates:
                            uses_selected = uses_selected_ops_only(r_nid)
                            status = "ILP-selected" if r_nid in selected_node_ids else "external"
                            print(f"      {r_nid}: grandchild_op={gc_op} ({status}, uses_selected_ops={uses_selected})")

                            if ('mul_fmp' in gc_op.lower() or 'mul_fm' in gc_op.lower()) and not output_r_add:
                                output_r_add = r_nid
                                print(f"        → Selected as OUTPUT")
                            elif 'mul_m1p' in gc_op.lower() and not denom_r_add:
                                denom_r_add = r_nid
                                print(f"        → Selected as DENOMINATOR")

                        if output_r_add and denom_r_add:
                            # Define 2-pass operations (no 3-pass global ops)
                            TWOPASS_OPS = {
                                'M_mul_em1m0p', 'M_sub_m1m0p', 'M_exp_m1m0p',
                                'M_sub_m1p', 'M_exp_m1p', 'M_mul_m1m0p', 'M_mul_m1p',
                                'R_max_m0', 'R_max_m1', 'R_add_m0', 'R_add_m1',
                                'T_split_m_m1m0', 'T_unsplit_m1m0_m', 'M_div_fp',
                                'M_mul_fmp', 'R_add_e', 'M_mul_emp', 'CreateTensor'
                            }

                            # Recursively add all 2-pass dependencies
                            def add_2pass_deps(nid, depth=0, is_mdiv_child=False):
                                if depth > 10 or nid in additional_nodes:
                                    return
                                if nid not in original_nodes:
                                    return

                                node = original_nodes[nid]
                                op = node.get('op', '')
                                node_eclass = node.get('eclass')

                                # Skip 3-pass operations and operations that use them
                                if op in ['M_div_mp', 'M_sub_mp', 'M_exp_mp', 'R_max_m', 'R_add_m', 'M_div_m1m0p']:
                                    print(f"      {'  ' * depth}Skipping 3-pass/mixed op: {nid} ({op})")
                                    return

                                # Check if there's an ILP-selected node with same operation
                                # Prefer ILP-selected over external nodes UNLESS this is for M_div_fp
                                # (M_div_fp needs distinct nodes, not same ILP node)
                                if nid not in selected_node_ids and not is_mdiv_child:
                                    # This is an external node - check if ILP selected same operation
                                    for ilp_nid in selected_node_ids:
                                        ilp_node = original_nodes.get(ilp_nid, {})
                                        if ilp_node.get('op') == op:
                                            # Found ILP-selected node with same operation!
                                            print(f"      {'  ' * depth}Using ILP-selected {op}: {ilp_nid} instead of {nid}")
                                            add_2pass_deps(ilp_nid, depth, is_mdiv_child)
                                            return

                                additional_nodes.add(nid)
                                if depth == 0:
                                    print(f"    Adding R_add_m1: {nid}")
                                else:
                                    # Debug: show if this is ILP-selected
                                    is_ilp = "ILP" if nid in selected_node_ids else "new"
                                    print(f"      {'  ' * depth}{nid} ({op}) [{is_ilp}]")

                                # Recursively add children
                                # Mark as M_div_fp child only for top-level (depth==0)
                                for c in node.get('children', []):
                                    add_2pass_deps(c, depth + 1, is_mdiv_child and depth == 0)

                            print(f"    ✓ Recursively adding 2-pass dependencies (keeping M_div_fp children distinct):")
                            add_2pass_deps(output_r_add, 0, is_mdiv_child=True)
                            add_2pass_deps(denom_r_add, 0, is_mdiv_child=True)
                        else:
                            print(f"    ✗ Could not find both output and denom: output={output_r_add}, denom={denom_r_add}")

        # Build mapping from eclass to original node IDs (for primitives)
        eclass_to_original_nodes = {}
        for node_id, node_data in original_nodes.items():
            eclass_id = node_data.get('eclass')
            if eclass_id:
                if eclass_id not in eclass_to_original_nodes:
                    eclass_to_original_nodes[eclass_id] = []
                eclass_to_original_nodes[eclass_id].append(node_id)

        # Find all primitive eclasses (i64, String)
        primitive_choices = {}
        for eclass_id, node_ids in eclass_to_original_nodes.items():
            if eclass_id.startswith('i64') or eclass_id.startswith('String'):
                # Pick the first (and usually only) node for this primitive
                if node_ids:
                    primitive_choices[eclass_id] = node_ids[0]

        # Merge primitive choices with ILP choices
        all_choices = {**primitive_choices, **choices}

        # Include selected nodes, primitives, and additional nodes for M_div_fp fix
        included_nodes = set(all_choices.values()) | additional_nodes

        if additional_nodes:
            print(f"  Total nodes after M_div_fp fix: {len(included_nodes)} (added {len(additional_nodes)} nodes)")
            print(f"    Additional nodes: {list(additional_nodes)}")

        for node_id in included_nodes:
            # Get from original data to preserve structure
            if node_id in original_nodes:
                # Debug: track if this is an additional node
                if node_id in additional_nodes:
                    print(f"    Processing additional node: {node_id} ({original_nodes[node_id].get('op')})")
                orig_node = original_nodes[node_id]
                eclass_id = orig_node.get('eclass')

                # Get original children (node IDs)
                orig_children = orig_node.get('children', [])

                # For non-primitive nodes, map children through choices
                if not (eclass_id and (eclass_id.startswith('i64') or eclass_id.startswith('String'))):
                    child_node_ids = []

                    # Special case: M_div_fp needs distinct numerator and denominator
                    if orig_node.get('op') == 'M_div_fp' and len(orig_children) == 2:
                        child0_id = orig_children[0]
                        child1_id = orig_children[1]
                        child0_eclass = original_nodes.get(child0_id, {}).get('eclass')
                        child1_eclass = original_nodes.get(child1_id, {}).get('eclass')

                        # If both children from same eclass OR same node ID, find DIFFERENT nodes
                        if (child0_eclass == child1_eclass or child0_id == child1_id) and child0_eclass:
                            # Find all nodes of same operation type in this eclass
                            child0_op = original_nodes.get(child0_id, {}).get('op', '')
                            same_eclass_nodes = [
                                nid for nid, n in original_nodes.items()
                                if n.get('eclass') == child0_eclass and n.get('op') == child0_op
                            ]

                            if len(same_eclass_nodes) >= 2:
                                # Use first two different nodes
                                child_node_ids = same_eclass_nodes[:2]
                                print(f"  M_div_fp fix: Using distinct {child0_op} nodes: {child_node_ids[0]}, {child_node_ids[1]}")
                            else:
                                # Only one node in this eclass - use nodes from additional_nodes
                                # These were already added by the first M_div_fp fix pass
                                print(f"  M_div_fp fix: Using nodes from extracted graph...")
                                all_r_add_m1_in_extracted = [
                                    nid for nid in included_nodes
                                    if nid in original_nodes and original_nodes[nid].get('op') == 'R_add_m1'
                                ]

                                if len(all_r_add_m1_in_extracted) >= 2:
                                    # Prefer ones that reduce different things (look 2 levels deep)
                                    output_r_add = None
                                    denom_r_add = None

                                    for r_nid in all_r_add_m1_in_extracted:
                                        r_node = original_nodes[r_nid]
                                        r_children = r_node.get('children', [])
                                        if r_children and r_children[0] in original_nodes:
                                            child1 = original_nodes[r_children[0]]
                                            grandchild_op = child1.get('op', '')

                                            # Look deeper if it's R_add_m0
                                            if grandchild_op == 'R_add_m0' and child1.get('children'):
                                                gc_id = child1.get('children')[0]
                                                if gc_id in original_nodes:
                                                    grandchild_op = original_nodes[gc_id].get('op', '')

                                            if ('mul_fmp' in grandchild_op.lower() or 'mul_fm' in grandchild_op.lower()) and not output_r_add:
                                                output_r_add = r_nid
                                            elif 'mul_m1p' in grandchild_op.lower() and not denom_r_add:
                                                denom_r_add = r_nid

                                    if output_r_add and denom_r_add:
                                        child_node_ids = [output_r_add, denom_r_add]
                                        print(f"    Using: numerator={output_r_add}, denominator={denom_r_add}")
                                    else:
                                        # Fallback: use first two
                                        child_node_ids = all_r_add_m1_in_extracted[:2]
                                        print(f"    Using first two from extracted: {child_node_ids}")
                                else:
                                    # Can't fix - use original
                                    child_node_ids = orig_children
                        else:
                            # Different eclasses - map through choices
                            for child_node_id in orig_children:
                                child_eclass = original_nodes.get(child_node_id, {}).get('eclass')
                                if child_eclass and child_eclass in all_choices:
                                    child_node_ids.append(all_choices[child_eclass])
                                else:
                                    child_node_ids.append(child_node_id)
                    else:
                        # Normal mapping for other operations
                        for child_node_id in orig_children:
                            child_eclass = original_nodes.get(child_node_id, {}).get('eclass')
                            if child_eclass and child_eclass in all_choices:
                                child_node_ids.append(all_choices[child_eclass])
                            else:
                                # Keep original reference for primitives
                                child_node_ids.append(child_node_id)
                else:
                    child_node_ids = orig_children

                nodes[node_id] = {
                    "op": orig_node.get('op', ''),
                    "children": child_node_ids,
                    "eclass": eclass_id,
                    "cost": orig_node.get('cost', 1.0),
                    "subsumed": False
                }
            elif node_id in self.egraph.enodes:
                # Fallback to egraph data
                enode = self.egraph.enodes[node_id]
                child_node_ids = []
                for child_eclass_id in enode.children:
                    if child_eclass_id in all_choices:
                        child_node_ids.append(all_choices[child_eclass_id])

                nodes[node_id] = {
                    "op": enode.op,
                    "children": child_node_ids,
                    "eclass": enode.eclass_id,
                    "cost": enode.cost,
                    "subsumed": False
                }

        # Post-process: remap children to use nodes that actually exist
        # This fixes cases where child pointers reference nodes not in extraction
        print("\n  Post-processing: Remapping children to existing nodes...")
        remapped_count = 0
        for node_id, node_data in nodes.items():
            if 'children' in node_data:
                new_children = []
                for child_id in node_data['children']:
                    if child_id in nodes:
                        # Child exists, keep it
                        new_children.append(child_id)
                    else:
                        # Child missing - find alternative with same operation
                        if child_id in original_nodes:
                            missing_op = original_nodes[child_id].get('op', '')
                            # Find a node with same operation that exists in extraction
                            found = False
                            for alt_id, alt_data in nodes.items():
                                if alt_data.get('op') == missing_op and alt_id != child_id:
                                    print(f"    Remapping {node_id}.child: {child_id}({missing_op}) → {alt_id}")
                                    new_children.append(alt_id)
                                    remapped_count += 1
                                    found = True
                                    break
                            if not found:
                                # Keep original (will cause error, but shows the issue)
                                new_children.append(child_id)
                        else:
                            new_children.append(child_id)

                node_data['children'] = new_children

        if remapped_count > 0:
            print(f"  Remapped {remapped_count} child pointers")

        # Filter class_data to only include selected eclasses
        class_data = {}
        original_class_data = original_data.get('class_data', {})
        for eclass_id in all_choices.keys():
            if eclass_id in original_class_data:
                class_data[eclass_id] = original_class_data[eclass_id]
            elif eclass_id in self.egraph.class_data:
                class_data[eclass_id] = self.egraph.class_data[eclass_id]

        return {
            "nodes": nodes,
            "root_eclasses": self.egraph.root_eclasses,
            "class_data": class_data,
            "extraction_info": {
                "num_eclasses": len(all_choices),
                "num_nodes": len(nodes),
                "selected_ops": list(set(
                    nodes[nid].get('op', '') for nid in nodes.keys()
                    if not nodes[nid].get('eclass', '').startswith('i64')
                    and not nodes[nid].get('eclass', '').startswith('String')
                ))
            }
        }

    def analyze(self, choices: Dict[str, str]) -> dict:
        """Analyze the extracted solution."""
        selected_nodes = set(choices.values())
        op_counts = {}
        total_cost = 0

        for node_id in selected_nodes:
            if node_id not in self.egraph.enodes:
                continue

            node = self.egraph.enodes[node_id]
            op_counts[node.op] = op_counts.get(node.op, 0) + 1
            total_cost += node.cost

        print("\n" + "=" * 60)
        print("Extraction Analysis")
        print("=" * 60)
        print(f"Selected eclasses: {len(choices)}")
        print(f"Selected nodes: {len(selected_nodes)}")
        print(f"Total cost: {total_cost}")
        print("\nOperator usage:")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            print(f"  {op}: {count}")

        # Determine algorithm type
        has_2pass_ops = any(op in op_counts for op in ['R_max_m0', 'R_add_m0', 'M_div_fp', 'T_split_m_m1m0'])
        has_3pass_ops = any(op in op_counts for op in ['R_max_m', 'R_add_m', 'M_div_mp'])

        if has_2pass_ops and not has_3pass_ops:
            algo = "2-pass (tiled)"
        elif has_3pass_ops and not has_2pass_ops:
            algo = "3-pass (global)"
        else:
            algo = "mixed"

        print(f"\nDetected algorithm type: {algo}")

        return {
            "num_eclasses": len(choices),
            "num_nodes": len(selected_nodes),
            "total_cost": total_cost,
            "op_counts": op_counts,
            "algorithm_type": algo
        }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ILP Extractor for Attention E-graph"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input egglog JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file (default: <input>_extracted.json)"
    )
    parser.add_argument(
        "--cost-model", "-c",
        choices=['2pass', '3pass', '2pass-no-mdiv', 'original'],
        default='2pass',
        help="Cost model to use (default: 2pass)"
    )
    parser.add_argument(
        "--solver", "-s",
        choices=['gurobi', 'cbc'],
        default='gurobi',
        help="ILP solver to use (default: gurobi)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=300,
        help="Solver timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--lp-output",
        help="Output LP file path (optional, for debugging)"
    )

    args = parser.parse_args()

    # Set output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.input)[0]
        output_path = f"{base}_{args.cost_model}_extracted.json"

    print("=" * 60)
    print("Attention ILP Extractor")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Cost model: {args.cost_model}")
    print(f"Solver: {args.solver}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Load e-graph and original data
    print("Loading e-graph...")
    with open(args.input, 'r') as f:
        original_data = json.load(f)
    egraph = EGraph.from_json(original_data)
    print(f"  EClasses: {len(egraph.eclasses)}")
    print(f"  ENodes: {len(egraph.enodes)}")
    print(f"  Root eclasses: {egraph.root_eclasses}")

    # Apply cost model
    if args.cost_model == '2pass':
        cost_model = COST_2PASS
    elif args.cost_model == '3pass':
        cost_model = COST_3PASS
    elif args.cost_model == '2pass-no-mdiv':
        cost_model = COST_2PASS_NO_MDIV
    else:
        cost_model = {}  # Use original costs

    if cost_model:
        print(f"\nApplying {args.cost_model} cost model...")
        egraph.apply_cost_model(cost_model)

    # Generate ILP
    lp_path = args.lp_output or f"/tmp/attention_extraction.lp"
    sol_path = f"/tmp/attention_extraction.sol"

    print(f"\nGenerating ILP...")
    ilp_gen = ILPGenerator(egraph)
    ilp_gen.generate(lp_path)

    # Solve ILP
    print(f"\nSolving ILP...")
    solver = ILPSolver(args.solver)
    success = solver.solve(lp_path, sol_path, args.timeout)

    if not success:
        print("\nILP solving failed!")
        return 1

    # Extract solution
    print(f"\nExtracting solution...")
    variables = solver.parse_solution(sol_path)
    print(f"  Parsed {len(variables)} variables")

    extractor = SolutionExtractor(egraph, ilp_gen)
    choices = extractor.extract(variables)
    print(f"  Selected {len(choices)} eclasses")

    # Analyze
    analysis = extractor.analyze(choices)

    # Build and save extracted JSON
    extracted_json = extractor.build_extracted_json(choices, original_data)
    extracted_json["analysis"] = analysis

    with open(output_path, 'w') as f:
        json.dump(extracted_json, f, indent=2)

    print(f"\nExtracted JSON saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
