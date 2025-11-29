# ILP Extractor for Attention E-graph

## Overview

The ILP extractor processes the full egglog JSON output (containing all equivalent expressions) and uses Integer Linear Programming to select the optimal implementation (2-pass or 3-pass) based on a configurable cost model.

## E-graph Structure (from egglog JSON)

```
nodes: {
  "node_id": {
    "op": "R_max_m",           # Operation type
    "children": ["child_id"],  # Child eclass IDs
    "eclass": "eclass_id",     # Which equivalence class this node belongs to
    "cost": 1                  # Node cost (from egglog)
  }
}
```

**Key concepts:**
- **EClass**: Equivalence class - a set of expressions that compute the same value
- **ENode**: A single expression (operator + children)
- Multiple ENodes can belong to the same EClass (they're equivalent)

---

## ILP Variables

### 1. EClass Activation Variables
```
A_<eclass_id> ∈ {0, 1}
```
- `A_<eclass_id> = 1` if this equivalence class is part of the extracted expression
- Example: `A_eclass_QK = 1` means the QK computation is needed

### 2. Node Selection Variables
```
N_<eclass_id>_<node_id> ∈ {0, 1}
```
- `N_<eclass_id>_<node_id> = 1` if this specific node is selected to represent its eclass
- Example: `N_eclass_max_node_R_max_m = 1` means we use global max (3-pass)
- Example: `N_eclass_max_node_R_max_m1 = 1` means we use hierarchical max (2-pass)

### 3. Operator Usage Variables
```
Op_<op_name> ∈ {0, 1}
```
- `Op_<op_name> = 1` if this operator type is used anywhere in the extraction
- Used for minimizing operator diversity (optional objective)
- Example: `Op_R_max_m = 1`, `Op_M_div_fp = 1`

### 4. Level Variables (for cycle prevention)
```
L_<eclass_id> ∈ [0, num_eclasses]
```
- Integer variable enforcing topological ordering
- Prevents selecting cyclic expressions

### 5. Opposite Variables (for cycle prevention)
```
Opp_<eclass_id>_<node_id> ∈ {0, 1}
```
- `Opp = 1 - N` (opposite of node selection)
- Used in level constraints

---

## Objective Function

### Option 1: Minimize Node Cost (Favor 2-pass or 3-pass)

```
Minimize: Σ (cost[node] × N_<eclass>_<node>)
```

**Cost Model for 2-pass (favor tiled operations):**
```python
COST_2PASS = {
    # 3-pass operations = EXPENSIVE
    'R_max_m': 100,      # Global max reduction
    'R_add_m': 100,      # Global sum reduction
    'M_div_mp': 100,     # In-loop division
    'M_sub_mp': 10,      # Non-tiled subtraction
    'M_exp_mp': 10,      # Non-tiled exp

    # 2-pass operations = CHEAP
    'R_max_m0': 1,       # Local max (per tile)
    'R_max_m1': 10,      # Cross-tile max
    'R_add_m0': 1,       # Local sum
    'R_add_m1': 10,      # Cross-tile sum
    'M_div_fp': 1,       # Post-loop division
    'M_exp_m1m0p': 1,    # Tiled exp
    'M_sub_m1m0p': 1,    # Tiled subtraction
    'M_exp_m1p': 1,      # Correction factor
    'T_split_m_m1m0': 1, # Tiling
}
```

**Cost Model for 3-pass (favor global operations):**
```python
COST_3PASS = {
    # 3-pass operations = CHEAP
    'R_max_m': 1,
    'R_add_m': 1,
    'M_div_mp': 1,
    'M_sub_mp': 1,
    'M_exp_mp': 1,

    # 2-pass operations = EXPENSIVE
    'R_max_m0': 100,
    'R_max_m1': 100,
    'R_add_m0': 100,
    'R_add_m1': 100,
    'M_div_fp': 100,
    'T_split_m_m1m0': 100,
}
```

### Option 2: Minimize Operator Types (from ISA compiler)

```
Minimize: Σ (weight[op] × Op_<op>)
```
- Minimizes the number of distinct operator types used
- Useful for ISA selection, less relevant for attention

---

## Constraints

### C1: One Node Per Activated EClass
```
∀ eclass: Σ N_<eclass>_<node> = A_<eclass>
```
- If an eclass is activated, exactly one node must be selected
- If not activated, no nodes selected

**Example:**
```
N_eclass_max_node_R_max_m + N_eclass_max_node_R_max_m1_R_max_m0 = A_eclass_max
```

### C2: Child Activation (Dependency)
```
∀ node with child eclass c: N_<eclass>_<node> ≤ A_<child_eclass>
```
- If a node is selected, all its child eclasses must be activated
- Propagates activation through the expression tree

**Example:**
```
N_eclass_sub_node_M_sub_mp ≤ A_eclass_QK      # subtraction needs QK
N_eclass_sub_node_M_sub_mp ≤ A_eclass_max     # subtraction needs max
```

### C3: Root Activation
```
∀ root eclass: A_<root_eclass> = 1
```
- The output eclass(es) must be activated
- For attention: the final `AV` output must be selected

**Example:**
```
A_eclass_AV = 1
```

### C4: Intersection Constraint
```
∀ eclass where ALL nodes share child c: A_<eclass> ≤ A_<child_c>
```
- If all nodes in an eclass depend on the same child, that child must be activated
- Optimization to reduce search space

### C5: Self-Loop Prevention
```
∀ node where eclass ∈ children: N_<eclass>_<node> = 0
```
- Nodes that reference their own eclass are invalid

### C6: Cycle Prevention (Level Ordering)
```
N + Opp = 1                                    # Opposite variable
L_child - L_parent + M × Opp ≥ 1               # Level ordering
```
- Ensures selected nodes form a DAG (no cycles)
- M = num_eclasses + 1 (big-M constant)

### C7: Operator Activation
```
∀ node: N_<eclass>_<node> ≤ Op_<node.op>
```
- If a node is selected, its operator must be marked as used
- Connects node selection to operator usage (for operator minimization objective)

---

## Example: Attention E-graph

Given 3-pass attention input, egglog creates equivalent 2-pass expressions:

```
EClass: eclass_max
  - node_R_max_m(eclass_QK)           # 3-pass: global max
  - node_R_max_m1(node_R_max_m0(...)) # 2-pass: local then global

EClass: eclass_output
  - node_R_add_m(M_mul_fmp(M_div_mp(...)))  # 3-pass
  - node_M_div_fp(R_add_m(...), R_add_m1(...))  # 2-pass
```

**With 2-pass cost model:**
- `N_eclass_max_node_R_max_m` has cost 100
- `N_eclass_max_node_R_max_m1` has cost 10
- ILP selects the hierarchical max (2-pass)

---

## Output

The ILP solver returns:
```python
{
    "eclass_id": "selected_node_id",
    ...
}
```

This mapping is used to:
1. Build a new JSON with only selected nodes
2. Generate Triton code from the extracted expression

---

## Implementation Plan

1. **Parse egglog JSON** → Build EClass/ENode objects
2. **Generate LP file** → Variables, objective, constraints
3. **Solve with Gurobi/CBC** → Get variable assignments
4. **Extract solution** → Map eclasses to selected nodes
5. **Output JSON** → Only the selected implementation

```python
# Usage
python attention_ilp_extractor.py \
    --input attention_egraph.json \
    --cost-model 2pass \
    --output attention_2pass.json
```
