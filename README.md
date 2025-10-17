# Einsum Notation in Egglog Python for FuseMax

This project defines data types in egglog Python to represent einsum notation from the FuseMax paper, enabling flexible transformation and rewriting of attention computations.

## File Structure

### 1. `einsum_types.py`
**Basic building blocks for einsum expressions:**

- **Ranks**: `Rank(name, size)` - Represents dimensions like M, N, K, E, F, P
  - `RankVariable` - Variables that iterate over ranks (m, n, k, etc.)
  - `RankExpr` - Expressions on rank variables (filtering, arithmetic)

- **Tensors**: `Tensor(name, shape)` - Multi-dimensional arrays
  - `TensorShape` - List of ranks defining tensor shape
  - `TensorAccess` - Accessing a tensor at specific indices

- **Operators**:
  - `MergeOp` - Controls which iteration space points to visit (∩, ∪, ←, 1)
  - `ComputeOp` - Operations to perform (+, ×, max, exp, etc.)

### 2. `einsum_expr.py`
**Einsum expressions and cascades:**

- **Actions**:
  - `MapAction` - Map operations (∧) between tensors
  - `ReduceAction` - Reduce operations (∨) over a rank
  - `UnaryOp` - Unary operations like sigmoid, exp

- **Expressions**:
  - `EinsumExpr` - Right-hand side of einsum (tensor access, binary ops, reductions)
  - `Einsum` - Complete einsum statement (LHS = RHS)

- **Cascades**:
  - `Cascade` - Sequence of dependent Einsums
  - `IterativeCascade` - Iterative einsums with initialization and stopping conditions

### 3. `attention_example.py`
**Example usage showing how to represent the 3-pass attention cascade**

## Key Concepts from FuseMax Paper

### Extended Einsums (EDGE Notation)

Traditional einsum: `Zm,n = Ak,m × Bk,n`

Extended einsum with explicit operations: `Zm,n = Ak,m · Bk,n :: ∧k ×(∩) ∨k +(∪)`

Where:
- `∧k ×(∩)` = Map action on rank k with multiply compute and intersection merge
- `∨k +(∪)` = Reduce action on rank k with add compute and union merge

### Attention as Cascades

**3-Pass Cascade** (Cascade 4 in paper):
```
QKm,p = Qe,p × Ke,m        /* Pass 1 */
GMp = QKm,p :: ∨m max(∪)
SNm,p = e^(QKm,p - GMp)     /* Pass 2 */
SDp = SNm,p
Am,p = SNm,p / SDp          /* Pass 3 */
AVf,p = Am,p × Vf,m
```

**1-Pass Cascade** (Cascade 5 in paper):
- Uses iterative computation with running max/denominator
- More complex but requires only one pass through data
- Better memory traffic characteristics

## Usage

### Creating a Simple Einsum (GEMM)

```python
from einsum_types import *
from einsum_expr import *

# Define ranks
M = Rank("M", 1024)
N = Rank("N", 1024)
K = Rank("K", 64)

# Define tensors
A = Tensor("A", TensorShape.empty().cons(K).cons(M))
B = Tensor("B", TensorShape.empty().cons(K).cons(N))
Z = Tensor("Z", TensorShape.empty().cons(M).cons(N))

# Create Einsum: Zm,n = Ak,m × Bk,n
# (Implementation in attention_example.py)
```

### Creating Attention Transformations

The goal is to represent different attention algorithms (3-pass, 2-pass, 1-pass) as cascades, then define rewrite rules to transform between them:

```python
# Define rewrite rule to go from 3-pass to 1-pass
@egraph.register
def transform_3pass_to_1pass(cascade_3pass: Cascade) -> Cascade:
    # Apply reassociation transformations
    # Return 1-pass cascade
    pass
```

## Next Steps

1. **Add rewrite rules**: Define transformations between different attention cascades
2. **Represent Flash Attention**: Use the 1-pass cascade representation
3. **Add cost models**: Analyze number of passes, memory traffic
4. **Implement extraction**: Extract optimized cascade after rewrites

## Design Flexibility

The data types are designed to support all transformations in the paper:
- ✅ Traditional einsums (matrix multiply, etc.)
- ✅ Extended einsums with custom operations (max, exp, etc.)
- ✅ Filtering (rank expressions like k≤i)
- ✅ Iterative computation (running sums, running max)
- ✅ Cascades of dependent einsums
- ✅ Multi-pass analysis

## References

- FuseMax paper (fusemax.pdf in this repo)
- Egglog Python documentation (example.txt in this repo)
- EDGE: Extended General Einsums notation
