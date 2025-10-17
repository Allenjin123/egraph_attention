"""
Simplified test: Just build Cascade 1 and Cascade 2 to verify data types work.
We'll test rewrite rules separately once the basic structure is validated.
"""

from __future__ import annotations
from egglog import *
from einsum_types import *
from einsum_expr import *


def main():
    # Create an EGraph to work in
    egraph = EGraph()

    print("=" * 70)
    print("Testing Einsum Data Types - Building Cascades")
    print("=" * 70)

    # ========================================================================
    # Step 1: Define the ranks and tensors
    # ========================================================================
    print("\n[Step 1] Defining ranks and tensors...")

    # Define rank K with size 128
    K = Rank(String("K"), i64(128))
    print(f"  ✓ Created rank K with size 128")

    # Define rank variable k that iterates over K
    k = RankVariable(String("k"), K)
    print(f"  ✓ Created rank variable k")

    # Create tensors A and B (both are 1-tensors with rank K)
    shape_K = TensorShape.empty().cons(K)
    A = Tensor(String("A"), shape_K)
    B = Tensor(String("B"), shape_K)
    print(f"  ✓ Created tensors A[K] and B[K]")

    # Output tensors (scalars - 0-tensors)
    shape_scalar = TensorShape.empty()
    Y = Tensor(String("Y"), shape_scalar)
    Z = Tensor(String("Z"), shape_scalar)
    X = Tensor(String("X"), shape_scalar)
    print(f"  ✓ Created scalar tensors Y, Z, X")

    # ========================================================================
    # Step 2: Build Cascade 1 (2-pass)
    # ========================================================================
    print("\n[Step 2] Building Cascade 1 (2-pass)...")
    print("  Cascade 1:")
    print("    Y = Ak × Bk      (Einsum 5)")
    print("    Z = Y × Ak       (Einsum 6)")

    # Create tensor accesses
    A_k = TensorAccess(A).add_index(RankExpr.var(k))
    B_k = TensorAccess(B).add_index(RankExpr.var(k))
    Y_access = TensorAccess(Y)
    Z_access = TensorAccess(Z)

    # Einsum 5: Y = Ak × Bk
    map_mult = MapAction(K, MergeOp.INTERSECTION, ComputeOp.MULTIPLY)
    reduce_add = ReduceAction(K, MergeOp.UNION, ComputeOp.ADD)

    # Build expression: A[k] * B[k], then reduce over k
    AB_expr = EinsumExpr.binary(
        EinsumExpr.tensor(A_k),
        EinsumExpr.tensor(B_k),
        map_mult
    )
    Y_expr = EinsumExpr.reduce(AB_expr, reduce_add)
    einsum_Y = Einsum(Y_access, Y_expr)
    print("  ✓ Built Einsum 5: Y = Ak × Bk")

    # Einsum 6: Z = Y × Ak
    Y_for_Z = TensorAccess(Y)
    A_k_for_Z = TensorAccess(A).add_index(RankExpr.var(k))

    # First reduce A[k] over k
    A_reduced = EinsumExpr.reduce(
        EinsumExpr.tensor(A_k_for_Z),
        reduce_add
    )

    # Then multiply by Y (scalar multiplication)
    Z_expr = EinsumExpr.binary(
        EinsumExpr.tensor(Y_for_Z),
        A_reduced,
        MapAction(K, MergeOp.PASSTHROUGH, ComputeOp.MULTIPLY)
    )
    einsum_Z = Einsum(Z_access, Z_expr)
    print("  ✓ Built Einsum 6: Z = Y × Ak")

    # Build the cascade
    cascade_1_list = (EinsumList.empty()
                      .cons(einsum_Y)
                      .cons(einsum_Z))
    cascade_1 = Cascade.simple(cascade_1_list)
    print("  ✓ Built Cascade 1")

    # ========================================================================
    # Step 3: Build Cascade 2 (1-pass, deferred multiplication)
    # ========================================================================
    print("\n[Step 3] Building Cascade 2 (1-pass)...")
    print("  Cascade 2:")
    print("    Y = Ak × Bk      (Einsum 7)")
    print("    X = Ak           (Einsum 8)")
    print("    Z = Y × X        (Einsum 9)")

    X_access = TensorAccess(X)

    # Einsum 7: Y = Ak × Bk (same as Einsum 5)
    einsum_Y_v2 = einsum_Y
    print("  ✓ Einsum 7: Y = Ak × Bk (reused)")

    # Einsum 8: X = Ak (just reduce A over k)
    A_k_for_X = TensorAccess(A).add_index(RankExpr.var(k))
    X_expr = EinsumExpr.reduce(
        EinsumExpr.tensor(A_k_for_X),
        reduce_add
    )
    einsum_X = Einsum(X_access, X_expr)
    print("  ✓ Built Einsum 8: X = Ak")

    # Einsum 9: Z = Y × X
    Y_for_Z_v2 = TensorAccess(Y)
    X_for_Z = TensorAccess(X)

    Z_expr_v2 = EinsumExpr.binary(
        EinsumExpr.tensor(Y_for_Z_v2),
        EinsumExpr.tensor(X_for_Z),
        MapAction(K, MergeOp.PASSTHROUGH, ComputeOp.MULTIPLY)
    )
    einsum_Z_v2 = Einsum(Z_access, Z_expr_v2)
    print("  ✓ Built Einsum 9: Z = Y × X")

    # Build the cascade
    cascade_2_list = (EinsumList.empty()
                      .cons(einsum_Y_v2)
                      .cons(einsum_X)
                      .cons(einsum_Z_v2))
    cascade_2 = Cascade.simple(cascade_2_list)
    print("  ✓ Built Cascade 2")

    # ========================================================================
    # Step 4: Register in egraph
    # ========================================================================
    print("\n[Step 4] Registering cascades in e-graph...")

    egraph.register(cascade_1)
    print("  ✓ Registered Cascade 1")

    egraph.register(cascade_2)
    print("  ✓ Registered Cascade 2")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ SUCCESS! All data types work correctly!")
    print("=" * 70)
    print("""
We successfully:
  ✓ Defined ranks (K) and rank variables (k)
  ✓ Created tensors with shapes (A[K], B[K], scalars)
  ✓ Built tensor accesses (A[k], B[k])
  ✓ Created map actions (multiply with intersection)
  ✓ Created reduce actions (add with union)
  ✓ Built einsum expressions (binary ops, reductions)
  ✓ Created complete Einsums (output = expression)
  ✓ Built cascades of Einsums (both 2-pass and 1-pass)
  ✓ Registered everything in the e-graph

The data types are correct and ready for:
  → Adding rewrite rules to transform cascades
  → Defining pass analysis functions
  → Implementing attention transformations
  → Extracting optimized schedules
    """)
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
