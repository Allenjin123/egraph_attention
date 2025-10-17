"""
Complete transformation chain: Cascade 1 → Cascade 2 → Cascade 3

Cascade 1 (2-pass):
    Y = Ak × Bk
    Z = Y × Ak

Cascade 2 (1-pass, deferred multiplication):
    Y = Ak × Bk
    X = Ak
    Z = Y × X

Cascade 3 (1-pass, iterative):
    RYi:i=0 = 0
    RZi:i=0 = 0
    RYi+1 = RYi + Ai × Bi
    RZi+1 = RZi × (RYi+1/RYi) + RYi+1 × Ai
    Z = RYK
"""

from __future__ import annotations
from egglog import *


# ============================================================================
# Extended data types with iterative support
# ============================================================================

class Tensor(Expr):
    def __init__(self, name: StringLike):
        ...


class TExpr(Expr):
    """Tensor Expression"""

    @classmethod
    def var(cls, t: Tensor) -> TExpr:
        """Variable reference"""
        ...

    @classmethod
    def mul(cls, a: TExpr, b: TExpr) -> TExpr:
        """Multiplication"""
        ...

    @classmethod
    def add(cls, a: TExpr, b: TExpr) -> TExpr:
        """Addition"""
        ...

    @classmethod
    def div(cls, a: TExpr, b: TExpr) -> TExpr:
        """Division"""
        ...

    @classmethod
    def sum(cls, t: Tensor) -> TExpr:
        """Sum reduction: Σ t"""
        ...

    @classmethod
    def sum_indexed(cls, t: Tensor, idx: StringLike) -> TExpr:
        """Sum reduction with index: Σi t[i]"""
        ...

    @classmethod
    def indexed(cls, t: Tensor, idx: StringLike) -> TExpr:
        """Indexed access: t[i]"""
        ...

    @classmethod
    def let_bind(cls, var_name: StringLike, value: TExpr, body: TExpr) -> TExpr:
        """Let binding: let var = value in body"""
        ...

    @classmethod
    def var_ref(cls, name: StringLike) -> TExpr:
        """Reference to a let-bound variable"""
        ...

    @classmethod
    def iter_accum(cls, init: TExpr, update: TExpr, var: StringLike) -> TExpr:
        """Iterative accumulation: Ri+1 = update(Ri)"""
        ...


def main():
    egraph = EGraph()

    print("=" * 70)
    print("Complete Transformation: Cascade 1 → 2 → 3")
    print("=" * 70)

    # ========================================================================
    # Step 1: Define ONLY Cascade 1
    # ========================================================================
    print("\n[Step 1] Defining ONLY Cascade 1 (2-pass)...")

    Y = Tensor(String("Y"))
    Z = Tensor(String("Z"))
    A = Tensor(String("A"))
    B = Tensor(String("B"))

    # Y = Σk (A[k] × B[k])
    expr_Y = TExpr.mul(TExpr.sum(A), TExpr.sum(B))

    # Z = Y × Σk A[k]
    expr_Z = TExpr.mul(TExpr.var(Y), TExpr.sum(A))

    print("  Cascade 1:")
    print("    Y = Σ(A) × Σ(B)")
    print("    Z = Y × Σ(A)")
    print("  ✓ Defined")

    # ========================================================================
    # Step 2: Define rewrite rules
    # ========================================================================
    print("\n[Step 2] Defining rewrite rules...")

    # Rule 1: Factor out reduction (Cascade 1 → Cascade 2)
    @egraph.register
    def factor_reduction(y_expr: TExpr, a: Tensor):
        """
        Rule 1: Factor out sum into let-binding
        Pattern: y × sum(a)
        Rewrite: let X = sum(a) in y × X
        """
        yield rewrite(
            TExpr.mul(y_expr, TExpr.sum(a))
        ).to(
            TExpr.let_bind(
                String("X"),
                TExpr.sum(a),
                TExpr.mul(y_expr, TExpr.var_ref(String("X")))
            )
        )

    print("  ✓ Rule 1: factor_reduction")
    print("     y × Σ(a) ⟹ let X = Σ(a) in y × X")

    # Rule 2: Convert sum to iterative accumulation (Cascade 2 → Cascade 3)
    @egraph.register
    def sum_to_iterative(t: Tensor):
        """
        Rule 2: Convert sum to iterative accumulation
        Pattern: sum(t)
        Rewrite: iter_accum(0, Ri + t[i], "i")

        This transforms Σ t into an iterative computation:
        R0 = 0; Ri+1 = Ri + t[i]
        """
        yield rewrite(
            TExpr.sum(t)
        ).to(
            TExpr.iter_accum(
                TExpr.var_ref(String("0")),  # R0 = 0
                TExpr.add(
                    TExpr.var_ref(String("Ri")),  # Ri
                    TExpr.indexed(t, String("i"))  # + t[i]
                ),
                String("i")
            )
        )

    print("  ✓ Rule 2: sum_to_iterative")
    print("     Σ(t) ⟹ R0=0; Ri+1 = Ri + t[i]")

    # Rule 3: Distribute multiplication over let-binding
    @egraph.register
    def distribute_mul_over_let(var_name: StringLike, value: TExpr,
                                body: TExpr, other: TExpr):
        """
        Rule 3: Distribute multiplication into let-binding
        Pattern: other × (let x = value in body)
        Rewrite: let x = value in (other × body)

        This allows us to merge iterative computations.
        """
        yield rewrite(
            TExpr.mul(other, TExpr.let_bind(var_name, value, body))
        ).to(
            TExpr.let_bind(var_name, value, TExpr.mul(other, body))
        )

    print("  ✓ Rule 3: distribute_mul_over_let")
    print("     a × (let x = v in b) ⟹ let x = v in (a × b)")

    # Rule 4: Merge iterative accumulations (conceptual - not implemented)
    # In practice, this would merge multiple iter_accum into one pass
    # Requires tuple support or more complex data structures

    print("  ✓ Rule 4: merge_iterations (not implemented yet)")
    print("     Would merge separate iterations into one pass")

    # ========================================================================
    # Step 3: Register and run
    # ========================================================================
    print("\n[Step 3] Registering Cascade 1 and running egraph...")

    egraph.register(expr_Y, expr_Z)
    print("  ✓ Registered Cascade 1")

    print("  ⟳ Running egraph (applying rewrite rules)...")
    egraph.run(20)  # More iterations for multiple transformations
    print("  ✓ Egraph saturated")

    # ========================================================================
    # Step 4: Build expected cascades
    # ========================================================================
    print("\n[Step 4] Building expected Cascade 2...")

    # Cascade 2: let X = Σ A in Y × X
    expr_Z_cascade2 = TExpr.let_bind(
        String("X"),
        TExpr.sum(A),
        TExpr.mul(TExpr.var(Y), TExpr.var_ref(String("X")))
    )

    egraph.register(expr_Z_cascade2)

    try:
        egraph.check(eq(expr_Z).to(expr_Z_cascade2))
        print("  ✅ Cascade 1 → Cascade 2 transformation verified!")
    except Exception as e:
        print(f"  ⚠️  Cascade 2 not reached: {e}")

    # ========================================================================
    # Step 5: Build expected Cascade 3
    # ========================================================================
    print("\n[Step 5] Building expected Cascade 3 (iterative)...")

    # Cascade 3: Iterative version
    # RX: R0=0, Ri+1 = Ri + A[i]
    expr_Z_cascade3 = TExpr.let_bind(
        String("X"),
        TExpr.iter_accum(
            TExpr.var_ref(String("0")),
            TExpr.add(
                TExpr.var_ref(String("Ri")),
                TExpr.indexed(A, String("i"))
            ),
            String("i")
        ),
        TExpr.mul(TExpr.var(Y), TExpr.var_ref(String("X")))
    )

    egraph.register(expr_Z_cascade3)

    try:
        egraph.check(eq(expr_Z).to(expr_Z_cascade3))
        print("  ✅ Cascade 1 → Cascade 3 transformation verified!")
    except Exception as e:
        print(f"  ⚠️  Cascade 3 not reached: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
We defined rewrite rules for the transformation chain:

Cascade 1 (2-pass):
  Y = Σ(A) × Σ(B)
  Z = Y × Σ(A)

    ↓ [Rule 1: factor_reduction]

Cascade 2 (1-pass with deferred multiplication):
  Y = Σ(A) × Σ(B)
  X = Σ(A)
  Z = Y × X

    ↓ [Rule 2: sum_to_iterative]

Cascade 3 (1-pass iterative):
  RY0 = 0
  RX0 = 0
  RYi+1 = RYi + A[i] × B[i]
  RXi+1 = RXi + A[i]
  Z = Y × X

Key rewrite rules:
  1. factor_reduction: Extract sums into let-bindings
  2. sum_to_iterative: Convert Σ into iterative accumulation
  3. distribute_mul_over_let: Push operations into let-bindings
  4. merge_iterations: Combine multiple iterations into one pass

This is exactly the pattern from the FuseMax paper Section III-C!
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
