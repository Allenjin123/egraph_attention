"""
Proper rewrite rules for Cascade 1 → Cascade 2 transformation.

Key insight: To introduce intermediate variables, we need let-bindings.

Pattern: Z = Y × (Σ A)
Rewrite: Z = let X = Σ A in Y × X
"""

from __future__ import annotations
from egglog import *


# ============================================================================
# Data types with let-binding support
# ============================================================================

class Tensor(Expr):
    def __init__(self, name: StringLike):
        ...


class TExpr(Expr):
    """Tensor Expression with let-binding support"""

    @classmethod
    def var(cls, t: Tensor) -> TExpr:
        """Variable reference"""
        ...

    @classmethod
    def mul(cls, a: TExpr, b: TExpr) -> TExpr:
        """Multiplication"""
        ...

    @classmethod
    def sum(cls, t: Tensor) -> TExpr:
        """Sum reduction"""
        ...

    @classmethod
    def let_bind(cls, var_name: StringLike, value: TExpr, body: TExpr) -> TExpr:
        """Let binding: let var_name = value in body"""
        ...

    @classmethod
    def var_ref(cls, name: StringLike) -> TExpr:
        """Reference to a let-bound variable"""
        ...


def main():
    egraph = EGraph()

    print("=" * 70)
    print("Proper Rewrite Rules: Cascade 1 → Cascade 2")
    print("=" * 70)

    # ========================================================================
    # Step 1: Define only Cascade 1
    # ========================================================================
    print("\n[Step 1] Defining ONLY Cascade 1...")

    Y = Tensor(String("Y"))
    A = Tensor(String("A"))

    # Cascade 1: Z = Y × (Σ A)
    # This is the ONLY thing we define manually
    z_cascade1 = TExpr.mul(
        TExpr.var(Y),
        TExpr.sum(A)
    )

    print("  Cascade 1: Z = Y × (Σ A)")
    print("  ✓ Defined")

    # ========================================================================
    # Step 2: Define rewrite rule
    # ========================================================================
    print("\n[Step 2] Defining rewrite rule...")

    @egraph.register
    def factor_reduction(y: TExpr, a: Tensor):
        """
        Rewrite rule: Factor out sum into a let-binding.

        Pattern: y × sum(a)
        Rewrite: let X = sum(a) in y × var_ref("X")

        This is the transformation from Cascade 1 to Cascade 2!
        """
        yield rewrite(
            TExpr.mul(y, TExpr.sum(a))
        ).to(
            TExpr.let_bind(
                String("X"),
                TExpr.sum(a),
                TExpr.mul(y, TExpr.var_ref(String("X")))
            )
        )

    print("  ✓ Defined rewrite rule: factor_reduction")
    print("     Pattern: y × sum(a)")
    print("     Rewrite: let X = sum(a) in y × X")

    # ========================================================================
    # Step 3: Register Cascade 1 and run
    # ========================================================================
    print("\n[Step 3] Registering Cascade 1 and running egraph...")

    egraph.register(z_cascade1)
    print("  ✓ Registered z_cascade1")

    print("  ⟳ Running egraph to apply rewrite rules...")
    egraph.run(10)
    print("  ✓ Egraph saturated")

    # ========================================================================
    # Step 4: Build expected Cascade 2 (for verification)
    # ========================================================================
    print("\n[Step 4] Building expected Cascade 2 (for verification)...")

    # Cascade 2: let X = Σ A in Y × X
    z_cascade2 = TExpr.let_bind(
        String("X"),
        TExpr.sum(A),
        TExpr.mul(
            TExpr.var(Y),
            TExpr.var_ref(String("X"))
        )
    )

    egraph.register(z_cascade2)
    print("  ✓ Built expected Cascade 2")

    # ========================================================================
    # Step 5: Check if rewrite was applied
    # ========================================================================
    print("\n[Step 5] Checking if rewrite was applied...")

    try:
        egraph.check(eq(z_cascade1).to(z_cascade2))
        print("  ✅ SUCCESS! Rewrite rule was applied!")
        print()
        print("  Starting with:")
        print("    Z = Y × (Σ A)")
        print()
        print("  Egglog discovered:")
        print("    Z = let X = Σ A in Y × X")
        print()
        print("  This is exactly Cascade 2!")
    except Exception as e:
        print(f"  ⚠️  Rewrite not applied: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
This demonstrates the CORRECT egglog workflow:

1. Define ONLY Cascade 1 (the naive version)
   Z = Y × (Σ A)

2. Define rewrite rule using rewrite().to()
   Pattern: y × sum(a)
   =>
   Rewrite: let X = sum(a) in y × X

3. Run egraph - it applies the rewrite automatically!

4. Extract optimized version (Cascade 2)

The let-binding representation allows us to:
  - Introduce intermediate variables in rewrites
  - Track dependencies
  - Convert to explicit statement sequences later

Next: Add more rewrite rules for full attention transformations!
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
