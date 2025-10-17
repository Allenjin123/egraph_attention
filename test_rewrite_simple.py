"""
Simplified test: Show that egglog can discover equivalent expressions.

Instead of rewriting cascades, we show that:
  Y × (Σk A[k])  is structurally different from  Y × X

But with the knowledge that X = Σk A[k], egglog can understand they're equivalent.
"""

from __future__ import annotations
from egglog import *


# ============================================================================
# Minimal data types for testing
# ============================================================================

class Tensor(Expr):
    """A tensor"""
    def __init__(self, name: StringLike):
        ...

class TExpr(Expr):
    """Tensor Expression"""

    @classmethod
    def var(cls, t: Tensor) -> TExpr:
        """Tensor variable"""
        ...

    @classmethod
    def mul(cls, a: TExpr, b: TExpr) -> TExpr:
        """Multiply two expressions"""
        ...

    @classmethod
    def sum(cls, t: Tensor) -> TExpr:
        """Sum reduction"""
        ...


def main():
    egraph = EGraph()

    print("=" * 70)
    print("Test: Expression Equivalence via Rewrite Rules")
    print("=" * 70)

    # ========================================================================
    # Define tensors
    # ========================================================================
    print("\n[Step 1] Defining tensors...")

    Y = Tensor(String("Y"))
    A = Tensor(String("A"))
    X = Tensor(String("X"))

    print("  ✓ Defined tensors Y, A, X")

    # ========================================================================
    # Define expressions
    # ========================================================================
    print("\n[Step 2] Building expressions...")

    # Expression 1: Y × (sum of A) - the INLINE version
    expr_inline = TExpr.mul(
        TExpr.var(Y),
        TExpr.sum(A)
    )
    print("  ✓ Built expr_inline: Y × (Σ A)")

    # Expression 2: Y × X where X = (sum of A) - the FACTORED version
    expr_factored = TExpr.mul(
        TExpr.var(Y),
        TExpr.var(X)
    )
    print("  ✓ Built expr_factored: Y × X")

    # Expression 3: X = sum of A (the definition)
    expr_x_def = TExpr.sum(A)
    print("  ✓ Built expr_x_def: X = Σ A")

    # ========================================================================
    # Define rewrite rules
    # ========================================================================
    print("\n[Step 3] Defining rewrite rules...")

    # Rule: Substitution - if X = something, you can substitute it
    # This shows that (Y × X) where X = ΣA is equivalent to Y × (ΣA)

    @egraph.register
    def substitution_rule(y: TExpr, x_val: TExpr, x_var: Tensor):
        """
        If we know X = x_val, then (Y × X) can be rewritten as (Y × x_val)
        This is just variable substitution.
        """
        # Pattern: mul(y, var(x_var))
        # Can be rewritten to: mul(y, x_val)
        # IF we separately assert that var(x_var) = x_val

        yield rewrite(
            TExpr.mul(y, TExpr.var(x_var))
        ).to(
            TExpr.mul(y, x_val)
        )

    print("  ✓ Defined substitution rule")

    # ========================================================================
    # Register expressions
    # ========================================================================
    print("\n[Step 4] Registering expressions...")

    egraph.register(expr_inline)
    print("  ✓ Registered expr_inline")

    egraph.register(expr_factored)
    print("  ✓ Registered expr_factored")

    # Key assertion: X = sum(A)
    # This tells egglog that var(X) and sum(A) are equivalent
    egraph.register(union(TExpr.var(X)).with_(expr_x_def))
    print("  ✓ Asserted: X = Σ A")

    # ========================================================================
    # Run egraph
    # ========================================================================
    print("\n[Step 5] Running egraph...")

    result = egraph.run(10)
    print(f"  ✓ Egraph ran for {result.iterations} iterations")

    # ========================================================================
    # Check equivalence
    # ========================================================================
    print("\n[Step 6] Checking if expressions are equivalent...")

    try:
        # Check if the inline and factored versions are equivalent
        egraph.check(eq(expr_inline).to(expr_factored))
        print("  ✅ SUCCESS! The expressions are equivalent!")
        print("     Y × (Σ A)  ≡  Y × X  (given X = Σ A)")
        print()
        print("  This shows that egglog can discover that factoring")
        print("  out the sum into a separate variable is valid!")
    except Exception as e:
        print(f"  ⚠️  Not equivalent: {e}")
        print()
        print("  This means we need to strengthen our rewrite rules")
        print("  to allow egglog to discover this equivalence.")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
This test demonstrates the core idea:

1. We define two structurally different expressions:
   - expr_inline:   Y × (Σ A)
   - expr_factored: Y × X

2. We tell egglog that X = Σ A

3. We define substitution rules

4. Egglog should discover they're equivalent!

If this works, we can scale it up to full cascades where:
  - Cascade 1: Uses inline reductions
  - Cascade 2: Factors out reductions to intermediate variables
  - Egglog discovers they compute the same thing!
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
