"""
Correct egglog approach: Assert equalities and let congruence do the work.

The key insight: Don't try to "rewrite" with new variables.
Instead, assert what we know and let egglog propagate equivalences.
"""

from __future__ import annotations
from egglog import *


# ============================================================================
# Minimal data types
# ============================================================================

class Tensor(Expr):
    def __init__(self, name: StringLike):
        ...

class TExpr(Expr):
    @classmethod
    def var(cls, t: Tensor) -> TExpr:
        ...

    @classmethod
    def mul(cls, a: TExpr, b: TExpr) -> TExpr:
        ...

    @classmethod
    def sum(cls, t: Tensor) -> TExpr:
        ...


def main():
    egraph = EGraph()

    print("=" * 70)
    print("Test: Cascade 1 → Cascade 2 via Equality Assertions")
    print("=" * 70)

    # ========================================================================
    # Step 1: Define tensors
    # ========================================================================
    print("\n[Step 1] Defining tensors...")

    Y = Tensor(String("Y"))
    A = Tensor(String("A"))
    X = Tensor(String("X"))

    print("  ✓ Y, A, X")

    # ========================================================================
    # Step 2: Define Cascade 1 expressions (inline)
    # ========================================================================
    print("\n[Step 2] Defining Cascade 1 (inline sum)...")

    # Z = Y × (sum of A) - this is the 2-pass version
    # Because we compute Y, then we compute sum(A), then multiply
    z_cascade1 = TExpr.mul(
        TExpr.var(Y),
        TExpr.sum(A)
    )

    print("  Cascade 1: Z = Y × (Σ A)")
    print("  ✓ Built z_cascade1")

    # ========================================================================
    # Step 3: Define Cascade 2 expressions (factored)
    # ========================================================================
    print("\n[Step 3] Defining Cascade 2 (factored)...")

    # First compute X = sum(A)
    x_def = TExpr.sum(A)

    # Then Z = Y × X
    z_cascade2 = TExpr.mul(
        TExpr.var(Y),
        TExpr.var(X)
    )

    print("  Cascade 2:")
    print("    X = Σ A")
    print("    Z = Y × X")
    print("  ✓ Built x_def and z_cascade2")

    # ========================================================================
    # Step 4: Assert the key equality
    # ========================================================================
    print("\n[Step 4] Asserting equalities...")

    # Register all expressions
    egraph.register(z_cascade1, z_cascade2, x_def)

    # KEY ASSERTION: Tell egglog that var(X) equals sum(A)
    # This is like saying "let X = sum(A)"
    egraph.register(union(TExpr.var(X)).with_(x_def))
    print("  ✓ Asserted: var(X) = sum(A)")

    # ========================================================================
    # Step 5: Run egraph (congruence closure)
    # ========================================================================
    print("\n[Step 5] Running egraph...")

    result = egraph.run(5)
    print(f"  ✓ Egraph saturated")

    # ========================================================================
    # Step 6: Check if Cascade 1 and Cascade 2 are equivalent
    # ========================================================================
    print("\n[Step 6] Checking equivalence...")

    print("\n  Checking: z_cascade1 ≡ z_cascade2")
    print("    z_cascade1 = Y × (Σ A)")
    print("    z_cascade2 = Y × X     (where X = Σ A)")

    try:
        egraph.check(eq(z_cascade1).to(z_cascade2))
        print("\n  ✅ SUCCESS! Cascades are equivalent!")
        print("\n  Egglog discovered that:")
        print("    Y × (Σ A)  ≡  Y × X  (when X = Σ A)")
        print("\n  This means Cascade 2 is a valid transformation of Cascade 1!")
        success = True
    except Exception as e:
        print(f"\n  ⚠️  Not equivalent: {e}")
        print("\n  This suggests we need congruence rules for TExpr.mul")
        success = False

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if success:
        print("""
✅ SUCCESS! The approach works!

What we did:
  1. Defined Cascade 1: Z = Y × (Σ A)      [2-pass: compute Σ twice]
  2. Defined Cascade 2: X = Σ A; Z = Y × X  [1-pass: compute Σ once]
  3. Asserted: var(X) = sum(A)
  4. Egglog used congruence closure to discover equivalence!

This validates that:
  → Our data types work
  → Egglog can discover cascade transformations
  → We can scale this to full attention cascades!

Next steps:
  → Add more algebraic rules (associativity, distributivity)
  → Scale to full Cascade 1 → Cascade 2 from paper
  → Then tackle 3-pass → 1-pass attention!
        """)
    else:
        print("""
The equivalence wasn't discovered automatically.

This might be because:
  → Egglog needs explicit congruence rules
  → We may need to add rewrite rules for algebraic properties
  → Or we need to structure the data types differently

But the data types themselves work correctly!
We just need to tune the equality propagation.
        """)

    print("=" * 70)


if __name__ == "__main__":
    main()
