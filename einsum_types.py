"""
Basic data types for representing Einsum notation in egglog Python.
Based on the FuseMax paper's Extended Einsums (EDGE) notation.
"""

from __future__ import annotations
from egglog import *
from typing import ClassVar


# ============================================================================
# Rank/Dimension Types
# ============================================================================

class Rank(Expr):
    """
    Represents a rank/dimension in a tensor.
    Examples: M, N, K, E, F, P from the paper
    """

    @method(egg_fn="Rank")
    def __init__(self, name: StringLike, size: i64Like):
        """Create a rank with a name and size"""
        ...

    def get_name(self) -> String:
        """Get the name of this rank"""
        ...

    def get_size(self) -> i64:
        """Get the size/shape of this rank"""
        ...


class RankVariable(Expr):
    """
    Represents a variable that iterates over a rank.
    Examples: m, n, k iterating over M, N, K
    """

    @method(egg_fn="RankVariable")
    def __init__(self, name: StringLike, rank: Rank):
        """Create a rank variable that iterates over a rank"""
        ...


class RankExpr(Expr):
    """
    Represents expressions on rank variables.
    Supports filtering like k≤i, k:k≤i, or arithmetic like m1×M0+m0
    """

    @method(egg_fn="RankVar")
    @classmethod
    def var(cls, var: RankVariable) -> RankExpr:
        """A simple rank variable"""
        ...

    @method(egg_fn="RankFilter")
    @classmethod
    def filter(cls, var: RankVariable, condition: StringLike) -> RankExpr:
        """A filtered rank variable like k:k≤i"""
        ...

    @method(egg_fn="RankArith")
    @classmethod
    def arith(cls, expr: StringLike) -> RankExpr:
        """An arithmetic expression like m1×M0+m0"""
        ...


# ============================================================================
# Tensor Types
# ============================================================================

class TensorShape(Expr):
    """
    Represents the shape of a tensor as a list of ranks.
    """

    @method(egg_fn="EmptyShape")
    @classmethod
    def empty(cls) -> TensorShape:
        """Empty shape (for scalars)"""
        ...

    @method(egg_fn="ConsShape")
    def cons(self, rank: Rank) -> TensorShape:
        """Add a rank to the shape"""
        ...


class Tensor(Expr):
    """
    Represents a tensor with a name and indexed by rank variables.
    Examples: Zm,n, Ak,m, Qe,p
    """

    @method(egg_fn="Tensor")
    def __init__(self, name: StringLike, shape: TensorShape):
        """Create a tensor with a name and shape"""
        ...

    def get_name(self) -> String:
        """Get the name of this tensor"""
        ...

    def get_shape(self) -> TensorShape:
        """Get the shape of this tensor"""
        ...


class TensorAccess(Expr):
    """
    Represents accessing a tensor at specific rank indices.
    Examples: Am,p, Qe,p, BQKm1,m0,p
    """

    @method(egg_fn="TensorAccess")
    def __init__(self, tensor: Tensor):
        """Create a tensor access"""
        ...

    def add_index(self, rank_expr: RankExpr) -> TensorAccess:
        """Add an index to the tensor access"""
        ...


# ============================================================================
# Merge and Compute Operators
# ============================================================================

class MergeOp(Expr):
    """
    Merge operators for map and reduce actions.
    """

    INTERSECTION: ClassVar[MergeOp]  # ∩ - intersection
    UNION: ClassVar[MergeOp]          # ∪ - union
    LEFT: ClassVar[MergeOp]           # ← - left only
    PASSTHROUGH: ClassVar[MergeOp]   # 1 - pass-through (all points)


class ComputeOp(Expr):
    """
    Compute operators for map and reduce actions.
    """

    # Basic arithmetic
    ADD: ClassVar[ComputeOp]       # +
    MULTIPLY: ClassVar[ComputeOp]  # ×
    SUBTRACT: ClassVar[ComputeOp]  # -
    DIVIDE: ClassVar[ComputeOp]    # ÷

    # Special operations
    MAX: ClassVar[ComputeOp]       # max
    EXP: ClassVar[ComputeOp]       # e^x
    SUB_THEN_EXP: ClassVar[ComputeOp]  # e^(a-b)

    @method(egg_fn="CustomCompute")
    @classmethod
    def custom(cls, name: StringLike) -> ComputeOp:
        """User-defined compute operation"""
        ...


# Initialize constants
INTERSECTION = constant("MergeIntersection", MergeOp)
UNION = constant("MergeUnion", MergeOp)
LEFT = constant("MergeLeft", MergeOp)
PASSTHROUGH = constant("MergePassthrough", MergeOp)

MergeOp.INTERSECTION = INTERSECTION
MergeOp.UNION = UNION
MergeOp.LEFT = LEFT
MergeOp.PASSTHROUGH = PASSTHROUGH

ADD = constant("ComputeAdd", ComputeOp)
MULTIPLY = constant("ComputeMultiply", ComputeOp)
SUBTRACT = constant("ComputeSubtract", ComputeOp)
DIVIDE = constant("ComputeDivide", ComputeOp)
MAX = constant("ComputeMax", ComputeOp)
EXP = constant("ComputeExp", ComputeOp)
SUB_THEN_EXP = constant("ComputeSubThenExp", ComputeOp)

ComputeOp.ADD = ADD
ComputeOp.MULTIPLY = MULTIPLY
ComputeOp.SUBTRACT = SUBTRACT
ComputeOp.DIVIDE = DIVIDE
ComputeOp.MAX = MAX
ComputeOp.EXP = EXP
ComputeOp.SUB_THEN_EXP = SUB_THEN_EXP
