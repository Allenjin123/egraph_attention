"""
Dimension Resolver for Graph-Driven Code Generation

Computes broadcast positions and reduce axes based on DimSpec.
This enables correct dimension-aware code generation from the parsed
computation graph.

Example:
    For M_mul_emp: Q[e,p] * K[e,m] -> [e,m,p]
    - Q needs unsqueeze at position 1 (for 'm')
    - K needs unsqueeze at position 2 (for 'p')
"""

from typing import List, Tuple, Dict, Optional
from egg_ops import DimSpec


class DimResolver:
    """
    Resolves dimension transformations for broadcasting and reduction.

    Uses a canonical dimension ordering to determine:
    1. Where to insert unsqueeze operations for broadcasting
    2. Which axis to reduce over for reduction operations
    """

    # Canonical ordering of all dimensions (most general to most specific)
    # This ordering determines the position of dimensions in output tensors
    CANONICAL_ORDER = ('f', 'e', 'm1', 'm0', 'm', 'p')

    def __init__(self):
        # Build position lookup for efficient queries
        self._dim_positions = {dim: i for i, dim in enumerate(self.CANONICAL_ORDER)}

    def compute_broadcast_positions(
        self,
        input_dims: DimSpec,
        output_dims: DimSpec
    ) -> List[int]:
        """
        Compute positions where unsqueeze operations are needed.

        Args:
            input_dims: Dimensions of the input tensor
            output_dims: Dimensions of the output (broadcast result)

        Returns:
            List of positions (in output order) where input needs unsqueeze

        Example:
            input_dims = ('e', 'p'), output_dims = ('e', 'm', 'p')
            returns [1]  # unsqueeze at position 1 to add 'm'
        """
        input_set = set(input_dims.dims)
        missing_dims = []

        for i, dim in enumerate(output_dims.dims):
            if dim not in input_set:
                missing_dims.append(i)

        return missing_dims

    def compute_unsqueeze_sequence(
        self,
        input_dims: DimSpec,
        output_dims: DimSpec
    ) -> List[int]:
        """
        Compute the sequence of unsqueeze operations needed.

        Unlike compute_broadcast_positions, this returns positions
        for sequential unsqueeze calls (where each unsqueeze shifts
        subsequent positions).

        Args:
            input_dims: Dimensions of the input tensor
            output_dims: Dimensions of the output (broadcast result)

        Returns:
            List of unsqueeze positions to apply sequentially

        Example:
            input_dims = ('e', 'p'), output_dims = ('e', 'm', 'p')
            returns [1]  # tensor.unsqueeze(1) adds dimension at position 1
        """
        input_list = list(input_dims.dims)
        output_list = list(output_dims.dims)

        unsqueeze_positions = []
        current_dims = input_list.copy()

        # For each dimension in output, check if we need to insert it
        for i, out_dim in enumerate(output_list):
            if i >= len(current_dims) or current_dims[i] != out_dim:
                # Need to insert this dimension
                if out_dim not in current_dims:
                    unsqueeze_positions.append(i)
                    current_dims.insert(i, out_dim)

        return unsqueeze_positions

    def compute_reduce_axis(
        self,
        input_dims: DimSpec,
        reduce_dim: str
    ) -> int:
        """
        Compute the axis index for a reduction operation.

        Args:
            input_dims: Dimensions of the input tensor
            reduce_dim: The dimension to reduce over

        Returns:
            Axis index for the reduction

        Raises:
            ValueError: If reduce_dim is not in input_dims

        Example:
            input_dims = ('e', 'm', 'p'), reduce_dim = 'e'
            returns 0  # reduce over axis 0
        """
        try:
            return input_dims.dims.index(reduce_dim)
        except ValueError:
            raise ValueError(
                f"Reduce dimension '{reduce_dim}' not found in input dims {input_dims}"
            )

    def compute_permutation(
        self,
        current_dims: DimSpec,
        target_dims: DimSpec
    ) -> Optional[List[int]]:
        """
        Compute permutation to reorder dimensions.

        Args:
            current_dims: Current dimension ordering
            target_dims: Target dimension ordering

        Returns:
            List of axis indices for permutation, or None if no permutation needed

        Example:
            current_dims = ('m', 'e', 'p'), target_dims = ('e', 'm', 'p')
            returns [1, 0, 2]  # swap first two dimensions
        """
        if current_dims.dims == target_dims.dims:
            return None

        if set(current_dims.dims) != set(target_dims.dims):
            raise ValueError(
                f"Cannot permute {current_dims} to {target_dims}: different dimensions"
            )

        permutation = []
        for dim in target_dims.dims:
            permutation.append(current_dims.dims.index(dim))

        return permutation

    def infer_broadcast_output(
        self,
        dims_a: DimSpec,
        dims_b: DimSpec
    ) -> DimSpec:
        """
        Infer the output dimensions from broadcasting two tensors.

        Uses canonical ordering to determine dimension positions.

        Args:
            dims_a: Dimensions of first tensor
            dims_b: Dimensions of second tensor

        Returns:
            Output dimensions after broadcasting

        Example:
            dims_a = ('e', 'p'), dims_b = ('e', 'm')
            returns DimSpec(('e', 'm', 'p'))  # following canonical order
        """
        all_dims = set(dims_a.dims) | set(dims_b.dims)

        # Sort by canonical order
        ordered = sorted(all_dims, key=lambda d: self._dim_positions.get(d, 999))

        return DimSpec(tuple(ordered))

    def dims_compatible(
        self,
        dims_a: DimSpec,
        dims_b: DimSpec
    ) -> bool:
        """
        Check if two DimSpecs are compatible for element-wise operations.

        Compatible means one is a subset of the other (can broadcast).

        Args:
            dims_a: Dimensions of first tensor
            dims_b: Dimensions of second tensor

        Returns:
            True if dimensions are compatible for broadcasting
        """
        set_a = set(dims_a.dims)
        set_b = set(dims_b.dims)

        return set_a.issubset(set_b) or set_b.issubset(set_a)

    def get_dim_position(self, dim: str) -> int:
        """Get canonical position of a dimension."""
        return self._dim_positions.get(dim, -1)

    def normalize_dims(self, dims: DimSpec) -> DimSpec:
        """Reorder dimensions to canonical order."""
        ordered = sorted(dims.dims, key=lambda d: self._dim_positions.get(d, 999))
        return DimSpec(tuple(ordered))


# Singleton instance for convenience
_resolver = None


def get_resolver() -> DimResolver:
    """Get the singleton DimResolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = DimResolver()
    return _resolver


def compute_broadcast_positions(input_dims: DimSpec, output_dims: DimSpec) -> List[int]:
    """Convenience function for computing broadcast positions."""
    return get_resolver().compute_broadcast_positions(input_dims, output_dims)


def compute_unsqueeze_sequence(input_dims: DimSpec, output_dims: DimSpec) -> List[int]:
    """Convenience function for computing unsqueeze sequence."""
    return get_resolver().compute_unsqueeze_sequence(input_dims, output_dims)


def compute_reduce_axis(input_dims: DimSpec, reduce_dim: str) -> int:
    """Convenience function for computing reduce axis."""
    return get_resolver().compute_reduce_axis(input_dims, reduce_dim)


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("Testing DimResolver")
    print("=" * 60)

    resolver = DimResolver()

    # Test 1: M_mul_emp: Q[e,p] * K[e,m] -> [e,m,p]
    print("\nTest 1: M_mul_emp broadcast")
    q_dims = DimSpec(('e', 'p'))
    k_dims = DimSpec(('e', 'm'))
    out_dims = DimSpec(('e', 'm', 'p'))

    q_unsqueeze = resolver.compute_unsqueeze_sequence(q_dims, out_dims)
    k_unsqueeze = resolver.compute_unsqueeze_sequence(k_dims, out_dims)

    print(f"  Q{q_dims} -> unsqueeze at {q_unsqueeze} -> {out_dims}")
    print(f"  K{k_dims} -> unsqueeze at {k_unsqueeze} -> {out_dims}")
    assert q_unsqueeze == [1], f"Expected [1], got {q_unsqueeze}"
    assert k_unsqueeze == [2], f"Expected [2], got {k_unsqueeze}"
    print("  ✓ Passed")

    # Test 2: M_mul_fmp: A[m,p] * V[f,m] -> [f,m,p]
    print("\nTest 2: M_mul_fmp broadcast")
    a_dims = DimSpec(('m', 'p'))
    v_dims = DimSpec(('f', 'm'))
    out_dims = DimSpec(('f', 'm', 'p'))

    a_unsqueeze = resolver.compute_unsqueeze_sequence(a_dims, out_dims)
    v_unsqueeze = resolver.compute_unsqueeze_sequence(v_dims, out_dims)

    print(f"  A{a_dims} -> unsqueeze at {a_unsqueeze} -> {out_dims}")
    print(f"  V{v_dims} -> unsqueeze at {v_unsqueeze} -> {out_dims}")
    assert a_unsqueeze == [0], f"Expected [0], got {a_unsqueeze}"
    assert v_unsqueeze == [2], f"Expected [2], got {v_unsqueeze}"
    print("  ✓ Passed")

    # Test 3: M_sub_mp: A[m,p] - B[p] -> [m,p]
    print("\nTest 3: M_sub_mp broadcast")
    a_dims = DimSpec(('m', 'p'))
    b_dims = DimSpec(('p',))
    out_dims = DimSpec(('m', 'p'))

    a_unsqueeze = resolver.compute_unsqueeze_sequence(a_dims, out_dims)
    b_unsqueeze = resolver.compute_unsqueeze_sequence(b_dims, out_dims)

    print(f"  A{a_dims} -> unsqueeze at {a_unsqueeze} -> {out_dims}")
    print(f"  B{b_dims} -> unsqueeze at {b_unsqueeze} -> {out_dims}")
    assert a_unsqueeze == [], f"Expected [], got {a_unsqueeze}"
    assert b_unsqueeze == [0], f"Expected [0], got {b_unsqueeze}"
    print("  ✓ Passed")

    # Test 4: R_add_e reduce
    print("\nTest 4: R_add_e reduction")
    in_dims = DimSpec(('e', 'm', 'p'))
    axis = resolver.compute_reduce_axis(in_dims, 'e')
    print(f"  {in_dims} reduce 'e' -> axis {axis}")
    assert axis == 0, f"Expected 0, got {axis}"
    print("  ✓ Passed")

    # Test 5: R_add_m reduce
    print("\nTest 5: R_add_m reduction")
    in_dims = DimSpec(('f', 'm', 'p'))
    axis = resolver.compute_reduce_axis(in_dims, 'm')
    print(f"  {in_dims} reduce 'm' -> axis {axis}")
    assert axis == 1, f"Expected 1, got {axis}"
    print("  ✓ Passed")

    # Test 6: Tiled dimensions
    print("\nTest 6: Tiled M_mul_em1m0p broadcast")
    q_dims = DimSpec(('e', 'p'))
    bk_dims = DimSpec(('e', 'm1', 'm0'))
    out_dims = DimSpec(('e', 'm1', 'm0', 'p'))

    q_unsqueeze = resolver.compute_unsqueeze_sequence(q_dims, out_dims)
    bk_unsqueeze = resolver.compute_unsqueeze_sequence(bk_dims, out_dims)

    print(f"  Q{q_dims} -> unsqueeze at {q_unsqueeze} -> {out_dims}")
    print(f"  BK{bk_dims} -> unsqueeze at {bk_unsqueeze} -> {out_dims}")
    assert q_unsqueeze == [1, 2], f"Expected [1, 2], got {q_unsqueeze}"
    assert bk_unsqueeze == [3], f"Expected [3], got {bk_unsqueeze}"
    print("  ✓ Passed")

    # Test 7: Broadcast output inference
    print("\nTest 7: Broadcast output inference")
    q_dims = DimSpec(('e', 'p'))
    k_dims = DimSpec(('e', 'm'))
    inferred = resolver.infer_broadcast_output(q_dims, k_dims)
    print(f"  {q_dims} × {k_dims} -> {inferred}")
    assert inferred.dims == ('e', 'm', 'p'), f"Expected ('e', 'm', 'p'), got {inferred.dims}"
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
