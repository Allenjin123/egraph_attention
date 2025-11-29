"""
Operation Registry for Egglog-to-Triton Compiler

Defines dimension semantics for each egglog operation used in attention algorithms.
Dimensions:
    e: embedding dimension (Q/K inner product)
    f: embedding dimension (V output, often same as e)
    m: sequence length for K/V
    p: sequence length for Q
    m0, m1: tiled versions of m (m = m1 × m0)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class DimSpec:
    """Dimension specification for a tensor."""
    dims: Tuple[str, ...]  # e.g., ('m', 'p') or ('m1', 'm0', 'p')

    def __str__(self):
        return f"[{', '.join(self.dims)}]"

    def without(self, dim: str) -> 'DimSpec':
        """Return new DimSpec with specified dimension removed."""
        return DimSpec(tuple(d for d in self.dims if d != dim))

    def has(self, dim: str) -> bool:
        """Check if dimension exists."""
        return dim in self.dims


@dataclass
class OpSpec:
    """Specification for an egglog operation."""
    name: str
    op_type: str  # 'map', 'reduce', 'tile', 'create'
    output_dims: DimSpec
    reduce_dim: Optional[str] = None  # for reduce ops
    compute_fn: Optional[str] = None  # PyTorch equivalent for verification
    triton_op: Optional[str] = None   # Triton operation (e.g., 'tl.exp', '+', '*')


# ============================================================================
# Operation Registry
# ============================================================================

OP_REGISTRY = {
    # ========== Map Operations (element-wise) ==========

    # M_mul_emp: Q[e,p] * K[e,m] -> temp[e,m,p] (broadcast multiply)
    'M_mul_emp': OpSpec(
        name='M_mul_emp',
        op_type='map',
        output_dims=DimSpec(('e', 'm', 'p')),
        triton_op='*',
        compute_fn='lambda Q, K: Q.unsqueeze(2) * K.unsqueeze(3)'  # [e,p,1] * [e,m,1] -> [e,m,p]
    ),

    # M_sub_mp: A[m,p] - B[p] -> C[m,p] (broadcast subtract)
    'M_sub_mp': OpSpec(
        name='M_sub_mp',
        op_type='map',
        output_dims=DimSpec(('m', 'p')),
        triton_op='-',
        compute_fn='lambda A, B: A - B.unsqueeze(0)'
    ),

    # M_exp_mp: exp(A[m,p]) -> B[m,p]
    'M_exp_mp': OpSpec(
        name='M_exp_mp',
        op_type='map',
        output_dims=DimSpec(('m', 'p')),
        triton_op='tl.exp',
        compute_fn='torch.exp'
    ),

    # M_div_mp: A[m,p] / B[p] -> C[m,p] (broadcast divide)
    'M_div_mp': OpSpec(
        name='M_div_mp',
        op_type='map',
        output_dims=DimSpec(('m', 'p')),
        triton_op='/',
        compute_fn='lambda A, B: A / B.unsqueeze(0)'
    ),

    # M_mul_fmp: A[m,p] * V[f,m] -> temp[f,m,p] (broadcast multiply for output)
    'M_mul_fmp': OpSpec(
        name='M_mul_fmp',
        op_type='map',
        output_dims=DimSpec(('f', 'm', 'p')),
        triton_op='*',
        compute_fn='lambda A, V: A.unsqueeze(0) * V.unsqueeze(2)'  # [1,m,p] * [f,m,1] -> [f,m,p]
    ),

    # ========== Tiled Map Operations ==========

    # M_mul_em1m0p: Q[e,p] * BK[e,m1,m0] -> temp[e,m1,m0,p]
    'M_mul_em1m0p': OpSpec(
        name='M_mul_em1m0p',
        op_type='map',
        output_dims=DimSpec(('e', 'm1', 'm0', 'p')),
        triton_op='*',
    ),

    # M_sub_m1m0p: A[m1,m0,p] - B[m1,p] -> C[m1,m0,p]
    'M_sub_m1m0p': OpSpec(
        name='M_sub_m1m0p',
        op_type='map',
        output_dims=DimSpec(('m1', 'm0', 'p')),
        triton_op='-',
    ),

    # M_sub_m1p: A[m1,p] - B[p] -> C[m1,p]
    'M_sub_m1p': OpSpec(
        name='M_sub_m1p',
        op_type='map',
        output_dims=DimSpec(('m1', 'p')),
        triton_op='-',
    ),

    # M_exp_m1m0p: exp(A[m1,m0,p]) -> B[m1,m0,p]
    'M_exp_m1m0p': OpSpec(
        name='M_exp_m1m0p',
        op_type='map',
        output_dims=DimSpec(('m1', 'm0', 'p')),
        triton_op='tl.exp',
    ),

    # M_exp_m1p: exp(A[m1,p]) -> B[m1,p]
    'M_exp_m1p': OpSpec(
        name='M_exp_m1p',
        op_type='map',
        output_dims=DimSpec(('m1', 'p')),
        triton_op='tl.exp',
    ),

    # M_mul_m1m0p: A[m1,m0,p] * B[m1,p] -> C[m1,m0,p]
    'M_mul_m1m0p': OpSpec(
        name='M_mul_m1m0p',
        op_type='map',
        output_dims=DimSpec(('m1', 'm0', 'p')),
        triton_op='*',
    ),

    # M_mul_m1p: A[m1,p] * B[m1,p] -> C[m1,p]
    'M_mul_m1p': OpSpec(
        name='M_mul_m1p',
        op_type='map',
        output_dims=DimSpec(('m1', 'p')),
        triton_op='*',
    ),

    # M_div_m1m0p: A[m1,m0,p] / B[p] -> C[m1,m0,p]
    'M_div_m1m0p': OpSpec(
        name='M_div_m1m0p',
        op_type='map',
        output_dims=DimSpec(('m1', 'm0', 'p')),
        triton_op='/',
    ),

    # M_div_fp: A[f,p] / B[p] -> C[f,p] (broadcast divide for output normalization)
    'M_div_fp': OpSpec(
        name='M_div_fp',
        op_type='map',
        output_dims=DimSpec(('f', 'p')),
        triton_op='/',
        compute_fn='lambda A, B: A / B.unsqueeze(0)'
    ),

    # M_mul_fm1m0p: A[m1,m0,p] * V[f,m1,m0] -> temp[f,m1,m0,p]
    'M_mul_fm1m0p': OpSpec(
        name='M_mul_fm1m0p',
        op_type='map',
        output_dims=DimSpec(('f', 'm1', 'm0', 'p')),
        triton_op='*',
    ),

    # ========== Reduce Operations ==========

    # R_add_e: sum over e dimension
    'R_add_e': OpSpec(
        name='R_add_e',
        op_type='reduce',
        output_dims=DimSpec(('m', 'p')),
        reduce_dim='e',
        triton_op='tl.sum',
        compute_fn='lambda x: x.sum(dim=0)'  # sum over e
    ),

    # R_add_m: sum over m dimension
    'R_add_m': OpSpec(
        name='R_add_m',
        op_type='reduce',
        output_dims=DimSpec(('p',)),  # or ('f', 'p') depending on input
        reduce_dim='m',
        triton_op='tl.sum',
        compute_fn='lambda x: x.sum(dim=-2)'  # sum over m (second to last)
    ),

    # R_max_m: max over m dimension
    'R_max_m': OpSpec(
        name='R_max_m',
        op_type='reduce',
        output_dims=DimSpec(('p',)),
        reduce_dim='m',
        triton_op='tl.max',
        compute_fn='lambda x: x.max(dim=0)[0]'
    ),

    # ========== Tiled Reduce Operations ==========

    # R_add_m0: sum over m0 (within tile)
    'R_add_m0': OpSpec(
        name='R_add_m0',
        op_type='reduce',
        output_dims=DimSpec(('m1', 'p')),
        reduce_dim='m0',
        triton_op='tl.sum',
    ),

    # R_add_m1: sum over m1 (across tiles)
    'R_add_m1': OpSpec(
        name='R_add_m1',
        op_type='reduce',
        output_dims=DimSpec(('p',)),
        reduce_dim='m1',
        triton_op='tl.sum',
    ),

    # R_max_m0: max over m0 (within tile)
    'R_max_m0': OpSpec(
        name='R_max_m0',
        op_type='reduce',
        output_dims=DimSpec(('m1', 'p')),
        reduce_dim='m0',
        triton_op='tl.max',
    ),

    # R_max_m1: max over m1 (across tiles)
    'R_max_m1': OpSpec(
        name='R_max_m1',
        op_type='reduce',
        output_dims=DimSpec(('p',)),
        reduce_dim='m1',
        triton_op='tl.max',
    ),

    # ========== Tiling Operations ==========

    # T_split_m_m1m0: Split m into m1 x m0 tiles
    'T_split_m_m1m0': OpSpec(
        name='T_split_m_m1m0',
        op_type='tile',
        output_dims=DimSpec(('e', 'm1', 'm0')),  # Transforms [e,m] -> [e,m1,m0]
    ),

    # T_unsplit_m1m0_m: Merge m1 x m0 back to m
    'T_unsplit_m1m0_m': OpSpec(
        name='T_unsplit_m1m0_m',
        op_type='tile',
        output_dims=DimSpec(('m', 'p')),  # Transforms [m1,m0,p] -> [m,p]
    ),

    # ========== Create Operations ==========
    'CreateTensor': OpSpec(
        name='CreateTensor',
        op_type='create',
        output_dims=DimSpec(('e', 'p')),  # Default, will be overridden based on tensor name
    ),
}


def get_op_spec(op_name: str) -> Optional[OpSpec]:
    """Get operation specification by name."""
    return OP_REGISTRY.get(op_name)


def infer_output_dims(op_name: str, input_dims: List[DimSpec]) -> DimSpec:
    """Infer output dimensions based on operation and input dimensions."""
    spec = get_op_spec(op_name)
    if spec is None:
        raise ValueError(f"Unknown operation: {op_name}")

    if spec.op_type == 'reduce':
        # For reduce ops, remove the reduce dimension from input
        if input_dims:
            return input_dims[0].without(spec.reduce_dim)

    return spec.output_dims


# Dimension size mapping (runtime configuration)
DIM_SIZES = {
    'e': 64,   # Embedding dimension (head_dim)
    'f': 64,   # Output embedding (same as e for self-attention)
    'm': 1024, # Key/Value sequence length
    'p': 1024, # Query sequence length
    'm0': 256, # Tile size (within tile)
    'm1': 4,   # Number of tiles (m / m0)
}


def set_dim_sizes(e: int = 64, m: int = 1024, p: int = None, tile_size: int = 256):
    """Configure dimension sizes for code generation."""
    global DIM_SIZES
    DIM_SIZES['e'] = e
    DIM_SIZES['f'] = e  # Assume f == e for self-attention
    DIM_SIZES['m'] = m
    DIM_SIZES['p'] = p if p is not None else m
    DIM_SIZES['m0'] = tile_size
    DIM_SIZES['m1'] = m // tile_size


if __name__ == "__main__":
    # Test the registry
    print("Operation Registry:")
    print("=" * 60)
    for name, spec in OP_REGISTRY.items():
        print(f"{name:20s} | {spec.op_type:8s} | {spec.output_dims}")
