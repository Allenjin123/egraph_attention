"""
Kernel Scaffolds for Triton Code Generation

Defines template scaffolds with "holes" that get filled by graph-derived operations.
The scaffolds provide the kernel structure (blocks, loops, memory access) while
the holes are filled with algorithm-specific computations from the graph.

Key Insight:
The kernel structure is consistent across attention algorithms. What varies is:
- Which operations happen inside each loop pass
- How many passes are needed
- What statistics to track (max, sum, correction factors)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class HoleType(Enum):
    """Types of holes in scaffolds."""
    INIT = "init"              # Initialization (before loops)
    PASS_BODY = "pass_body"    # Inside a pass loop
    PASS_END = "pass_end"      # After pass loop completes
    FINALIZE = "finalize"      # Final computations before store


@dataclass
class ScaffoldHole:
    """
    Defines an insertion point in the scaffold.

    Attributes:
        name: Unique identifier for this hole
        hole_type: Type of hole (init, pass_body, etc.)
        pass_num: Which pass this hole belongs to (1, 2, 3, or None for global)
        description: Human-readable description
        available_vars: Variables available at this point
        expected_outputs: Variables this hole should produce
    """
    name: str
    hole_type: HoleType
    pass_num: Optional[int] = None
    description: str = ""
    available_vars: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)


@dataclass
class LoopInfo:
    """Information about a loop in the scaffold."""
    loop_var: str        # e.g., "block_n"
    loop_range: str      # e.g., "range(num_blocks_n)"
    block_var: str       # e.g., "k" (the loaded block)
    block_size: str      # e.g., "BLOCK_N"


# ============================================================================
# Base Scaffold Class
# ============================================================================

class KernelScaffold:
    """Base class for kernel scaffolds."""

    def __init__(self):
        self.holes: Dict[str, ScaffoldHole] = {}
        self.hole_contents: Dict[str, List[str]] = {}

    def add_hole(self, hole: ScaffoldHole):
        """Register a hole in the scaffold."""
        self.holes[hole.name] = hole
        self.hole_contents[hole.name] = []

    def fill_hole(self, name: str, code_lines: List[str]):
        """Fill a hole with generated code."""
        if name not in self.holes:
            raise ValueError(f"Unknown hole: {name}")
        self.hole_contents[name] = code_lines

    def get_hole_code(self, name: str, indent: str = "        ") -> str:
        """Get the code for a hole with proper indentation."""
        if name not in self.hole_contents:
            return f"{indent}# [HOLE: {name}] - not filled"
        lines = self.hole_contents.get(name, [])
        if not lines:
            return f"{indent}pass  # [HOLE: {name}] - empty"
        return "\n".join(f"{indent}{line}" for line in lines)

    def generate(self) -> str:
        """Generate the complete kernel code."""
        raise NotImplementedError


# ============================================================================
# Three-Pass Attention Scaffold
# ============================================================================

class ThreePassScaffold(KernelScaffold):
    """
    Scaffold for 3-pass attention kernel.

    Structure:
    - Pass 1: Load K blocks, compute QK, find row-wise max
    - Pass 2: Load K blocks, compute exp(QK - max), sum
    - Pass 3: Load K/V blocks, compute normalized attention, accumulate output
    """

    def __init__(self):
        super().__init__()
        self._define_holes()

    def _define_holes(self):
        """Define all holes for 3-pass scaffold."""
        # Initialization
        self.add_hole(ScaffoldHole(
            name="init",
            hole_type=HoleType.INIT,
            description="Initialize accumulators before loops",
            available_vars=["q", "scale", "BLOCK_M", "BLOCK_N", "BLOCK_K"],
            expected_outputs=["row_max"]
        ))

        # Pass 1: Find max
        self.add_hole(ScaffoldHole(
            name="pass1_qk",
            hole_type=HoleType.PASS_BODY,
            pass_num=1,
            description="Compute QK scores",
            available_vars=["q", "k", "scale", "qk"],
            expected_outputs=["qk_scaled"]
        ))

        self.add_hole(ScaffoldHole(
            name="pass1_update_max",
            hole_type=HoleType.PASS_BODY,
            pass_num=1,
            description="Update row-wise max",
            available_vars=["qk_scaled", "row_max"],
            expected_outputs=["row_max"]
        ))

        # Pass 2: Compute exp sum
        self.add_hole(ScaffoldHole(
            name="pass2_exp",
            hole_type=HoleType.PASS_BODY,
            pass_num=2,
            description="Compute exp(QK - max)",
            available_vars=["qk_scaled", "row_max"],
            expected_outputs=["p"]
        ))

        self.add_hole(ScaffoldHole(
            name="pass2_update_sum",
            hole_type=HoleType.PASS_BODY,
            pass_num=2,
            description="Update row-wise sum",
            available_vars=["p", "row_sum"],
            expected_outputs=["row_sum"]
        ))

        # Pass 3: Compute output
        self.add_hole(ScaffoldHole(
            name="pass3_normalize",
            hole_type=HoleType.PASS_BODY,
            pass_num=3,
            description="Normalize attention weights",
            available_vars=["qk_scaled", "row_max", "row_sum"],
            expected_outputs=["p_norm"]
        ))

        self.add_hole(ScaffoldHole(
            name="pass3_accumulate",
            hole_type=HoleType.PASS_BODY,
            pass_num=3,
            description="Accumulate output",
            available_vars=["p_norm", "v", "acc"],
            expected_outputs=["acc"]
        ))

    def generate(self) -> str:
        """Generate the 3-pass kernel code."""
        return f'''
@triton.jit
def _attention_3pass_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """3-pass attention kernel generated from egglog graph."""
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Query block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Load Q block
    q_ptrs = Q_ptr + (pid_batch * stride_qb + pid_head * stride_qh +
                      offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # === INITIALIZATION ===
{self.get_hole_code("init")}
    row_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)

    num_blocks_n = tl.cdiv(seq_len_k, BLOCK_N)

    # === PASS 1: Compute QK and find row-wise max ===
    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block
        k_ptrs = K_ptr + (pid_batch * stride_kb + pid_head * stride_kh +
                          offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale
        qk = tl.where(offs_n[None, :] < seq_len_k, qk, float('-inf'))

{self.get_hole_code("pass1_qk")}

        # Update row max
        block_max = tl.max(qk, axis=1)
        row_max = tl.maximum(row_max, block_max)

{self.get_hole_code("pass1_update_max")}

    # === PASS 2: Compute exp sum ===
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + (pid_batch * stride_kb + pid_head * stride_kh +
                          offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale
        qk = tl.where(offs_n[None, :] < seq_len_k, qk, float('-inf'))

        # Compute exp(qk - max)
        p = tl.exp(qk - row_max[:, None])
        row_sum += tl.sum(p, axis=1)

{self.get_hole_code("pass2_exp")}
{self.get_hole_code("pass2_update_sum")}

    # === PASS 3: Compute output ===
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + (pid_batch * stride_kb + pid_head * stride_kh +
                          offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale
        qk = tl.where(offs_n[None, :] < seq_len_k, qk, float('-inf'))

        # Normalized attention weights
        p = tl.exp(qk - row_max[:, None]) / row_sum[:, None]

{self.get_hole_code("pass3_normalize")}

        # Load V and accumulate
        v_ptrs = V_ptr + (pid_batch * stride_vb + pid_head * stride_vh +
                          offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        v_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)

{self.get_hole_code("pass3_accumulate")}

    # Store output
    out_ptrs = Out_ptr + (pid_batch * stride_ob + pid_head * stride_oh +
                          offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    out_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)
'''


# ============================================================================
# Two-Pass Attention Scaffold
# ============================================================================

class TwoPassScaffold(KernelScaffold):
    """
    Scaffold for 2-pass (FuseMax) attention kernel.

    Structure:
    - Pass 1: Load K blocks, compute QK, find local max per tile, compute global max
    - Pass 2: Load K/V blocks, apply correction factor, accumulate normalized output
    """

    def __init__(self):
        super().__init__()
        self._define_holes()

    def _define_holes(self):
        """Define all holes for 2-pass scaffold."""
        # Pass 1 holes
        self.add_hole(ScaffoldHole(
            name="pass1_local_max",
            hole_type=HoleType.PASS_BODY,
            pass_num=1,
            description="Compute local max for tile",
            available_vars=["qk", "local_max"],
            expected_outputs=["local_max"]
        ))

        self.add_hole(ScaffoldHole(
            name="pass1_global_max",
            hole_type=HoleType.PASS_BODY,
            pass_num=1,
            description="Update global max from local max",
            available_vars=["local_max", "global_max"],
            expected_outputs=["global_max"]
        ))

        # Pass 2 holes
        self.add_hole(ScaffoldHole(
            name="pass2_correction",
            hole_type=HoleType.PASS_BODY,
            pass_num=2,
            description="Compute correction factor exp(local_max - global_max)",
            available_vars=["local_max", "global_max"],
            expected_outputs=["correction"]
        ))

        self.add_hole(ScaffoldHole(
            name="pass2_local_softmax",
            hole_type=HoleType.PASS_BODY,
            pass_num=2,
            description="Compute local softmax with local max",
            available_vars=["qk", "local_max"],
            expected_outputs=["local_exp", "local_sum"]
        ))

        self.add_hole(ScaffoldHole(
            name="pass2_accumulate",
            hole_type=HoleType.PASS_BODY,
            pass_num=2,
            description="Apply correction and accumulate",
            available_vars=["local_exp", "correction", "v", "acc"],
            expected_outputs=["acc", "global_sum"]
        ))

        self.add_hole(ScaffoldHole(
            name="finalize",
            hole_type=HoleType.FINALIZE,
            description="Final normalization by global sum",
            available_vars=["acc", "global_sum"],
            expected_outputs=["acc"]
        ))

    def generate(self) -> str:
        """Generate the 2-pass kernel code."""
        return f'''
@triton.jit
def _attention_2pass_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    scale,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """2-pass (FuseMax) attention kernel generated from egglog graph."""
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Query block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Load Q block
    q_ptrs = Q_ptr + (pid_batch * stride_qb + pid_head * stride_qh +
                      offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # === PASS 1: Find global max across all tiles ===
    global_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)

    for tile_idx in range(NUM_TILES):
        start_n = tile_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K tile
        k_ptrs = K_ptr + (pid_batch * stride_kb + pid_head * stride_kh +
                          offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK for this tile
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale
        qk = tl.where(offs_n[None, :] < seq_len_k, qk, float('-inf'))

        # Local max for this tile
        local_max = tl.max(qk, axis=1)

{self.get_hole_code("pass1_local_max")}

        # Update global max
        global_max = tl.maximum(global_max, local_max)

{self.get_hole_code("pass1_global_max")}

    # === PASS 2: Compute output using global max for correction ===
    global_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for tile_idx in range(NUM_TILES):
        start_n = tile_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K tile (again)
        k_ptrs = K_ptr + (pid_batch * stride_kb + pid_head * stride_kh +
                          offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Recompute QK
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale
        qk = tl.where(offs_n[None, :] < seq_len_k, qk, float('-inf'))

        # Local max for this tile
        local_max = tl.max(qk, axis=1)

        # Local softmax (numerically stable within tile)
        local_exp = tl.exp(qk - local_max[:, None])
        local_sum = tl.sum(local_exp, axis=1)

{self.get_hole_code("pass2_local_softmax")}

        # Correction factor: exp(local_max - global_max)
        correction = tl.exp(local_max - global_max)

{self.get_hole_code("pass2_correction")}

        # Corrected sum
        global_sum += local_sum * correction

        # Load V tile
        v_ptrs = V_ptr + (pid_batch * stride_vb + pid_head * stride_vh +
                          offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        v_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Corrected attention weights
        corrected_exp = local_exp * correction[:, None]
        acc += tl.dot(corrected_exp.to(v.dtype), v)

{self.get_hole_code("pass2_accumulate")}

    # Final normalization
    acc = acc / global_sum[:, None]

{self.get_hole_code("finalize", indent="    ")}

    # Store output
    out_ptrs = Out_ptr + (pid_batch * stride_ob + pid_head * stride_oh +
                          offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    out_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)
'''


# ============================================================================
# Scaffold Factory
# ============================================================================

def get_scaffold(algorithm_type: str) -> KernelScaffold:
    """Get the appropriate scaffold for an algorithm type."""
    if algorithm_type in ('3pass', 'three_pass'):
        return ThreePassScaffold()
    elif algorithm_type in ('2pass', 'two_pass'):
        return TwoPassScaffold()
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("Testing Kernel Scaffolds")
    print("=" * 60)

    # Test 3-pass scaffold
    print("\n3-Pass Scaffold Holes:")
    scaffold_3pass = ThreePassScaffold()
    for name, hole in scaffold_3pass.holes.items():
        print(f"  {name}: {hole.description}")

    # Fill some holes
    scaffold_3pass.fill_hole("init", ["# Custom initialization"])
    scaffold_3pass.fill_hole("pass1_qk", ["# Custom QK computation"])

    # Generate
    print("\n3-Pass Kernel (first 100 lines):")
    code = scaffold_3pass.generate()
    for i, line in enumerate(code.split('\n')[:100]):
        print(line)

    print("\n" + "=" * 60)

    # Test 2-pass scaffold
    print("\n2-Pass Scaffold Holes:")
    scaffold_2pass = TwoPassScaffold()
    for name, hole in scaffold_2pass.holes.items():
        print(f"  {name}: {hole.description}")
