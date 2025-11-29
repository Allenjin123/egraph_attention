import triton
import triton.language as tl
import torch

@triton.jit
def attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Graph-driven kernel generation
    # Memory passes: 2
    # Sync levels: 3

    # Get block indices
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute offsets for Q (this block of queries)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # Initialize accumulators for global reductions
    global_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    global_sum = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    global_sum = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Memory Pass 0: Iterate over all K/V blocks
    NUM_TILES = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(NUM_TILES):
        # Load K block
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)

        # Operations for sync level 0
        # QK = Q @ K^T (optimized)
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]
        # Tiling: K blocks loaded per iteration
        rmaxm0_R_max_m0 = tl.max(qk, axis=1)
        msubm1m0p_M_sub_m1m0p = qk - rmaxm0_R_max_m0
        mexpm1m0p_M_exp_m1m0p = tl.exp(msubm1m0p_M_sub_m1m0p)
        rmaxm1_R_max_m1 = tl.max(rmaxm0_R_max_m0, axis=0)
        # ^ Global reduction
        raddm0_R_add_m0 = tl.sum(mexpm1m0p_M_exp_m1m0p, axis=1)

    # Memory Pass 1: Iterate over all K/V blocks
    NUM_TILES = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(NUM_TILES):
        # Load K block
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)

        # Load V block
        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        # Operations for sync level 1
        msubm1p_M_sub_m1p = rmaxm0_R_max_m0 - rmaxm1_R_max_m1
        mexpm1p_M_exp_m1p = tl.exp(msubm1p_M_sub_m1p)
        mmulm1m0p_M_mul_m1m0p = mexpm1m0p_M_exp_m1m0p * mexpm1p_M_exp_m1p
        mmulm1p_M_mul_m1p = raddm0_R_add_m0 * mexpm1p_M_exp_m1p
        # Untiling: accumulated across all tiles
        mmulfmp_M_mul_fmp = mmulm1m0p_M_mul_m1m0p * v
        raddm_R_add_m = tl.sum(mmulfmp_M_mul_fmp, axis=1)
        # ^ Global reduction
        raddm1_R_add_m1 = tl.sum(mmulm1p_M_mul_m1p, axis=0)
        # ^ Global reduction

    # Post-loop operations (don't need K/V access)
    mdivfp_M_div_fp = raddm_R_add_m / raddm1_R_add_m1

    # Store output
    Out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, acc)

# Kernel generated directly from computation graph