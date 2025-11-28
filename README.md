# Attention Optimization with Egglog

Implementing transformer attention algorithms in egglog to explore rewrite rules for optimizing attention computation (e.g., transforming standard attention to Flash Attention).

## Files

### Core Attention Implementations
- **`attention.egg`** - 3-pass attention algorithm (standard implementation)
- **`attention_2pass.egg`** - 2-pass attention algorithm (from FuseMax paper)

### Documentation
- **`NOTATION_SUMMARY.md`** - Explains the M/R operator notation
- **`TILING_NOTATION.md`** - Explains the tiling operation syntax
- **`2PASS_FINAL.md`** - Detailed explanation of 2-pass algorithm

### Reference
- **`fusemax.pdf`** - FuseMax paper (MICRO 2024)
- **`example.txt`** - Egglog Python documentation

### Visualizations
- **`attention.svg`** - Visualization of 3-pass attention
- **`attention_2pass.svg`** - Visualization of 2-pass attention

## Operator Notation

**Design**: Map and Reduce are **always separate operators** to enable compositional rewrite rules.

### Map Operations: `M_<operation>_<iteration_space>`
- Element-wise operations, no reduction
- Example: `M_mul_emp` - Multiply over {e,m,p}

### Reduce Operations: `R_<operation>_<rank>`
- Collapse one dimension
- Example: `R_add_e` - Sum over e dimension

### Example: Matrix Multiply
```egglog
temp_{e,m,p} = M_mul_emp(Q, K)    ; Map: element-wise multiply
QK_{m,p} = R_add_e(temp)          ; Reduce: sum over e
```
Corresponds to: `QK_{m,p} = Σ_e Q_{e,p} × K_{e,m}`

## Tiling Notation

### Format: `T_<operation>_<from>_<to>`

```egglog
T_split_m_m1m0      ; Split m into m1 (tiles) and m0 (within tile)
T_unsplit_m1m0_m    ; Combine m1 and m0 back into m
```

## 3-Pass Attention

Standard attention with three passes over the data:

```egglog
; Pass 1: Compute QK and global max
QK_temp = M_mul_emp(Q, K)
QK = R_add_e(QK_temp)
GM = R_max_m(QK)

; Pass 2: Compute softmax numerator and denominator
QK_shifted = M_sub_mp(QK, GM)
SN = M_exp_mp(QK_shifted)
SD = R_add_m(SN)

; Pass 3: Normalize and compute output
A = M_div_mp(SN, SD)
AV_temp = M_mul_fmp(A, V)
AV = R_add_m(AV_temp)
```

**Memory**: O(M × P) for full QK and SN matrices

## 2-Pass Attention

Tiled attention with two passes (from FuseMax paper):

```egglog
; Tile K only
BK = T_split_m_m1m0(K, 4)

; Pass 1: Local statistics
BQK_temp = M_mul_em1m0p(Q, BK)
BQK = R_add_e(BQK_temp)
LM = R_max_m0(BQK)                     ; Local max per tile
GM = R_max_m1(LM)                      ; Global max ← BARRIER
BQK_shifted = M_sub_m1m0p(BQK, LM)
SLN = M_exp_m1m0p(BQK_shifted)
SLD = R_add_m0(SLN)

; Pass 2: Correction
LM_GM_diff = M_sub_m1p(LM, GM)
PRM = M_exp_m1p(LM_GM_diff)
CN = M_mul_m1m0p(SLN, PRM)
CD = M_mul_m1p(SLD, PRM)
GD = R_add_m1(CD)                      ; Global denominator
A_tiled = M_div_m1m0p(CN, GD)
A = T_unsplit_m1m0_m(A_tiled)
AV_temp = M_mul_fmp(A, V)
AV = R_add_m(AV_temp)
```

**Memory**: O(M0 × P) per tile, where M0 << M

## Key Differences

| Aspect | 3-Pass | 2-Pass |
|--------|--------|--------|
| Tiling | None | K only |
| Passes | 3 | 2 |
| Memory | O(M × P) | O(M0 × P) |
| Scalability | Limited by memory | Works with long sequences |
| Barriers | 2 (GM, SD) | 1 (GM) |

## Running the Examples

```bash
# 3-pass attention
egglog attention.egg

# 2-pass attention
egglog attention_2pass.egg
```

## Next Steps

1. Define rewrite rules to transform 3-pass → 2-pass
2. Implement 1-pass algorithm (FlashAttention-2 from FuseMax paper)
3. Add cost models for extraction
4. Explore other tiling strategies

## References

- **FuseMax Paper**: "FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design" (MICRO 2024)
- **Egglog**: https://github.com/egraphs-good/egglog
