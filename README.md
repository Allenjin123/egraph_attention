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

### Format: `M_<iteration_space>_R_<reduce_op>_<reduce_rank>`

- **M_** = Map operation (defines iteration space)
- **<iteration_space>** = ALL ranks being iterated (output shape + reduced ranks)
- **R_** = Reduce operation
- **<reduce_op>** = Operation for reduction (add, max, none)
- **<reduce_rank>** = Rank being reduced (or "none")

### Example
```egglog
M_mul_emp_R_add_e
```
- Iterate over {e, m, p}
- Reduce using add over e
- Output: {m, p}
- Corresponds to: `QK_{m,p} = Σ_e Q_{e,p} × K_{e,m}`

## Tiling Notation

### Format: `T_<operation>_<from>_<to>`

```egglog
T_split_m_m1m0      ; Split m into m1 (tiles) and m0 (within tile)
T_unsplit_m1m0_m    ; Combine m1 and m0 back into m
```

## 3-Pass Attention

Standard attention with three passes over the data:

```egglog
QK_{m,p} = Q_{e,p} @ K_{e,m}           ; Pass 1: Compute scores
GM_p = max_m(QK_{m,p})                 ; Pass 1: Global max
SN_{m,p} = exp(QK_{m,p} - GM_p)        ; Pass 2: Numerator
SD_p = sum_m(SN_{m,p})                 ; Pass 2: Denominator
A_{m,p} = SN_{m,p} / SD_p              ; Pass 3: Normalize
AV_{f,p} = A_{m,p} @ V_{f,m}           ; Pass 3: Output
```

**Memory**: O(M × P) for full QK and SN matrices

## 2-Pass Attention

Tiled attention with two passes (from FuseMax paper):

```egglog
BK_{e,m1,m0} = split(K_{e,m})          ; Tile K only

; Pass 1: Local statistics
BQK_{m1,m0,p} = Q_{e,p} @ BK
LM_{m1,p} = max_{m0}(BQK)              ; Local max per tile
GM_p = max_{m1}(LM)                    ; Global max ← BARRIER
SLN_{m1,m0,p} = exp(BQK - LM)
SLD_{m1,p} = sum_{m0}(SLN)

; Pass 2: Correction
PRM_{m1,p} = exp(LM - GM)              ; Correction factor
CN = SLN * PRM                         ; Correct numerator
CD = SLD * PRM                         ; Correct denominator
GD_p = sum_{m1}(CD)                    ; Global denominator
A_tiled = CN / GD
A_{m,p} = unsplit(A_tiled)
AV_{f,p} = A @ V
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
