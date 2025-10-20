# Operator Notation Summary

## Notation Convention

### Format: `M_<iteration_space>_R_<reduce_op>_<reduce_rank>`

- **M_** = Map operation (defines iteration space)
- **<iteration_space>** = ALL ranks being iterated over (including those to be reduced)
- **R_** = Reduce operation
- **<reduce_op>** = Operation used for reduction (add, max, none)
- **<reduce_rank>** = Which rank is being reduced (or "none" if no reduction)

### Key Insight

The iteration space in `M_` includes ALL dimensions, even those that will be reduced away. This directly corresponds to the Einsum iteration space.

## 3-Pass Attention Operations

```egglog
; Tensor shapes
Q_{e,p}    ; Query:  embedding × sequence
K_{e,m}    ; Key:    embedding × sequence
V_{f,m}    ; Value:  embedding × sequence

; Operations
M_mul_emp_R_add_e       ; Iterate {e,m,p}, reduce e
                        ; QK_{m,p} = Σ_e Q_{e,p} × K_{e,m}

M_none_mp_R_max_m       ; Iterate {m,p}, reduce m
                        ; GM_p = max_m(QK_{m,p})

M_subexp_mp_R_none      ; Iterate {m,p}, no reduction
                        ; SN_{m,p} = exp(QK_{m,p} - GM_p)

M_none_mp_R_add_m       ; Iterate {m,p}, reduce m
                        ; SD_p = Σ_m SN_{m,p}

M_div_mp_R_none         ; Iterate {m,p}, no reduction
                        ; A_{m,p} = SN_{m,p} / SD_p

M_mul_fmp_R_add_m       ; Iterate {f,m,p}, reduce m
                        ; AV_{f,p} = Σ_m A_{m,p} × V_{f,m}
```

## 2-Pass Attention Operations

### Tiling Strategy

**Key difference**: Only tile K and V (not Q!)

```
m = m1 × m0
- m1: tile index (which block)
- m0: position within tile (within block)

K_{e,m} → BK_{e,m1,m0}
V_{f,m} → BV_{f,m1,m0}
Q_{e,p} stays untiled
```

### Operations

```egglog
; Tensor shapes after tiling
Q_{e,p}         ; Query: NOT tiled
BK_{e,m1,m0}    ; Key: tiled along m
BV_{f,m1,m0}    ; Value: tiled along m

; Pass 1: Local statistics
M_mul_em1m0p_R_add_e      ; Iterate {e,m1,m0,p}, reduce e
                          ; BQK_{m1,m0,p} = Σ_e Q_{e,p} × BK_{e,m1,m0}

M_none_m1m0p_R_max_m0     ; Iterate {m1,m0,p}, reduce m0
                          ; LM_{m1,p} = max_{m0}(BQK_{m1,m0,p})

; ──────────────────────────────────────────────────────────
; BARRIER: Global reduction over m1
; ──────────────────────────────────────────────────────────
M_none_m1p_R_max_m1       ; Iterate {m1,p}, reduce m1
                          ; GM_p = max_{m1}(LM_{m1,p})

M_subexp_m1m0p_R_none     ; Iterate {m1,m0,p}, no reduction
                          ; SLN_{m1,m0,p} = exp(BQK - LM)

M_none_m1m0p_R_add_m0     ; Iterate {m1,m0,p}, reduce m0
                          ; SLD_{m1,p} = Σ_{m0} SLN_{m1,m0,p}

; Pass 2: Correction
M_sub_m1p_R_none          ; Iterate {m1,p}, no reduction
                          ; Subtract element-wise

M_exp_m1p_R_none          ; Iterate {m1,p}, no reduction
                          ; PRM_{m1,p} = exp(LM_{m1,p} - GM_p)

M_mul_m1m0p_R_none        ; Iterate {m1,m0,p}, no reduction
                          ; CN_{m1,m0,p} = SLN_{m1,m0,p} × PRM_{m1,p}

M_mul_m1p_R_none          ; Iterate {m1,p}, no reduction
                          ; CD_{m1,p} = SLD_{m1,p} × PRM_{m1,p}

M_div_m1m0p_R_none        ; Iterate {m1,m0,p}, no reduction
                          ; A_{m1,m0,p} = CN_{m1,m0,p} / CD_{m1,p}

M_mul_fm1m0p_R_add_m0     ; Iterate {f,m1,m0,p}, reduce m0
                          ; AV_{f,m1,p} = Σ_{m0} A_{m1,m0,p} × BV_{f,m1,m0}
```

## Key Differences: 3-Pass vs 2-Pass

### Iteration Spaces

**3-Pass**: Simple ranks {e, m, p, f}
**2-Pass**: Tiled ranks {e, m1, m0, p, f} where m = m1 × m0

### Tiling

**3-Pass**: No explicit tiling
**2-Pass**:
- Tile K and V: `T_split_m`
- Q remains untiled
- Compute BQK from Q and tiled BK

### Barriers

**3-Pass**:
1. `M_none_mp_R_max_m`: Global max over all m
2. `M_none_mp_R_add_m`: Global sum over all m

**2-Pass**:
1. `M_none_m1p_R_max_m1`: Global max over tiles (m1)
   - This is the ONLY barrier between Pass 1 and Pass 2!

## Why This Notation is Better

### 1. Clarity of Iteration Space
```egglog
M_mul_emp_R_add_e
```
Immediately tells you:
- Iterate over {e, m, p}
- Reduce over e
- Result has shape {m, p}

### 2. Explicit Reduction
```egglog
M_div_mp_R_none
```
The `R_none` makes it explicit that there's no reduction (element-wise operation).

### 3. Identifies Barriers
```egglog
M_none_m1p_R_max_m1  ← Reduces over m1 (tiles)
                     ← This requires ALL tiles to finish!
                     ← BARRIER!
```

### 4. Matches Einsum Notation
```
QK_{m,p} = Σ_e Q_{e,p} × K_{e,m}
           ↓
M_mul_emp_R_add_e(Q, K)
      ↑↑↑     ↑    ↑
      emp = iteration space {e,m,p}
           add = reduction operation
               e = reduce over e dimension
```

## Consistency Rules

1. **Always specify iteration space in M**: Include ALL ranks, even those being reduced
2. **Always include R_<op>_<rank>**: Even if `R_none` (no reduction)
3. **Rank order in iteration space**: Alphabetical or logical (e.g., em1m0p for tiled)
4. **Tiled dimensions**: Use m1, m0 notation to show hierarchy

This makes every operation self-documenting and unambiguous!
