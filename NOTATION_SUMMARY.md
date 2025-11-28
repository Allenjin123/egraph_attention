# Operator Notation Summary

## Design Philosophy: Split Map and Reduce

Map and reduce operations are **always separate** to enable compositional rewrite rules. This allows inserting transformations (e.g., tiling) between map and reduce.

## Notation Convention

### Map Operations: `M_<operation>_<iteration_space>`

- **M_** = Map operation (element-wise, no reduction)
- **<operation>** = The computation (mul, sub, exp, div)
- **<iteration_space>** = ALL ranks in the output tensor

**Example**: `M_mul_emp` - Element-wise multiply over iteration space {e,m,p}

### Reduce Operations: `R_<operation>_<rank>`

- **R_** = Reduce operation (collapses a dimension)
- **<operation>** = Reduction operator (add, max)
- **<rank>** = Which rank to reduce

**Example**: `R_add_e` - Reduce using addition over rank e

### Tiling Operations: `T_<operation>_<from>_<to>`

**Split**: `T_split_m_m1m0` - Split dimension m into m1 and m0
**Unsplit**: `T_unsplit_m1m0_m` - Combine m1 and m0 back into m

## 3-Pass Attention

```egglog
; Step 1: QK = Q @ K
QK_temp = M_mul_emp(Q, K)     ; Map multiply: temp_{e,m,p}
QK = R_add_e(QK_temp)         ; Reduce e: QK_{m,p}

; Step 2: GM = max(QK)
GM = R_max_m(QK)              ; Reduce m: GM_p

; Step 3: SN = exp(QK - GM)
QK_shifted = M_sub_mp(QK, GM) ; Map subtract
SN = M_exp_mp(QK_shifted)     ; Map exp

; Step 4: SD = sum(SN)
SD = R_add_m(SN)              ; Reduce m: SD_p

; Step 5: A = SN / SD
A = M_div_mp(SN, SD)          ; Map divide

; Step 6: AV = A @ V
AV_temp = M_mul_fmp(A, V)     ; Map multiply: temp_{f,m,p}
AV = R_add_m(AV_temp)         ; Reduce m: AV_{f,p}
```

## 2-Pass Attention

### Tiling: Only K (not Q, not V)

```egglog
BK = T_split_m_m1m0(K, 4)     ; K_{e,m} → BK_{e,m1,m0}
```

### Pass 1: Local Statistics + Global Max

```egglog
; BQK = Q @ BK (tiled matmul)
BQK_temp = M_mul_em1m0p(Q, BK)    ; Map: temp_{e,m1,m0,p}
BQK = R_add_e(BQK_temp)           ; Reduce e: BQK_{m1,m0,p}

; Local max per tile
LM = R_max_m0(BQK)                ; LM_{m1,p} = max_{m0}(BQK)

; ──────────────── BARRIER ────────────────
GM = R_max_m1(LM)                 ; GM_p = max_{m1}(LM)
; ─────────────────────────────────────────

; Local numerator
BQK_shifted = M_sub_m1m0p(BQK, LM)
SLN = M_exp_m1m0p(BQK_shifted)    ; SLN_{m1,m0,p}

; Local denominator
SLD = R_add_m0(SLN)               ; SLD_{m1,p} = Σ_{m0} SLN
```

### Pass 2: Correction + Output

```egglog
; Correction factor
LM_GM_diff = M_sub_m1p(LM, GM)
PRM = M_exp_m1p(LM_GM_diff)       ; PRM_{m1,p} = exp(LM - GM)

; Correct numerator and denominator
CN = M_mul_m1m0p(SLN, PRM)        ; CN_{m1,m0,p}
CD = M_mul_m1p(SLD, PRM)          ; CD_{m1,p}

; Global denominator
GD = R_add_m1(CD)                 ; GD_p = Σ_{m1} CD

; Normalize
A_tiled = M_div_m1m0p(CN, GD)     ; A_{m1,m0,p} = CN / GD

; Untile and compute final output
A = T_unsplit_m1m0_m(A_tiled)     ; A_{m,p}
AV_temp = M_mul_fmp(A, V)
AV = R_add_m(AV_temp)             ; AV_{f,p}
```

## Benefits of Split M/R Design

### 1. Compositional Rewrite Rules

**Example**: Inserting tiling between map and reduce

```egglog
; 3-pass (no tiling)
temp = M_mul_emp(Q, K)
QK = R_add_e(temp)

; Rewrite rule can match and transform:
temp = M_mul_emp(Q, K)
temp_tiled = insert_tiling(temp)    ; ← Inserted by rewrite rule
QK_tiled = R_add_e(temp_tiled)
```

### 2. Clear Operator Semantics

**Map operators** - Always element-wise, preserve all dimensions:
- `M_mul_emp`: Multiply Q and K element-wise over {e,m,p}
- `M_exp_mp`: Exponentiate element-wise over {m,p}

**Reduce operators** - Always collapse one dimension:
- `R_add_e`: Sum over e dimension
- `R_max_m`: Max over m dimension

### 3. Easier Pattern Matching

Rewrite rules can match specific patterns:
```egglog
; Match: Map multiply followed by reduce
M_mul_<space>(A, B) followed by R_add_<rank>

; Transform: Insert tiling
M_mul_<tiled_space>(A, B_tiled) followed by R_add_<rank>
```

### 4. No Ambiguity

Every operator is unambiguous:
- `M_sub_mp` - Subtract over {m,p}
- `M_exp_mp` - Exp over {m,p}
- `R_max_m1` - Reduce max over m1

No need for "none" placeholders!

## Summary: 3-Pass vs 2-Pass

| Aspect | 3-Pass | 2-Pass |
|--------|--------|--------|
| **Map ops** | 6 (mul, sub, exp, div, mul) | 9 |
| **Reduce ops** | 4 (add×2, max×1) | 5 (add×3, max×2) |
| **Tiling** | 0 | 2 (split, unsplit) |
| **Barriers** | 2 (GM, SD) | 1 (GM) |
| **Passes** | 3 | 2 |

The split design makes it clear where rewrite rules can transform the computation!
