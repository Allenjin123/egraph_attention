# Final 2-Pass Attention Algorithm

## Key Design Decisions

### What to Tile

✅ **Tile K only** (not Q, not V)
```egglog
BK_{e,m1,m0} = split(K_{e,m})
Q_{e,p} stays untiled
V_{f,m} stays untiled
```

**Rationale:**
- We tile K because we need to compute attention scores in blocks
- Q doesn't need tiling because it's used in the outer product with each tile of K
- V doesn't need tiling because we untile A before multiplying

### Computation Flow

```
Pass 1: Compute Local Statistics
├─ BK = split(K)
├─ BQK = Q @ BK (tiled matmul)
├─ LM = max_{m0}(BQK) per tile
├─ GM = max_{m1}(LM) ← BARRIER #1
├─ SLN = exp(BQK - LM)
└─ SLD = sum_{m0}(SLN)

Pass 2: Correct and Compute Output
├─ PRM = exp(LM - GM)
├─ CN = SLN * PRM
├─ CD = SLD * PRM
├─ GD = sum_{m1}(CD) ← Synchronization #2
├─ A_tiled = CN / GD
├─ A = unsplit(A_tiled)
└─ AV = A @ V (standard matmul)
```

## Complete Algorithm

```egglog
; Tile K only
BK_{e,m1,m0} = split(K, 4 tiles)

; Pass 1: Local statistics + Global max
BQK_{m1,m0,p} = Q_{e,p} @ BK_{e,m1,m0}     ; M_mul_em1m0p_R_add_e
LM_{m1,p} = max_{m0}(BQK)                  ; M_none_m1m0p_R_max_m0
GM_p = max_{m1}(LM)                        ; M_none_m1p_R_max_m1 ← BARRIER
SLN_{m1,m0,p} = exp(BQK - LM)              ; M_subexp_m1m0p_R_none
SLD_{m1,p} = sum_{m0}(SLN)                 ; M_none_m1m0p_R_add_m0

; Pass 2: Correction + Output
PRM_{m1,p} = exp(LM - GM)                  ; M_sub_m1p_R_none, M_exp_m1p_R_none
CN_{m1,m0,p} = SLN * PRM                   ; M_mul_m1m0p_R_none
CD_{m1,p} = SLD * PRM                      ; M_mul_m1p_R_none
GD_p = sum_{m1}(CD)                        ; M_none_m1p_R_add_m1 ← Sync
A_{m1,m0,p} = CN / GD                      ; M_div_m1m0p_R_none
A_{m,p} = unsplit(A)                       ; T_unsplit_m
AV_{f,p} = A_{m,p} @ V_{f,m}               ; M_mul_fmp_R_add_m
```

## Why Not Tile V?

### If we tiled V (incorrect approach):
```
A_{m1,m0,p} (tiled)
BV_{f,m1,m0} (tiled)
AV_{f,m1,p} = sum_{m0} A * BV  ← Reduces m0, leaves m1
unsplit(AV_{f,m1,p}) → ???     ← Doesn't work! m0 already reduced
```

### Correct approach (untile A first):
```
A_{m1,m0,p} (tiled)
A_{m,p} = unsplit(A)           ← Reconstruct full A
V_{f,m} (never tiled)
AV_{f,p} = A @ V               ← Standard matmul
```

## Two Synchronization Points

### 1. Global Max (Pass 1 → Pass 2 Barrier)
```egglog
GM_p = max_{m1}(LM_{m1,p})  ; M_none_m1p_R_max_m1
```
- Reduces over m1 (tile dimension)
- **Must wait for ALL tiles** to compute LM
- Creates the boundary between Pass 1 and Pass 2

### 2. Global Denominator (Within Pass 2)
```egglog
GD_p = sum_{m1}(CD_{m1,p})  ; M_none_m1p_R_add_m1
```
- Reduces over m1 (tile dimension)
- Needed to normalize by global softmax denominator
- Still within Pass 2 (both depend on GM)

## Comparison with 3-Pass

| Aspect | 3-Pass | 2-Pass |
|--------|--------|--------|
| **Tiling** | None | K only |
| **Global reductions** | 2 (GM, GD over all m) | 2 (GM, GD over tiles m1) |
| **Pass barrier** | 2 barriers (GM, SD) | 1 barrier (GM) |
| **Memory** | O(M×P) for QK, SN | O(M0×P) per tile |
| **Operations** | 6 | 13 (more compute) |
| **Trade-off** | Simple, memory intensive | Complex, memory efficient |

## Benefits of 2-Pass

1. **Memory Efficiency**: Only need to store one tile of attention scores at a time
2. **Reduced Barriers**: One major pass barrier (at GM) instead of two
3. **Scalability**: Works with arbitrarily long sequences (M can be very large)
4. **Simpler V handling**: V stays untiled, final matmul is standard

## Memory Footprint

**3-Pass:**
```
QK_{m,p}: M × P floats
SN_{m,p}: M × P floats
Total: 2 × M × P
```

**2-Pass:**
```
BQK_{m1,m0,p}: M0 × P floats (one tile at a time)
LM_{m1,p}: M1 × P floats (small, M1 = M/M0)
CN_{m1,m0,p}: M0 × P floats (one tile at a time)
Total: ~M0 × P (much smaller if M0 << M)
```

For M=1024, M0=256 (4 tiles):
- 3-Pass: 2 × 1024 × P
- 2-Pass: ~256 × P (4x reduction!)
