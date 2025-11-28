# Tiling Operation Notation

## New Explicit Syntax

### T_split_m_m1m0
**Format**: `T_split_<source>_<result_dimensions>`

**Meaning**: Split dimension `m` into `m1` (tiles) and `m0` (within tile)

**Example**:
```egglog
BK_{e,m1,m0} = T_split_m_m1m0(K_{e,m}, 4)
```
- Input: `K_{e,m}` - 2D tensor
- Output: `BK_{e,m1,m0}` - 3D tensor (m is split into m1 × m0)
- Parameter: `4` - number of tiles (m1 = 4)

**Transformation**:
```
K_{e,m} where m ∈ [0, M)
  ↓
BK_{e,m1,m0} where m1 ∈ [0, M1) and m0 ∈ [0, M0)
  with M = M1 × M0
  and BK_{e,m1,m0} = K_{e,m1×M0+m0}
```

### T_unsplit_m1m0_m
**Format**: `T_unsplit_<source_dimensions>_<result>`

**Meaning**: Combine dimensions `m1` and `m0` back into `m`

**Example**:
```egglog
A_{m,p} = T_unsplit_m1m0_m(A_tiled_{m1,m0,p})
```
- Input: `A_tiled_{m1,m0,p}` - 3D tensor with tiled dimension
- Output: `A_{m,p}` - 2D tensor (m1 and m0 collapsed into m)

**Transformation**:
```
A_tiled_{m1,m0,p} where m1 ∈ [0, M1), m0 ∈ [0, M0)
  ↓
A_{m,p} where m ∈ [0, M)
  with A_{m,p} = A_tiled_{m1,m0,p} where m = m1 × M0 + m0
```

## Why This Notation is Better

### Old Notation (Ambiguous)
```egglog
T_split_m      ; Split m... into what?
T_unsplit_m    ; Unsplit what into m?
```

### New Notation (Explicit)
```egglog
T_split_m_m1m0     ; Split m into m1 and m0
T_unsplit_m1m0_m   ; Combine m1 and m0 into m
```

## Pattern

```
T_<operation>_<from>_<to>
```

**Split**: `from` is single dimension, `to` is multiple dimensions
```
T_split_m_m1m0    ; m → (m1, m0)
```

**Unsplit**: `from` is multiple dimensions, `to` is single dimension
```
T_unsplit_m1m0_m  ; (m1, m0) → m
```

## Benefits

1. **Self-documenting**: Immediately clear what transformation is happening
2. **Symmetric**: Split and unsplit are clearly inverses
3. **Extensible**: Could support other tiling patterns:
   - `T_split_m_m1m2m0` - Split into 3 levels
   - `T_split_mp_m1m0p1p0` - Tile both m and p dimensions
4. **Type-safe**: The dimension names in the operator match the tensor shapes

## Usage in 2-Pass Attention

```egglog
; Original tensors
K_{e,m}        ; Key tensor

; Tile K
BK_{e,m1,m0} = T_split_m_m1m0(K_{e,m}, 4)

; ... compute attention weights in tiles ...

; Untile A
A_tiled_{m1,m0,p}  ; Tiled attention weights
A_{m,p} = T_unsplit_m1m0_m(A_tiled_{m1,m0,p})

; Final multiply with V
AV_{f,p} = A_{m,p} @ V_{f,m}
```

This makes the tiling structure completely explicit in the code!
