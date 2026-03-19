# Homomorphic Ranking via SIMD Matrix Construction

## Problem

Given a plaintext vector **v** = (v₀, v₁, ..., v_{n-1}) encrypted under CKKS, compute
for each element its **rank from the bottom**:

```
rank[k] = |{ j : v[j] < v[k] }|
```

without decrypting.

---

## Core Idea: Single-Evaluation Comparison

The key insight is to avoid O(n²) separate ciphertext comparisons by constructing two
n×n comparison matrices inside a single ciphertext using SIMD slot packing, then
evaluating the comparison function once across all n² slots in parallel.

The n² slots of the ciphertext are treated as a row-major n×n matrix. Two matrices
are built:

```
vR[k][j] = v[j]   (each row is a copy of the full vector)
vC[k][j] = v[k]   (each column is a copy of the full vector)
```

The difference matrix `D[k][j] = vC[k][j] - vR[k][j] = v[k] - v[j]` is positive
exactly when `v[k] > v[j]`, which is what we need to count.

---

## Algorithm Steps

### Step 1 — Build vR: `replicateRow`

**Input:** `[v₀, v₁, ..., v_{n-1}, 0, ..., 0]`

Uses ⌊log₂ n⌋ doublings. At each step `i = n/2, n/4, ..., 1`, the current
accumulator is right-rotated by `i·n` slots and added back to itself. After
⌊log₂ n⌋ steps the first n² slots hold the periodic repetition of the input row.

For n=4: shifts −8, −4 → 2 Galois keys, 2 additions.

**Galois keys required:** `{−n, −2n, ..., −(n/2)·n}` (⌊log₂ n⌋ keys)

---

### Step 2 — Build vC: `transposeRowToColumn` + `replicateColumn`

**`transposeRowToColumn`** places `v[k]` at slot position `k·n` ("first column")
using the algorithm from Halevi & Shoup, "Bootstrapping for HElib" (2015), §2.3 Alg. 1:

```
for i = 1 to ⌈log₂ n⌉:
    X ← X + (X ≫ n(n−1)/2^i)
X ← MaskC(X, 0)           // zero all slots except k·n
```

The ⌈log₂ n⌉ right-rotations scatter the n input values to the diagonal positions
`{0, n, 2n, ..., (n−1)n}`. The mask selects only those positions, discarding all
cross-contamination. This costs 1 `multiply_plain` + 1 `rescale` (depth +1).

**Galois keys required:** `{−n(n−1)/2, −n(n−1)/4, ..., −⌈n(n−1)/n⌉}`

**`replicateColumn`** then spreads each isolated `v[k]` at slot `k·n` across all
slots in its row (`k·n` to `k·n+n−1`) by applying n−1 independent right-rotations
by 1, 2, ..., n−1 and summing:

```
vC = Σ_{i=0}^{n-1} RightRot(masked, i)
```

**Galois keys required:** `{−1, −2, ..., −(n−1)}` (n−1 keys)

---

### Step 3 — Difference Matrix

```
ct_diff = ct_col − ct_row
```

A single ciphertext subtraction. No level consumed.

---

### Step 4 — Sign Approximation: `sign_approx_deg3`

The sign function is not a polynomial, so it is approximated. The implementation uses
the **degree-3 Chebyshev-minimax approximation** on [−1, 1]:

```
f(x) = (3/2)x − (1/2)x³
```

This is evaluated using a standard squaring chain:

| Operation                            | Depth cost |
|--------------------------------------|------------|
| x² = x·x, relin, rescale            | +1         |
| x³ = x·x², relin, rescale           | +1         |
| 1.5·x via multiply_plain, rescale   | +1         |
| **Total**                            | **3 levels** |

**Accuracy constraint.** `f(x) ≈ sign(x)` only when |x| is bounded away from 0.
For the decoded rank error to be < 0.5 (correct integer rounding), all non-zero
pairwise differences must satisfy:

```
|(1 + f(d_ij)) − indicator(v[k] > v[j])·2| < 0.5   for all k ≠ j
```

With the degree-3 polynomial this is achievable when the minimum non-zero pairwise
difference ≥ 1/3 and the input range spans [0, 1] exactly (so the maximum difference
hits 1.0 where `f(1) = 1` exactly). For a uniform n-element set v[i] = i/(n−1)
this requires **n ≤ 4**.

For larger n, a higher-degree composite approximation is needed (e.g., the iterative
minimax approach of Han & Ki, or the Remez-optimal degree-15/27 polynomials used in
HEAAN bootstrapping).

---

### Step 5 — Shift to Counting Domain

```
ct_indicator = ct_sign + 1.0        (plaintext addition, free)
```

This maps `f(x) ∈ (−1, 1)` to the range `(0, 2)`. Each slot now contributes:
- ≈ 2 when `v[k] > v[j]`
- ≈ 1 when equal (self-comparison, `j = k`)
- ≈ 0 when `v[k] < v[j]`

The ÷2 normalization is deliberately omitted here because at depth 4 (the maximum for
this parameter set) a `multiply_plain` would consume the last available modulus.
Instead, the final sum is decoded as:

```
rank[k] = (sum[k] − 1) / 2
```

---

### Step 6 — Row Summation: `sumRows`

A logarithmic left-rotation tree accumulates the n columns of each row into position
`k·n`:

```
for step = 1, 2, 4, ..., n/2:
    result += LeftRot(result, step)
```

After ⌈log₂ n⌉ steps, slot `k·n` holds `Σ_j ct_indicator[k][j] ≈ 2·rank[k] + 1`.

**Galois keys required:** `{+1, +2, +4, ..., +n/2}` (⌊log₂ n⌋ keys)

---

## Level Budget (N=8192, 128-bit security)

| Phase                                    | Depth     |
|------------------------------------------|-----------|
| `transposeRowToColumn` (mask multiply)   | 0 → 1     |
| `sign_approx_deg3`                       | 1 → 4     |
| **Total**                                | **4 of 4** |

Parameters: `{40, 30, 30, 30, 30}+{40}` = 200 bits ≤ 218-bit security limit for
N=8192 at 128-bit security (lattice-estimator). `Q_size = 5` → max depth = 4.

---

## Complexity

| Resource                  | Cost                                       |
|---------------------------|--------------------------------------------|
| Ciphertext multiplications | O(log n) for sign polynomial              |
| Rotations / Galois keys   | O(n + log n), dominated by replicateColumn |
| Ciphertexts               | O(1) — all operations in-place            |
| HE depth                  | O(1) — fixed 4 levels regardless of n     |

All `replicateColumn` rotations are data-independent and can be dispatched to separate
CUDA streams in parallel (demonstrated in `17_ckks_comparison.cpp`).

---

## Literature Cross-Reference

| Component                        | Reference |
|----------------------------------|-----------|
| Row encoding / shift-and-add doubling | Halevi & Shoup, "Algorithms in HElib", CRYPTO 2014 |
| `transposeRowToColumn` shift schedule `n(n−1)/2^i` | Halevi & Shoup, "Bootstrapping for HElib", 2015, §2.3 Alg. 1 |
| Degree-3 sign polynomial `3/2·x − 1/2·x³` | Standard Chebyshev approximation; see also Cheon et al., "Efficient Homomorphic Comparison Methods with Optimal Complexity", ASIACRYPT 2020 |
| Higher-degree / composite sign for large n | Han & Ki, "Better Bootstrapping for Approximate Homomorphic Encryption", CT-RSA 2020 |
| CKKS scheme foundation | Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers", ASIACRYPT 2017 |
