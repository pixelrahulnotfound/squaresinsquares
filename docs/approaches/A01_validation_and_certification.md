# A01 — Validation & certification infrastructure

**Priority: must‑have (prerequisite for everything else).**

## Why

A puzzle that fails the official checker is scored as the *grid*, not as zero.
A single broken puzzle at `n = 900` costs `30/30 − L/30 ≈ 0.1`, i.e. more than
the entire gap between public portfolio and LB #1. And the failure mode is
silent: the file looks fine, `L` is small, the metric says "overlap undecidable".

Three independent traps:

1. **Exact touching is undecidable** in interval arithmetic at 500 dps. Optimal
   packings touch; written naively they are rejected.
2. **Decimal round‑trip.** `L` is measured at 60 dps from the strings you write.
   `%.6f` moves a corner by 5e‑7, which can both inflate `L` and create
   overlaps.
3. **Radians vs degrees.** Half the public notebooks work in degrees.

## Method

A four‑stage gate, run on the *final CSV bytes* before upload:

```
stage 1  schema      : 1000 puzzles, ids contiguous, n rows each, finite floats
stage 2  float SAT   : 4 axes per pair, penetration < 1e-9  (fast, numba)
stage 3  dilation    : f = max over near pairs of (R(δ)+clr)/max|proj|
                       if 0.999 < f < 1.02: rescale centres about bbox centre
stage 4  official    : metric/squares_in_squares_metric.py at 50 dps
                       -> every puzzle must be strictly feasible, L must match
```

Stage 4 is the only one that counts; stages 2–3 exist to make stage 4 pass
without escalating to 500 dps (which is ~50× slower).

## Math

For squares `i, j` with centres `c_i, c_j`, angles `θ_i, θ_j`, `δ = θ_i − θ_j`:

```
R(δ)      = ½ (1 + |cos δ| + |sin δ|)          # sum of half-extents along the
                                                 # separating axis, in units of
                                                 # the projection of the unit
                                                 # edge onto that axis
axes(i,j) = {u_i, v_i, u_j, v_j},  u = (cosθ, sinθ), v = (−sinθ, cosθ)
pen_ij    = R(δ) − max_{a∈axes} |(c_i − c_j)·a|
feasible  ⟺ pen_ij ≤ 0  ∀ i<j,  and  |c_i·e| + h_i ≤ L  ∀ i, e ∈ {x,y}
```

`R` is the correct threshold because on the *fixed* axis `a` the two squares'
projections are intervals of half‑width `|a·u_i|/2` and `|a·u_j|/2`; their sum
is `(|cosδ|+|sinδ|)/2 + 1/2`. Using only the two axes of one square (a common
bug — including in my first version of `tools/score.py`) under‑detects overlaps
when the other square's edge is the better separator.

## Dilation

Rescale every centre about the layout centroid:

```
c_i ← c̄ + s·(c_i − c̄),  s = 1 + ε,  ε ∈ {1e-12, 1e-11, 1e-10, 1e-9}
```

`L` grows by `≈ 2ε·max|c − c̄|`, i.e. `1e-12` → `~1e-12` in score. Free.
Choose the smallest `ε` that passes stage 4; if none does, the packing is
genuinely invalid.

## Expected gain

No direct score; it protects every other approach. Empirically, careless
pipelines lose 0.05–0.3 to undecidable puzzles.

## Cost

Low. `tools/score.py` already does stages 2–4 (`--official`); stage 3 needs
wiring into the writer.

## Plan

* `src/io.py` — read/write CSV with `repr`‑precision, per‑n dict of arrays.
* `src/geom.py` — `penetration_matrix`, `axes`, `bbox_L`, numba kernels.
* `src/validate.py` — `gate(csv_path) -> (ok, report)`; calls the official
  metric in a subprocess so a metric change cannot break our code.
* `tools/score.py` gains `--gate` which runs 1→4 and refuses to pass a file.

## Decisions needed from you

1. Do you want **exact rational certificates** (mpmath `Fraction`) alongside
   floats, or is "dilate + official metric" enough? (Cheaper: dilation.)
2. Accept a fixed clearance `1e-9` everywhere, or per‑puzzle minimal `ε`?
