# A04 — Fault‑line / gadget DP composition

**Priority: high. This is the only *structural* lever that makes one discovery
pay off across dozens of puzzles at once.**

## Why

Write `n = K² + D`, box side `L = K + f`, `0 < f < 1`. Put an integer `K×K`
lattice at the origin and a second one shifted by `(f, f)`. They meet along a
monotone staircase. Each side is *integer*, so it is trivially valid; the only
freedom is the staircase, and a staircase is a short sequence of **gadgets**.

A gadget `(w, h, Δm)` is a `(w+f)×(h+f)` block holding `w·h + Δm` squares
(`Δm ≥ 0` is the surplus over the integer grid). If you find a gadget with
`Δm = 1` and small `f`, you have improved *every* `n` whose decomposition
contains that gadget — often 20–60 puzzles at once.

`cdeotte` (22.3116) and `alexchilton` (22.3079) both reach their score mainly
through this mechanism; `lightningv08` adds a GPU search over `31·31·63` gadgets.

## Method

### State

Process the box column by column. A column `c` is split by the staircase into
an `A` part (below/left, lattice at origin) and a `B` part (above/right, lattice
shifted by `(f,f)`). Track:

```
(cols_done, rowsA, rowsB, surplus, tip_type)
```

* `rowsA`, `rowsB` — how many integer rows are consumed on each side,
* `surplus` — `Σ Δm` so far,
* `tip_type` — shape of the last gadget (the staircase must stay monotone).

### Transition

A gadget occupying `w` columns with height profile `h_A(1..w)`, `h_B(1..w)`:

```
rowsA += Σ h_A,  rowsB += Σ h_B,  surplus += Δm,  cols += w
```

Feasibility of the chain: at every column, `rowsA + rowsB ≤ K` (the two lattices
must not collide across the staircase), and the profile must be monotone.

### Objective

For a target `n = K² + D`, find the chain minimising `f` subject to
`Σ (w·h + Δm) ≥ D`, then `L = K + f`. Equivalently: maximise `D` at fixed `f`.
Because `f` enters every gadget identically, the DP is over `(rowsA, rowsB)` and
returns, for each `f` on a grid (`Δf = 1e-3`), the maximum achievable surplus.
Monotone in `f`, so binary search on `f` per `n`.

### Where gadgets come from

1. **Crops of good packings.** Take a certified packing at side `L = K+f`,
   subtract the two lattices, and the remainder *is* a gadget. `alexchilton`
   enumerates crops of Couzo witnesses and of basin‑hopped layouts.
2. **Local search in gadget space.** For a fixed `(w, h)`, optimise the
   `(w·h+1)` squares' positions inside `(w+f)×(h+f)` with A06/A07 — a tiny
   `3·(wh+1)`-dimensional problem, thousands of times cheaper than a full `n`.
3. **MILP refill** (A05): fix the tilted part, fill the rest upright exactly.

### Reconstruction

A chain → coordinates: place `A` squares on the origin lattice, `B` squares on
the shifted lattice, offset each gadget by its cumulative `(cols, rowsA)`.

## Expected gain

* Public implementations: `cdeotte` 0.06 of its total came from refill on
  gadget crops; `alexchilton` reports the DP pass alone worth ~0.002–0.01.
* With **searched** gadgets (rather than crops) the upside is much larger: a
  `Δm=1` gadget at `f = 0.30` for `w,h ≈ 6–10` would improve tens of large‑`n`
  puzzles by `1e-3…1e-2` each.
* Realistic target: **0.02–0.06**.

## Cost

Medium. The DP itself is milliseconds; the gadget search is the cost and it is
embarrassingly parallel (each `(w,h)` independent).

## Risks

* The staircase must be genuinely monotone; a sloppy reconstruction produces
  overlaps at the seam (this is exactly what `kirilltym`'s "seam LP" repairs).
* Gadgets found at one `f` may not be achievable at the smaller `f` needed for a
  specific `n`; keep the `f → max surplus` curve, not just the best gadget.
* Diminishing returns if the public portfolio already used the same crops.

## Plan

* `src/gadget/dp.py` — the chain DP, `f`‑sweep, reconstruction.
* `src/gadget/mine.py` — crop enumeration from any packing source.
* `src/gadget/search.py` — per‑`(w,h)` gadget optimiser (calls A06/A07).
* `src/gadget/seam.py` — post‑composition seam LP (shares code with A08).

## Decisions needed from you

1. Should the DP optimise **`L` per `n`** or **a single `f` per `K`** (simpler,
   more reusable)? I recommend per‑`K` `f`, then per‑`n` fine‑tune with A08.
2. Budget for gadget search: how many CPU‑hours? This is the main knob.
3. Do we allow **multi‑step staircases with two different `f` values** (more
   general, harder DP)? Public code uses a single `f`.
