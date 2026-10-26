# A13 — Exact symbolic refinement & rational certification

**Priority: medium. Does not lower `L`; it protects the score and gives
provenance.**

## Why

Two separate benefits:

1. **Certification safety.** A numerically "touching" packing is undecidable in
   interval arithmetic. Replacing it with an exact rational witness (or with a
   provably `1e-12`‑separated float layout) guarantees the puzzle is not
   silently scored as the grid.
2. **Provenance / tie‑breaking.** When two sources tie on `L`, prefer the one
   with an exact certificate — it is the one that will still certify after the
   decimal round trip, and it is defensible.

It also *reveals structure*: the contact equations often have algebraic
solutions, and knowing `L` exactly (as a minimal polynomial) sometimes shows
that a slightly different contact pattern gives a better root.

## Method

### 13.1 Read the contact structure
From a converged layout, take every pair with `|pen| < 1e-8` and every wall
contact. Each contact `(i,j,a)` gives one equation:

```
a(θ_i, θ_j)·(c_i − c_j) = R(θ_i − θ_j)
```

with `a` the separating axis. Unknowns: `2n` centres + `n` angles + `L`.
Symmetry: fix translation (2 dof) and, optionally, one global rotation.

### 13.2 Square the system
Count equations vs unknowns:
* square system → Newton in high precision, unique solution;
* underdetermined → add the **Fritz‑John / Jacobian‑determinant condition**

```
Det[ Grad( {s, f₁, f₂, …}, {s, a, b, …} ) ] = 0
```

which selects the stationary point of the constrained system (Ellsworth's
device). Then Newton.

### 13.3 Solve at high precision
`mpmath` with `mp.dps = 200…500`, or `msolve`/`Singular` for the polynomial
system. Verify the residual is `< 10^-150`.

### 13.4 Recover the algebraic number
`mpmath.pslq` / `algdep` / `RootApproximant` on `L` to find the minimal
polynomial. Degrees observed in the literature: 8 (`n=11`), 18 (`n=17`),
40 (`n=300`), 62 (`n=1453`).

### 13.5 Write the witness
Either
* **rational** (`Fraction` with a common denominator `Q`), validated exactly
  (`pen_ij ≤ 0` in `Fraction` arithmetic, no dilation), or
* **float** with a `1e-12` dilation and a proof that the interval metric passes.

## Expected gain

* Score: `0` directly, but it can *recover* up to `0.1` by preventing
  undecidable puzzles.
* Indirect: exact `L` values let the merge compare sources without float noise,
  and the algebraic structure occasionally suggests a better pattern.

## Cost

Medium. High‑precision Newton is seconds per puzzle; `algdep` is fast. The hard
part is the contact‑equation bookkeeping.

## Risks

* Over‑determined or inconsistent systems (a contact that is really a
  near‑contact) → drop contacts with the smallest residuals and retry.
* `algdep` false positives — always verify the candidate polynomial by
  evaluating it at 1000 dps.

## Plan

* `src/exact/contacts.py` — extract equations from a layout.
* `src/exact/newton.py` — mpmath Newton with the Jacobian‑determinant closure.
* `src/exact/algdep.py` — minimal polynomial + verification.
* `src/exact/witness.py` — rational writer + exact validator.

## Decisions needed from you

1. Do we invest in this now, or only if A01's dilation proves insufficient on
   real submissions? (I lean "later": dilation is cheaper and sufficient for
   scoring.)
2. Should exact witnesses be stored alongside every improved puzzle (nice for
   the write‑up, costs time)?
