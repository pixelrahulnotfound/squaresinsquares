# A06 — Smooth‑SAT penalty L‑BFGS quench

**Priority: high. Cheapest per‑point gain in the whole project; every other
search method ends with this.**

## Why

Given a layout, the last `1e-4…1e-3` of `L` is pure numerical polishing: push
squares until contacts are exact, then shrink the box. The trick that makes it
work is that non‑overlap is **non‑smooth** (max over 4 axes, `|·|`, `cos` at
kinks), and a naive gradient method stalls. The fix is a *continuation*: solve a
smooth surrogate, then tighten the smoothing until it is exact.

`malyshevdanil` ships the reference implementation (custom batched CUDA L‑BFGS
with `M=8`, Armijo `c₁=1e-4`, ≤24 halvings); we port it to CPU/numba.

## Energy

For container side `L`, centres `c`, angles `θ`:

```
pen_ij = R(θ_i − θ_j) − max_{a ∈ axes(i,j)} |(c_i − c_j)·a|
E(c,θ) = Σ_{i<j} φ(pen_ij)²  +  λ Σ_i Σ_{walls} φ(wall_i)²
```

with `φ(z) = max(0, z)` and `λ ≈ 25`. Note `E = 0` ⟺ feasible.

## Smoothing (the essential part)

```
|z|      →  √(z² + τ²) − τ
max(a,b)→  ½(a + b + √((a−b)² + τ²))
cos x    →  √(cos²x + τ²) − τ + 1     (keeps sign, C¹ at 0)
```

Schedule `τ: 1e-3 → 1e-5 → 1e-8 → 0` (exact). At each stage run L‑BFGS to
`‖g‖∞ < 1e-10` or a step budget; warm‑start the next stage.

## Gradient

With `a*` the argmax axis and `s = sign((c_i−c_j)·a*)`:

```
∂E/∂c_i =  2 φ(pen) · (−s a*)          (smoothed via the max surrogate)
∂E/∂c_j =  2 φ(pen) · (+s a*)
∂E/∂θ_i =  2 φ(pen) · R'(δ),  R'(δ) = ½(−sinδ·sgn(cosδ) + cosδ·sgn(sinδ))
```

`sgn(cosδ)` and `sgn(sinδ)` are themselves smoothed by the same `√(x²+τ²)`
trick. **Always verify against finite differences** at `τ = 1e-4` — a wrong
angle gradient silently degrades every result.

## The squeeze loop

```
repeat:
    minimise E at current L          # quench
    if E > tol: L += η; continue     # relax
    L -= η;  if E > tol: L += η; η *= 0.7   # bisection on the feasible boundary
until η < 1e-9
```

## Expected gain

* Per puzzle: `1e-4…1e-3` in `L`, i.e. `1e-5…3e-4` in score.
* Over 400 puzzles: **0.005–0.02**.
* Also the inner loop of A07/A08/A10, so its quality compounds.

## Cost

Low on CPU: `3n` variables, `O(n²)` pairs; `n = 1000` → 500k pairs per
evaluation. Use a **spatial hash / cell list** so only near pairs are evaluated
(`malyshevdanil` and `lightningv08` both do this; it is worth ~30×).

## Risks

* Non‑convex: the quench converges to a *solid* configuration (no single move
  helps) but not necessarily a good one. Always pair with A07.
* `λ` too high → squares crushed together; too low → box not respected.
  Tune per `n`.
* Float32 is fine for proposing, **never** for the final coordinates.

## Plan

* `src/energy.py` — numba kernels: `penetration_matrix`, `energy`, `gradient`,
  spatial hash. Finite‑difference test in CI.
* `src/quench.py` — continuation schedule + L‑BFGS‑B (`scipy`) and a batched
  numba variant for many starts.
* `src/squeeze.py` — the box‑shrinking loop with bisection.

## Decisions needed from you

1. `scipy.optimize.minimize(method="L-BFGS-B")` (simple, slow per call) or a
   custom batched numba L‑BFGS (10–50× faster for many starts)? I recommend the
   custom one, reusing the existing CUDA design.
2. Should the quench optimise `L` **directly** (soft‑min over the box) or use the
   squeeze loop? The squeeze loop is more robust; soft‑min is faster.
