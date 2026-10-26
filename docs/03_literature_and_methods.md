# 03 — Literature and method survey

Scope: everything that can move `s(n)` for unit squares in a square, plus
transferable machinery from adjacent packing problems. Citations are the real
papers; the two 2026 arXiv IDs are from the live research record.

---

## 1. The problem itself

**E. Friedman, "Packing Unit Squares in Squares: A Survey and New Results"**,
*Electronic Journal of Combinatorics*, DS7 (1998, rev. 2009).
The canonical reference: history, best known packings and lower bounds for
`n ≤ 100`, proofs for `n = 2,3,5,7,8,14,15,24,35`, and the **unavoidable point
set** method for lower bounds.

**Erdős & Graham (1975)** — wasted area `W(s) = s² − max{n : s(n) ≤ s}`;
constructive `W(s) = O(s^{7/11})`. Improvements: Chung–Graham `O(s^{0.631})`,
then `O(s^{0.625})`, `O(s^{0.6})`. **Roth & Vaughan**: `W(s) ≥ 10^{-100}√(s·|s −
round s + ½|)`, so `W` is not `O(s^α)` for `α < ½`.

**Proven values.** `s(k²) = k`; `s(k²−1) = s(k²−2) = k` (Nagamochi 2005; the
`k²−1` case re‑proved by Karakuş 2026, `k²−2` by chelokot's Lean proof after
Nagamochi's Lemma 1 was found false in 2026); `s(k²−3)` for k ≤ 7 (Kearney–Shiu
2002 for n=6; Bentz 2010 for 13, 46; Bentz 2016 for 22, 33); `s(5) = 2 + 1/√2`
(Göbel 1979); `s(10) = 3 + 1/√2` (Stromquist 2003). `s(11) = 3.87708359…`
(Trump 1979) had an AI‑assisted optimality proof announced 2026‑09‑29.

**Record table:** D. Ellsworth's continuation of Friedman's Packing Center —
all `n ≤ 324` plus selected larger cases, as SVGs with 33‑digit coordinates and
(minimal polynomial, Mathematica source) for the analytically solved ones.
**Above n = 324 there is no published table** — this matters enormously for the
competition (see `01_score_anatomy.md`).

---

## 2. Search algorithms that actually produced records

### 2.1 Perturbed billiard / inflation — Gensane & Ryckelynck (DCG 2005)

The first method that worked for squares. Links `s(n)` to the maximal
*inflation* `ω(C)` of a configuration and maximises it with a stochastic
billiard in four layers:

1. **Random walking** — pick one square, propose a displacement uniform in a
   ball of radius ε, project back into the container, accept only if no overlap.
2. **Stochastic billiard** — run (1); double ε on improvement, halve otherwise;
   stop at ε₂.
3. **Perturbation** — displace *all* squares simultaneously by ε.
4. **With perturbations** — alternate (3) and (2), restore on failure, shrink ε.

Layer 3 is the key: (2) converges to *solid* configurations (no single move
helps) but not to local optima; simultaneous kicks let the search walk along the
solid manifold. Results: improved n = 11, 29, 37; alternative optimum n = 18.

**Why it matters here:** it is the cleanest "feasible‑only" global search, and it
is trivially parallel. It is also the direct ancestor of the `repair`‑gated
moves used by every top Kaggle notebook.

### 2.2 GPU simulated annealing — Schadt / Ellsworth (current record engine)

Closed source, but the published basin statistics are the most useful data in
the field: for `s(51)`, 3004 basins were found and **only 4** refine to the
record; 23.6 s per basin ⇒ ~4.9 GPU‑hours per hit. For `s(55)`: 1893 basins,
record found 3 times, 2.6 s per basin.

Lesson: the objective has thousands of near‑equal local optima and the record is
a needle. **Search budget, not cleverness, is the binding constraint** at small
and mid `n`. Also: seeds matter — records are found from neighbours with squares
removed/straightened.

### 2.3 Basin hopping — Wales & Doye (J. Phys. Chem. A 1997)

Fixed container side, minimise overlap with L‑BFGS, perturb a few squares,
Metropolis accept, and **squeeze** the box whenever feasible. Implemented
open‑source by `evand/square-packing` (Rust, MIT) and used by
`alexchilton`/`malyshevdanil` in this competition.

### 2.4 Simulated annealing / threshold accepting / extremal optimisation

* **SA** — Kirkpatrick et al. 1983; for packing: Zhang & Huang (2004) for
  circles. Kaggle versions: single‑square Gaussian moves, rigid‑body subset
  moves, integer overlap count as energy.
* **Threshold accepting** — Dueck & Scheuer (1990): accept if `ΔE < T`, no
  probability. Cheaper than Metropolis, often equal quality on rugged discrete
  landscapes. Untried here — cheap experiment.
* **Extremal optimisation** — Boettcher & Percus (2001): rank variables by
  fitness, always mutate the worst. Natural fit: mutate the square with the
  largest penetration contribution. Untried here.
* **Great deluge / late acceptance hill climbing** — due to Burke & Bykov
  (2017); trivially better than plain SA on some packing instances.

### 2.5 General‑purpose global optimisation — Berthold, Kamp, Mexi, Pokutta,
Pólik (arXiv:2601.05943, arXiv:2605.04850)

SCIP 10 + FICO Xpress 9.8, Farkas‑lemma non‑overlap (per pair, non‑negative
multipliers whose combination of edge normals is equal and opposite — a
separating axis recovered as a solver variable), 5 000 s multistart + 10 000 s
solve, 48 cores. On squares‑in‑squares: matches records only to `n ≈ 16`, then
falls behind (n=27 returns Göbel's 28‑square arrangement, n=29 returns the
trivial grid). Their own diagnosis: pairwise constraints grow quadratically;
they propose **lazy separation of violated constraints**.

**Why it matters:** do not expect a MIP solver to win at n > 20. But the
Farkas/phi‑function formulation is the right way to build a *small‑n exact*
model, and lazy constraint generation is directly applicable to our LP/MILP.

### 2.6 AlphaEvolve and the LLM‑evolution ecosystem

**Novikov et al., AlphaEvolve (2025)** — LLM‑generated‑code evolutionary search;
improved 11 hexagons in a hexagon (3.931 vs 3.943) by *tilting inner pieces at
varying angles*. Replications: OpenEvolve, ShinkaEvolve, CodeEvolve; successors
ThetaEvolve (2511.23473), FM Agent (2510.26144), ImprovEvolve (2602.10233),
Helix (2603.07642), SeaEvo (2604.24372), flow‑based structure discovery
(2601.18005).

**Critically: no AlphaEvolve‑class system has ever been applied to
squares‑in‑squares.** In the OpenEvolve replication the LLM converged on writing
"SciPy SLSQP from multiple starts" — i.e. it rediscovered local search.

**Why it matters:** an LLM‑evolution loop over *move operators / constructions*
is a genuine, unexplored lever here, and it is cheap to try on CPU. It will not
beat a GPU annealer on raw throughput, but it may discover a construction family
nobody has written down.

### 2.7 Interval branch and bound (optimality proofs)

Montanher, Neumaier, Markót, Domes, Schichl (J. Global Optim. 2018): the only
rigorous computer‑assisted result for rotatable squares — 3 squares in a circle,
`r₃ ∈ [1.28847050800547, 1.28847050800553]`, 628 s for the decisive subproblem.
Markót & Csendes (SIAM J. Optim. 2005) proved optimality for 28–30 circles in a
square in ~50 CPU‑hours; Markót (2021) for 31–33 in 13–61 CPU‑hours.

**Why it matters:** proofs are out of scope for the competition (we only need
upper bounds), but the *machinery* — interval SAT, sentinels, tiling for
symmetry — is what makes our certification safe.

---

## 3. Adjacent machinery worth importing

### 3.1 VLSI floorplanning / rectangle packing representations

The composition problem ("tile a `K×K` square with rectangular blocks") is the
classic floorplanning problem:

* **Slicing trees / guillotine cuts** — Otten (1982). A DP over rectangles cut
  by full‑width/height lines. Exactly the structure of the "L‑cut" and
  "rectangular reclosure" constructions used by `avazaku`/`cdeotte`.
* **Sequence pair** — Murata et al. (1996): any non‑slicing floorplan is
  representable by two permutations; O(n log n) evaluation. A different
  neighbourhood than slicing trees, and it covers non‑guillotine layouts.
* **Corner Block List** — Hong et al. (2000); **O‑tree**, **BSG**. All give
  move sets (`swap`, `rotate`, `move`) on a compact encoding — a much better
  global move set than perturbing coordinates.
* **`packingsolver`** (C++, MIT): open‑source rectangle/guillotine/box solver
  with tree search and column generation. Directly reusable for the
  "tile the integer part of the box with blocks" subproblem.

### 3.2 Nesting / collision detection

**`jagua-rs`** (Rust, MPL‑2.0; INFORMS J. Computing): quadtree + hazard proximity
grid, continuous rotation, millions of collision queries per second. The right
geometry backend if we build a custom searcher in Rust. **`sparrow`** is the
nesting optimiser on top; wrong objective (strip packing), but its move set and
its "explore/exploit item order" logic transfer.

### 3.3 Metaheuristics for continuous non‑convex problems

* **CMA‑ES** — Hansen (2006). Handles `3n`-dimensional, ill‑conditioned,
  non‑separable objectives; used widely for circle packing. Expensive per
  generation but excellent at the final squeeze.
* **Differential evolution** — Storn & Price (1997). Cheap, embarrassingly
  parallel, good for the "shrink the box" phase.
* **L‑BFGS‑B with exact gradients** — the workhorse; `malyshevdanil` shows the
  correct smoothing schedule.
* **Lovász / S‑lemma relaxations** — for containment (Berthold et al. 2026).

### 3.4 Large neighbourhood search (LNS)

Shaw (1998); Ropke & Pisinger (2006). Ruin a region (remove `k` squares),
re‑optimise it exactly (small MILP or exhaustive placement), keep if better.
For packing, "ruin" = delete the squares touching a random disc/seam; "recreate"
= refill by exact upright MILP (`fill_rect`) or by LP. This is exactly what
`cdeotte`'s FTMC + refill does, but it can be made systematic.

### 3.5 Learning‑based layout

* **Fang, Rao, Zhao, Du** — *Mathematics* 11(2):327 (2023): hybrid MC/Q/Sarsa for
  2D irregular packing.
* **Huang, Wang, Zhou, Lu** — arXiv:2211.09382, IEEE RA‑L: hierarchical RL,
  heightmap observations, sequence + placement.
* **Crescitelli & Oshima** — IJSC (2024): DDQN with dense reward.
* **Abdou, Ibrahim, Binder, Huber** — *J. Intelligent Manufacturing* (2025):
  vision‑based DRL for rotational placement; 97 % faster, 11 % better
  utilisation than open‑source nesting.
* **Lehuger & Henon‑Just** — arXiv:2606.10611: **Polygons Transformer (PoT)**,
  geometry‑aware encoder + CORL training, competitive with Sparrow.
* **DIFUSCO** — Sun & Yang, NeurIPS 2023: discrete diffusion for combinatorial
  optimisation (TSP, MIS). A diffusion model over *placement/ordering* is a
  plausible generative initialiser here.

**Honest assessment:** all published learning results are for *sequence/placement
selection* in bin/strip packing, where a constructive policy is natural. Our
problem is a pure continuous non‑convex minimisation with no natural sequence.
The realistic uses of ML here are (a) a **learned improvement operator**
(predict which squares to kick, à la learned LNS), (b) a **surrogate** to rank
candidate basins cheaply, (c) **LLM code evolution** of constructions. All three
are cheap to test and none has been tried publicly.

---

## 4. Symbolic refinement (useful for provenance, not score)

Ellsworth's pipeline: read the contact structure off a numerical solution →
write one polynomial per contact → if underdetermined, add the
**Jacobian‑determinant condition** `Det[Grad[{s,f₁,f₂,…},{s,a,b,…}]] = 0`
(Fritz‑John first‑order condition in determinant form) → `FindRoot` to hundreds
of digits → `RootApproximant`/`algdep`/PSLQ to recover the minimal polynomial.

Degrees observed: 8 (n=11), 18 (n=17), 40 (n=300), 62 (n=1453). Exact
verification then costs 0.35 s (n=11, pure Python). Tools: `msolve`, Singular,
Macaulay2, PARI/GP `algdep`, SymPy `nsimplify`, `HomotopyContinuation.jl`
`certify`, `alphaCertified`.

**For us:** symbolic refinement does not lower `L`, but it gives us exact
witnesses that certify cleanly at any precision, and it protects against the
"touching is undecidable" trap. Low priority.

---

## 5. Bibliography (compact)

```
Friedman 2009        EJC DS7                     squares-in-squares survey, lower bounds
Gensane & Ryckelynck 2005  Discrete Comput Geom  inflation / stochastic billiard
Wales & Doye 1997    J Phys Chem A               basin hopping
Dueck & Scheuer 1990 J Comput Phys              threshold accepting
Boettcher & Percus 2001 PRL                     extremal optimization
Hansen 2006          CMA-ES tutorial
Storn & Price 1997   J Global Optim              differential evolution
Shaw 1998            LNS
Ropke & Pisinger 2006 LNS for VRPTW
Lubachevsky & Stillinger 1990  J Stat Phys      billiards / compression
Stoyan, Romanova, Pankratov     phi-functions for packing
Murata et al. 1996   IEEE TCAD                   sequence pair
Otten 1982           slicing floorplans
Hong et al. 2000     IEEE TCAD                   corner block list
Montanher et al. 2018 J Global Optim             interval B&B, squares in circle
Markót & Csendes 2005 SIAM J Optim               circle packing proofs
Berthold et al. 2026 arXiv:2601.05943           Farkas non-overlap, global solvers
Berthold et al. 2026 arXiv:2605.04850           S-lemma containment
Novikov et al. 2025  AlphaEvolve
Sun & Yang 2023      NeurIPS                     DIFUSCO
Lehuger & Henon-Just 2026 arXiv:2606.10611      Polygons Transformer
Fang et al. 2023     Mathematics 11(2):327        hybrid RL packing
Huang et al. 2022    arXiv:2211.09382, RA-L      hierarchical RL packing
Abdou et al. 2025    J Intell Manuf              DRL rotational nesting
```
