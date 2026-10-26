# 02 — Public notebook landscape (22 notebooks pulled and dissected)

Lower score is better. "LB" = the notebook's own claimed score (verified where
possible by re-scoring its exported file; some are only stated in prose).

## 0. Summary table

| notebook | LB | real algorithm | key idea worth stealing |
|---|---:|---|---|
| `arunsworkspace/...baseline-pipeline` | — | single‑square SA, penalty 25 | clean SAT validator; **writes degrees** (broken) |
| `zkskhurram/kaggle-squares-in-squares` | ~28.8 | grid + Göbel n=5 | provenance table only |
| `mirzamilanfarabi/squares-in-squares-v2` | — | numba penalty SA, rigid‑body moves | rigid subset moves + 42 600 s budget |
| `avikdas567/...sat-dynamics` | 25.44 | 150‑packing lookup + torch Adam | "centroid dilation theorem" for safe output |
| `yuki16/squares-baseline-merge-best-v2` | — | diagonal‑band closed form + SA | closed‑form 45° band counter |
| `avazaku/squarecraft-beyond-the-grid` | 22.7417 | construction catalogue + sparse LP | rectangular/asymmetric tilted cores |
| `kozykappa/subset-transfer-...` | 22.4753 | subset transfer + contact LP + compaction | uniform centre compaction λ formula |
| `skarin/santa-2025-...-what-transfers` | 22.4795 | Santa‑2025 playbook + closure ops | frame/deletion closure algebra |
| `seshurajup/1000-puzzles-ensemble` | 22.3747 | best‑per‑n merge | merge rule + `min_gap` screen |
| `kirilltym/...seam-lp` | 22.3856 | seam reopening + trust‑region LP | re‑open seams after composition |
| `cdeotte/...ensemble-22-376` | 22.3763 | fixed‑point bank of 545 puzzles | int16/int32 fixed‑point transport |
| `kirilltym/...edge-lp` | 22.3661 | boundary‑conditioned angle snapping | only snap angles in the outer shell |
| `cdeotte/...ftmc-milp-refill` | 22.3116 | FTMC gadget mutation + `fill_rect` MILP | decouple tilted core from upright fill |
| `alexchilton/certified-merge-...` | 22.5978 | per‑n argmin of public files | certification gate on decimal strings |
| `alexchilton/certified-witnesses+basin-hopping` | 22.3084 | Couzo certs + evand hop + fault DP | gadget DP, crop enumeration |
| `alexchilton/fault-line-dp+basin-hopping` | 22.3079 | same + 3‑pass DP loop + gadget quench | `fq` = moving separating lines |
| `malyshevdanil/neighbour-transfer-squeeze` | 22.3078 | n±1 transfer, border expansion, contact LP, CUDA L‑BFGS | full, shipped implementation of the whole stack |
| `lightningv08/cuda-packing-search` | 22.3078 | GPU basin hopping + fault DP + adaptive seams | support‑profile DP with dominated‑frontier pruning |
| `avazaku/squarecraft-iii-seam-surgery` | ~22.3078 | dual‑guided contact surgery | shadow prices rank which contacts to free |
| `lucifer19/think-rotate-verify` | — | artifact reproducer + verifier | 8‑stage verification harness |
| `kmmmvs/squares-dashboard` | 22.3078 | read‑only dashboard | provenance/attribution + hashes |
| `inversion/per-puzzle-leaderboard` | — | viewer | per‑puzzle LB schema (`n, L@50dp, team, date`) |

---

## 1. The methods, grouped

### (a) Construction / generative (no search)

* **`avazaku/beyond-the-grid`** — the only notebook that builds the score from
  scratch. Ladder: grid 28.94 → public catalogue 25.77 → outer rows 23.42 →
  Göbel + subdivision 23.13 → rectangular cores 22.84 → asymmetric cores 22.74 →
  sparse LP polish 22.7416.
* **`yuki16`** — closed‑form diagonal 45° band count
  `P(K,t) = #{(i,j) ∈ [0,K−1]² : i+j ≤ t}` with a piecewise formula, plus rows
  of 45° squares; scans `ΔL = 0.01`, `Δu = 0.02`.
* **`avikdas567`** — embeds 150 certified packings for n ≤ 324.

### (b) Continuous local optimisation

Universal energy for a fixed container side `L`:

```
E = Σ_{i<j} max(0, pen_ij)²  +  λ_wall Σ_i Σ_walls max(0, violation)²
pen_ij = R(θ_i−θ_j) − max_{4 axes} |proj_axis(c_i−c_j)|
R(δ)   = ½ (1 + |cos δ| + |sin δ|)
```

* `malyshevdanil` ships the best version: **smooth‑SAT continuation**
  `|z| → √(z²+τ²) − τ`, `cos → √(cos²+τ²)`, soft‑max over axes, τ = 1e‑5 → 1e‑8
  → exact; custom batched CUDA L‑BFGS (`M=8`, Armijo c₁=1e‑4, ≤24 halvings).
* `cdeotte` uses the same energy with a `2e‑10` contact clearance.
* `kirilltym` uses a PyTorch "soft pressure" variant (float32 propose, float64
  repair).
* `yuki16` / `mirzamilanfarabi` use it inside SA with integer overlap counts.

### (c) Exact repair / certification

Every serious notebook ends with the same primitive:

```
repair(p):  f = max over near pairs of (R(δ) + clearance) / max_axis|proj|
            if 0.9999 < f < 1.003: scale centres about bbox centre by f
```

then the official metric on the **written decimal strings**.

### (d) LP over the contact graph

Fix the angle of every square; non‑overlap between a pair becomes a *choice* of
separating axis. With the axis fixed, non‑overlap is linear in the centres:

```
(v·(c_i − c_j)) ≥ R(θ_i − θ_j)      for the chosen axis v
```

so "minimise the box side subject to these half‑planes" is an LP in
`2n + 1` variables. `scipy.optimize.linprog(method="highs")` solves it in ms.
This is the single most reused primitive in the whole competition:

* `malyshevdanil/contact_polish` — free centres **and** angles, trust region,
  one‑sided derivatives at right‑angle kinks, `t + 2e‑6·Σu` objective,
  28–300 rounds, step ×1.3 on success / ×0.4 on failure.
* `kozykappa` — same, 3 trust schedules, 3 passes.
* `kirilltym` — `reclose(trust, tie, rounds, seconds)` around sampled contact
  graphs; "seam reopening" = re‑solve after composition.
* `avazaku/II` — a **MILP** instead: binary `z_ij` picks the axis, big‑M from
  the trust radius; HiGHS; then soft‑pressure L‑BFGS; then free‑angle LP.
* `avazaku/III` — "seam surgery": solve a fixed‑angle LP, read **shadow prices**
  (dual values) to rank which contact constraints are most binding, then spend
  the MILP budget on those pairs.

### (e) Composition / transfer — where the big wins are

* **Fault‑line DP** (`alexchilton`, `cdeotte`, `lightningv08`):
  write `n = K² + D`, box `L = K + f`. Two integer lattices, one shifted by
  `(f, f)`, meet along a monotone staircase. A **gadget** `(w, h, Δm)` holds
  `w·h + Δm` squares in `(w+f)×(h+f)`. DP state `(cols, rows, surplus, tip
  type)`, value = minimax `f`. Improving one gadget lowers every `n` whose
  chain contains it.
* **Subset transfer** (`kozykappa`, `malyshevdanil`):
  `n+1 → n` by deleting the most "free" square; `n−1 → n` by inserting into the
  largest clearance hole; `m → n` by deleting `m−n` edge‑shallow squares.
* **Closure algebra** (`skarin`):
  `L(n) ≤ L(m)` for all `m > n` (any subset is valid), and the **frame**
  `n → n + j(2⌊L⌋ + j)` at side `L + j`.
* **Composition DP over n** (`malyshevdanil.choose`): forward expansion,
  backward running‑minimum — a 6‑layer monotone DP.

### (f) GPU / engineering

* `lightningv08`: CUDA basin hopping (spatial hash, per‑island L‑BFGS), CUDA
  fault composition over `31·31·63` gadgets, adaptive seams with support
  profiles, 261 s GPU for 52+10 replacements.
* `malyshevdanil`: CuPy `RawModule`, custom batched L‑BFGS, analytic gradient
  verified against finite differences, τ‑specialised JIT modules.
* `cdeotte`: fixed‑point int16/int32 bank, LZMA+base64 payload in one cell.

---

## 2. What every notebook has in common

1. A **certified merge** step at the end (per‑n argmin + official metric).
2. The same **penetration formula** `R(δ) = ½(1+|cos δ|+|sin δ|)`.
3. The same **repair** dilation before writing.
4. A **portfolio**: nobody searches all 1000 from scratch; they inherit the best
   public packings and improve a handful.

## 3. What nobody public does (→ our openings)

* No open **GPU annealer** equivalent to Schadt/Ellsworth. `lightningv08` is the
  closest but only 261 s of CUDA and a narrow gadget search.
* No systematic **exact MILP for the small‑n records** beyond `avazaku/II`.
* No **learning‑based** search at all (no RL, no diffusion, no learned move
  policy, no LLM code evolution) — AlphaEvolve was never applied to `s(n)`.
* No **systematic search over gadget space** with a proper DP objective; the
  public gadgets are hand‑picked crops.
* No **per‑puzzle leaderboard harvesting**: the inversion notebook is a viewer
  with no data; a script that scrapes daily per‑puzzle L values would give a
  free map of where competitors are weak (public LBs are per‑puzzle visible in
  `submission_details.csv` of each public notebook).
