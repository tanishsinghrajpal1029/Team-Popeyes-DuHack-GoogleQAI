# Architecture and performance

How `create_dicke_state` builds its circuits, how we measure them, and where they stand.
Every number below comes from a CSV in `experiments/results/` and can be regenerated with
the commands in README.md. All performance numbers are **simulated**; nothing here has run on
hardware.

## 1. What we optimize

The judges score each (N, n) case from one 10k-shot run per basis, and sum the cases:

    score = 1 / (z + x)

- **z** = χ² / (K · f), where K = C(N, n) is the number of valid bitstrings, f is the
  fraction of shots with exactly n ones, and χ² is Pearson's statistic for the K valid
  counts against uniform.
- **x** = Σ_k (measured − ideal)² / ideal² over the average weight-k X-basis correlators,
  k = 1 … M. Where the ideal value is 0, the plain squared error is used.

### Metrics we report, and which one is the objective

| metric | what it is | role |
|---|---|---|
| **E[score]** | expected single-draw score: ∫ pdf(χ²) / (χ²/(K f) + x) dχ², with χ² ~ noncentral χ²(K−1, λ) | **the objective**: the judges' total is a sum of draws, so its expectation is Σ E[score] |
| median score | the typical single draw | risk view: what one run usually gives |
| % of ceiling | median ÷ median of a *perfect* state, which is limited only by shot noise | comparable across cases with very different K |
| f | right-weight fraction (leakage) | diagnostic |
| χ²/(K−1) | 1 means shot noise only; the excess is population imbalance, λ = 10000 · f · mean(δ²) | diagnostic |
| x | X term at infinite shots (coherence and phases) | diagnostic |

E[score] is computed semi-analytically (`experiments/scoring.expected_score`; it matches
10⁶-draw Monte Carlo at every K we checked). We do not use resampled means. For K ≤ 3 the
score is extremely heavy-tailed: E[1/χ²₁] diverges, and only the x floor keeps the
expectation finite. A resampled mean there is dominated by a handful of lucky draws.

**What follows from this:**
- E[score] ≈ K·f/(K−3) for a perfect state with K > 3, but it is far larger for K ≤ 3: about
  60–180 for W₂.
- **For K ≤ 3**, the X-term floor x and the imbalance λ dominate. Fewer CZs matter less
  there than the right qubits and coherence.
- **For large n**, x grows as large as z (high-weight correlators decay), so CZ count is the
  lever.

**Assumptions:**
- max_correlator_size M = N, the strictest choice. The CSVs also record M = 4.
- The calibrations are the notebook's fake-data generators with seeds 0, 1, 2.

## 2. Architecture (`main.py`)

```
create_dicke_state(N, n, calibration)
 └─ build(..., w_tree, synth, scs, tiny, dd, bryan)   defaults = the submission
     ├─ k = min(n, N−n); k = 2 with N ≥ 6, or k ≥ 3 with N ≥ 9 → Bryan's solution (vendored;
     │     its own compile + DD, our DD is not added); on any exception fall through
     └─ _build_core: if n > N/2, prepare D_{N−n} and flip every qubit at the end
         ├─ k = 0  → product state (no gates, or X on all), best-readout qubits
         ├─ k = 1, N = 2 → W pair chosen by simulating every placement (predicted E[score])
         ├─ k = 1  → W broadcast tree
         ├─ (N, k) in _SYNTH with raw CZ ≤ SCS CZ → whole-state exact circuit, placed on chip
         ├─ k ≤ 5  → Bärtschi–Eidenbenz from synthesized SCS blocks on a path
         ├─ else   → Bärtschi–Eidenbenz, routed (RouteCQC) on a low-error path
         ├─ compile to cirq.CZ + PhasedXZGate; every moment is all-CZ or all-1q
         └─ dynamical decoupling: XY4 in idle windows with ≥ 4 free 1q slots
```

On the hardware track, k = 2 at N = 6–8 goes to Bryan before `_SYNTH` is consulted, and every
other k ≥ 2 case up to N = 8 has a `_SYNTH` entry, so SCS and routed B-E run there only as
fallbacks. They are the main path for N > 20; the bonus entry has its own candidate set
(section 5).

The Dicke state is permutation-symmetric, so we return wherever the logical qubits end up.
Routing permutations never need undoing.

### Constructions

**Portfolio with Bryan's solution.** Bryan's branch (`origin/bryan`) is an independent
solution. It generates candidates (W trees, Bärtschi–Eidenbenz on paths and ladders, and a
two-arm "split" B-E that runs two half-size B-E circuits in parallel), places them, and
keeps the one with the best predicted log-fidelity. Under our harness (`vs_bryan.csv`) it
beats ours for k = 2 with N ≥ 6 (1–7%, about 3 fewer CZ) and for k ≥ 3 with N ≥ 9 (21–39%;
D¹⁰₅ needs 85 CZ vs our 185). Ours win the W states, k = 2 up to N = 5, and k ≥ 3 up to
N = 8 (by up to +28%). `main.py` vendors his file verbatim (four renames to avoid name
clashes) and dispatches by that measured rule. The bonus entry picks the shallowest of several
candidates, his among them (section 5). His derivations of every building block (known-target rotations,
the 2-CZ transfer bound, the 5-CZ G₃, the merged 4-CZ top block) are in
`docs/bryan_constructions.md`.

- **W tree (k = 1).**
  - Amplitude starts on a root. Each round, every qubit holding amplitude hands its
    subtree's share to one fresh neighbour.
  - The first split is 1 CZ, because its input is a product state. Every later split is a
    2-CZ Givens rotation. That totals 2N − 3 CZ, the same as a chain, but depth is ~log N
    rounds instead of N.
  - The circuit is excitation-preserving, so every branch carries one excitation the whole
    time. T1 exposure is therefore balanced across branches; we tested this.
- **Whole-state synthesis (N ≤ 8, k ≥ 2;** `experiments/synth.py`**).**
  - Dicke states are real, and {Ry, CZ} is universal for real states. So the ansatz is
    layers of Ry on every qubit plus brickwork CZs on a fixed shape (a path or a 2×m ladder).
  - Exact-gradient L-BFGS drives infidelity below 1e-13, then greedy pruning deletes CZs
    while the circuit stays exact.
  - Results: D⁴₂ 5 CZ, D⁶₃ 15, D⁷₃ 25, D⁸₄ 27 (29 in the table; 2 cancel when compiled) (routed B-E needs 21, 80, 98, 162).
  - Synthesis takes minutes per case, so the angles are cached in `main.py`. They depend only
    on the shape. Placement (8 grid symmetries × translations, calibrated qubits and couplers
    only) is decided at build time.
- **Robust angle picks (**`experiments/vqe.py`**).**
  - One CZ structure has many exact angle sets, and they react differently to coherent
    Ry over-rotation, which our noise model lacks.
  - We deploy the set with the least worst-case regret over an over-rotation range. It is
    chosen on calibration seed 0 and checked on seeds 1–2.
  - Effect: sensitivity to 1% over-rotation drops 1.1–5.5× (D⁵₂: λ 50 → 9). The cost under our
    standard noise model is between −0.9 and +2.6 points of ceiling share.
  - Status: deployed for every synthesized case (N ≤ 8) except D⁸₃, whose 31-CZ structure
    (exported Oct 25) has no alternative exact angle sets yet: random restarts rarely reach
    exactness on it (0 of 4 in a timing test). Its sensitivity is close to the old 43-CZ pick:
    λ 107 vs 86 for uniform 1% over-rotation, 86 vs 112 for per-qubit 1%
    (`vqe_coherent_8_3_cz31.csv` vs `vqe_coherent_8_3.csv`).
- **SCS blocks (k ≤ 5;** `experiments/scs_synth.py`**).**
  - B-E's Split & Cyclic Shift block SCS_{m,k} acts on k+1 consecutive line qubits, and its
    input is always a sorted string. That makes each block a small isometry, which we
    synthesize exactly on a path, so the whole B-E circuit needs no routing.
  - Cost per inner block: 2, 4, 12, 17, 29 CZ for k = 1–5. D^N_2 costs about 4N CZ.
  - The CZ pattern is pruned once per k; the angles are re-fit for every m.
- **Routed B-E (fallback).** The paper's gates, with the doubly-controlled Ry hand-decomposed
  into 4 CNOT + 4 Ry (cirq's generic decomposition needs 10 CZ). Placed on a beam-searched
  path and routed with `cirq.RouteCQC`. Raw and CZ-precompiled variants are both tried, and
  the one with fewer CZ is kept.

### Layout objective (all constructions)

Every calibrated rate enters as **reported value + 1σ** (its stated uncertainty). This
hedges against qubits that only look good by chance. It is a no-op on the notebook's
uniform-uncertainty calibrations (0 of 75 placements changed).

- CZ error × uses, plus single-qubit error × gate count.
- Readout error weighted by how often each qubit reads 1.
- Idle time × (1/T1 + 1/Tφ).
- log(1 + λ_ro/(K−1)), where λ_ro is the χ² excess predicted from the spread of readout
  asymmetry (e1 − e0) across the chosen qubits. Pre-compensating amplitudes would break
  exactness, so qubit choice is the only remedy.

### Dynamical decoupling

- Echoing every idle window cost up to ~1 point of ceiling share in simulation: sub-0.3 µs
  idles have nothing to refocus, so only the pulse error shows.
- On long circuits the sign depends on the true T2*. With T2* = 20 µs, D⁷₂ went 62% → 52%;
  with T2* = 5 µs, D⁷₃ went 18% → 33%.
- Decision (Richard): echo only windows with ≥ 4 free single-qubit slots. Short circuits are
  essentially untouched.

### Guarantees (tests/)

- Exact noiseless state, including a check with the notebook's own (fixed) `test_circuit`.
- Only `cirq.CZ` on calibrated pairs, `PhasedXZGate` on calibrated qubits, and
  `ResetChannel`. This is stricter than `Willow105.validate_circuit`.
- ≤ 1,750 moments and under 60 s build time. Builds typically take 0.1–3 s; the worst in
  the hardening sweep is 8.2 s.
- Coverage: every N ≤ 10 and every n, on 3 calibrations plus a 10%-dropout calibration.

## 3. How we evaluate

1. Seeded fake calibrations from the notebook's generators.
2. Our noise model (`experiments/noise.py`):
   - calibrated depolarizing error after each CZ and each non-virtual PhasedXZ;
   - T1 = 68 µs (Willow spec sheet) and Tφ = 100 µs (assumed), per moment duration
     (25 ns for 1q layers and 42 ns for CZ layers, assumed);
   - asymmetric readout flips.
3. Exact density-matrix simulation in Cirq of the Z basis, and of the X basis via an
   H-equivalent layer with its own noise.
4. 20,000 simulated 10k-shot runs through the judges' scorer, plus the semi-analytic E[score].

**Not modeled:** coherent errors (studied separately in `vqe.py`), ZZ crosstalk, leakage,
per-qubit T1 spread, and drift. Real hardware will score lower than these numbers.

## 4. Current performance

Idle noise, averaged over 3 calibrations, M = N. "Base" is this morning's routed B-E,
with a chain for W states.

| case | K | construction | CZ (base → now) | f | χ²/(K−1) | x | E[score] (base → now) | median | % of ceiling (base → now) |
|---|---|---|---|---|---|---|---|---|---|
| W2 | 2 | tree, pair chosen by predicted E[score] | 2 → 1 | 0.990 | 1.041 | 0.0005 | 54.26 → 65.55 | 4.157 | 96% → 96% |
| W3 | 3 | tree | 4 → 3 | 0.977 | 1.019 | 0.0013 | 7.45 → 7.97 | 2.052 | 92% → 97% |
| W4 | 4 | tree | 6 → 5 | 0.966 | 1.013 | 0.0019 | 3.37 → 3.41 | 1.609 | 95% → 96% |
| D4,2 | 6 | whole-state synth | 21 → 5 | 0.968 | 1.019 | 0.0058 | 1.55 → 1.85 | 1.300 | 84% → 95% |
| W5 | 5 | tree | 8 → 7 | 0.956 | 1.021 | 0.0024 | 2.26 → 2.27 | 1.383 | 92% → 92% |
| D5,2 | 10 | whole-state synth | 35 → 10 | 0.951 | 1.049 | 0.0119 | 1.04 → 1.27 | 1.069 | 75% → 89% |
| W6 | 6 | tree | 10 → 9 | 0.945 | 1.016 | 0.0030 | 1.81 → 1.84 | 1.277 | 92% → 93% |
| D6,2 | 15 | whole-state synth | 45 → 18 | 0.914 | 1.054 | 0.0233 | 0.84 → 1.05 | 0.953 | 69% → 85% |
| D6,3 | 20 | whole-state synth | 80 → 15 | 0.923 | 1.014 | 0.0347 | 0.54 → 1.03 | 0.959 | 48% → 88% |
| W7 | 7 | tree | 12 → 11 | 0.932 | 1.016 | 0.0036 | 1.58 → 1.59 | 1.199 | 91% → 92% |
| D7,2 | 21 | whole-state synth | 63 → 20 | 0.910 | 1.054 | 0.0273 | 0.68 → 0.98 | 0.912 | 60% → 84% |
| D7,3 | 35 | whole-state synth | 98 → 25 | 0.893 | 1.013 | 0.0855 | 0.45 → 0.89 | 0.859 | 42% → 82% |
| W8 | 8 | tree | 14 → 13 | 0.924 | 1.017 | 0.0035 | 1.40 → 1.44 | 1.141 | 88% → 90% |
| D8,2 | 28 | whole-state synth | 72 → 26 | 0.878 | 1.068 | 0.0381 | 0.58 → 0.89 | 0.847 | 53% → 80% |
| D8,3 | 56 | whole-state synth | 123 → 31 | 0.865 | 1.026 | 0.1115 | 0.39 → 0.81 | 0.792 | 38% → 77% |
| D8,4 | 70 | whole-state synth | 162 → 27 | 0.868 | 1.024 | 0.1209 | 0.24 → 0.80 | 0.785 | 24% → 77% |
| W9 | 9 | tree | 16 → 15 | 0.914 | 1.015 | 0.0040 | 1.33 → 1.34 | 1.097 | 89% → 89% |
| D9,2 | 36 | SCS blocks | 89 → 30 | 0.822 | 1.022 | 0.0564 | 0.45 → 0.83 | 0.805 | 42% → 77% |
| D9,3 | 84 | SCS blocks | 147 → 71 | 0.706 | 1.161 | 0.3397 | 0.27 → 0.52 | 0.513 | 26% → 50% |
| D9,4 | 126 | SCS blocks | 211 → 109 | 0.595 | 1.060 | 1.0528 | 0.19 → 0.36 | 0.356 | 19% → 35% |
| W10 | 10 | tree | 18 → 17 | 0.902 | 1.020 | 0.0042 | 1.24 → 1.26 | 1.055 | 87% → 88% |
| D10,2 | 45 | SCS blocks | 97 → 34 | 0.798 | 1.022 | 0.0653 | 0.38 → 0.79 | 0.769 | 36% → 74% |
| D10,3 | 120 | SCS blocks | 171 → 81 | 0.667 | 1.167 | 0.3987 | 0.23 → 0.47 | 0.471 | 22% → 47% |
| D10,4 | 210 | SCS blocks | 238 → 126 | 0.544 | 1.052 | 1.2592 | 0.17 → 0.31 | 0.314 | 17% → 31% |
| D10,5 | 252 | SCS blocks (k=5) | 324 → 185 | 0.487 | 1.086 | 1.9975 | 0.12 → 0.24 | 0.237 | 12% → 24% |

The D⁷/D⁸ rows predate their robust angle picks. Those picks change E[loss] by −2.1 to +0.6%
with no over-rotation, on held-out calibrations, so the rows were not re-simulated.

**Portfolio (current `main.py`, `portfolio.csv`).** Σ E[score] **102.0** and Σ median **27.9**,
vs 101.0 / 26.9 for our constructions alone and 100.5 / 27.2 for Bryan's alone. The per-case
rows above are our constructions; for k = 2 with N ≥ 6 and k ≥ 3 with N ≥ 9 the deployed numbers
are Bryan's (`vs_bryan.csv`). Bonus: 43.39 (`bonus_variants.csv`; section 5), up from 37.15
when the entry only chose between ours (23.8) and Bryan's (35.3).

**Against the preliminary combined submission** (`prelim_vs_latest.csv`; teammates'
`tanish_submission.py` at commit `1995558`, which uses an older copy of ours for 4 ≤ N ≤ 8 and
Aanha's builder for N ≥ 9; same harness, 3 calibrations):

| | preliminary | current |
|---|---|---|
| Σ median (25 cases, N ≤ 10) | 27.53 | **27.99** |
| Σ E[score] | 90.77 | **102.88** |
| bonus `score_solution` | 33.67 | **43.39** |

The current submission wins k = 2 at N ≥ 6 (up to +16% at D¹⁰₂, from 31 vs 49 CZ), k = 3–4 at
N = 9–10 (+4–5%) and N ≤ 3. The preliminary one wins D¹⁰₅ (−5%), D⁹₄ (−4%) and W₉/W₁₀
(−0.5/−1.5%); see section 6 for why.

**Totals over these 25 cases (our constructions only):**

| | baseline | now | perfect state |
|---|---|---|---|
| Σ E[score] (objective) | 82.8 | **99.7** | 218.7 |
| … of which K ≥ 4 | 21.1 | **26.2** | 33.2 |
| … of which K ≤ 3 (W₂, W₃) | 61.7 | **73.5** | 185.5 |
| Σ median score | 22.2 | **26.9** | 33.1 |

## 5. Robustness, bonus, and known gaps

**Hardening** (`experiments/harden.py`, 969 builds, plus `tests/test_robustness.py`):
- Every N ≤ 12 with every n, on 6 calibrations (1–15% qubit dropout): all exact and valid.
- N = 13–30 (rules, build time, moment budget) and W states up to N = 105: all valid.
- No crashes anywhere.
- `NotImplementedError` only where the chip has no connected layout of size N (N ≥ 100 with
  dropouts).
- After the portfolio merge, nothing exceeds 1,750 moments (max 623 for N ≤ 30) and the worst
  build takes 8.2 s.
- Calibration dicts with reversed or duplicated pair keys, list or numpy values, missing
  readout entries, or couplers to uncalibrated qubits all work.

**Bonus entry** (`create_dicke_state_adaptive`, `bonus_candidates`; `experiments/bonus_depth.py`,
`results/bonus_variants.csv`). The organizers' first feedback (Sat night) put our bonus at 37.15
against a leader at 41.10. All 56 cases were present and exact, so the gap was depth. The entry
now builds every exact candidate below, checks nothing else, and returns the shallowest after
`_pack` (mixed 1q/CZ moments). `score_solution` over 2 ≤ N ≤ 15: **43.39** (was 37.15; the leader's first-round number was 41.10).

- *Unitary on purpose.* `cirq_google.Willow105.validate_circuit` rejects classically
  controlled operations (`X.with_classical_controls(m)`: "gate which is not supported");
  plain measurement and reset pass. A feedforward circuit would fail the validator the
  notebook tells judges to use, and for N ≤ 15 every measurement-based scheme we costed is
  deeper anyway (section 6 of `docs/bryan_constructions.md`).
- *Candidates.* (1) ours via `build(..., dd=False, bryan=False)` for k ≤ 1 or N ≤ 8; (2) Bryan's
  bonus solution; (3) for k = 1, Aanha's broadcast W tree (ported from `origin/Aanha` `w_tree.py`
  with uniform costs, every root): its staggered rounds pack 1–2 moments shallower than ours
  (W₉ 12 vs 14, W₁₀–₁₂ 13 vs 14, W₁₃–₁₅ 14 vs 15, W₅ 9, W₆/W₇ 10); (4) for k ≥ 2, two-arm split
  variants: thermometer on either arm, bottom-up or top-down thermometer order, and the last
  m0 ∈ {3, …, 7} positions of each arm's B-E unitary replaced by a synthesized block
  (`_TAIL`, `experiments/tail_synth.py`); m0 equal to the arm length makes the whole arm one
  block, which is what wins for k = 2 (U₇ r≤2, 27 moments, vs a 37-moment sweep arm) and
  k ≤ 4 at N ≤ 12 (U₆ r≤4, 43 vs 97); (5) whole-state synthesized circuits ranked by layer
  count rather than CZ count (`_BONUS_SYNTH`, `synth.py --bonus-export`): D⁵₂ 13 (was 15),
  D⁶₃ 15 (17), D⁷₃ 21 (23), D⁸₄ 24 (25), D⁹₂ 25 (28), D⁹₄ 59 (70).
- *Where the depth was.* A split circuit is 2·(CZ depth)+1 moments with the CZ depth set by
  the B-E sweep: D¹⁵₇ was 160 moments, 147 of them the 8-qubit arm alone; each arm position
  costs ~18 moments at k ≥ 5 (10 at k = 3, 5 at k = 2), and the last four positions are a
  strictly serial 41-moment tail (35 at k = 3). The tail is the small Dicke unitary
  U_{m0}: |0^{m0−r}1^r⟩ → |D^{m0}_r⟩, r ≤ min(k, m0), an isometry we synthesize exactly with
  `scs_synth.BatchCircuit` (fewest layers first): U₃ r≤3 17 moments (sweep 19), U₄ r≤3 23 (35),
  U₄ r≤4 29 (41), U₅ r≤3 31 (49), U₅ r≤4 43 (69), U₆ r≤2 27, U₆ r≤3 39 (63), U₆ r≤4 43 (81),
  U₄ r≤2 19, U₅ r≤2 23, U₆ r≤2 27, U₇ r≤2 27, U₇ r≤3 43 (77), U₇ r≤4 59 (one seed of 20, 69 min), U₈ r≤2 31; U₅ r≤5, U₆ r≤5 and U₈ r≤3 were not found exact within 25–60 min on 10–13 seeds each (best 1e-3 to 1e-2); a 2×4 ladder shape
  for the 6–8 qubit blocks was worse than the path (U₆ r≤4 not exact at 30 layers vs 21). The layer count found varies
  2–6 between seeds (U₄ r≤4: 14–21 layers over 12 seeds), so every block got 8–13 seeds.
  Effect: D⁹₃ 54 → 36, D¹¹₄ 88 → 51, D¹²₄ 88 → 51, D¹³₂/D¹⁴₂ 40 → 31, D¹⁵₂ 46 → 36,
  D¹³₃/D¹⁴₃ 82 → 48/49, D¹⁴₄ 106 → 78, D¹⁵₃ 96 → 62, D¹⁵₄ 124 → 92, D¹⁵₇ 160 → 144. The thermometer order never changed a winner (bottom-up always won); putting
  the thermometer on the non-critical arm did, by 2–6 moments.
- k = 2: a *partial* tail does not beat the 4-CZ top blocks (U₃ r≤2 15 vs 13, U₄ r≤2 19 vs 19),
  but a whole-arm block does (U₆/U₇ r≤2 at 27 moments), so D¹¹₂–D¹⁵₂ dropped 3–10 moments;
  D¹⁰₂ (arms of 5) stays at Bryan's 28.
- Every candidate is checked for exactness (1e-9), routing and the strict gate whitelist in
  `bonus_depth.py` and `tests/test_validity.py` (all 56 cases, plus a no-regression check
  against `notebook_bonus.csv`). Slowest build: about 3 s.

**Known gaps, by expected value:**
1. **W₂/W₃ X-term floor.** These cases still hold most of the remaining expected gap: 73.5 of
   a possible 185.5. W₂ now picks its qubit pair by predicted E[score]: +9% expected, −5% median,
   a deliberate trade toward the objective. For W₃ the same selector gained nothing, so it is
   off there.
2. **D⁸₃ now 31 CZ** (was 43; `synth/D8_3_path_s1.json` from the overnight run, exported with
   `synth.py --export --nmax 8`). On 6 calibrations (`d83_export.csv`, 20k repeats) the median
   rises 0.745 → 0.792 on seeds 0–2 and 0.727 → 0.776 on held-out seeds 3–5 (+6.3/+6.7%), with
   f 0.83 → 0.86 and x 0.15 → 0.11. It still lacks a robust angle pick (see above). The k=2 all-ones SCS block costs 7 CZ;
   another model reports a 4-CZ equivalent, which would save 3 CZ in every k ≥ 3 SCS circuit.
3. **Hardware gap.** Coherent errors, crosstalk, and the true T2* are unknown. For small K,
   1% over-rotation alone would cost about half the score. The hedges are the robust angle
   picks (all synthesized cases) and long-idle DD. Placement ranks every rate at
   value + 1σ, so qubits that only look good by chance are not preferred.

## 6. Big N: diagnosis and what we tried

**Diagnosis** (Bryan's circuits, deployed for k ≥ 3 at N ≥ 9; `portfolio.csv`):

| case | CZ | duration | f | x (M=N) | median (M=N) | median (M=4) | correlator ratio, weight 2 … max |
|---|---|---|---|---|---|---|---|
| D9,3 | 50 | 1.83 µs | 0.777 | 0.225 | 0.651 | 0.702 | 0.81 … 0.67 |
| D9,4 | 63 | 2.37 µs | 0.734 | 0.523 | 0.504 | 0.613 | 0.76 … 0.56 |
| D10,3 | 59 | 1.83 µs | 0.748 | 0.252 | 0.619 | 0.671 | 0.80 … 0.65 |
| D10,4 | 77 | 2.37 µs | 0.693 | 0.618 | 0.481 | 0.602 | 0.75 … 0.52 |
| D10,5 | 85 | 2.64 µs | 0.668 | 1.020 | 0.391 | 0.564 | 0.72 … 0.45 |

- One quantity, overall fidelity (~0.6–0.75), drives both terms: z ≈ 1/f, and x ≈ Σ over even
  weights of (1 − ratio)², which grows with M. Imbalance is fine (χ²/(K−1) ≈ 1.02–1.09).
- Gates and idling matter about equally. D10,5: gate noise only → median 0.53, idle only →
  0.59, both → 0.375. So both CZ count and duration have to drop.

**N = 11–15** (`bign/traj_*.csv`, quantum trajectories, 300 per case, calibration seed 0,
debiased; `experiments/trajectory.py`). Median score, M = N:

| k | N=11 | N=12 | N=13 | N=14 | N=15 |
|---|---|---|---|---|---|
| 1 | 1.048 | 0.999 | 0.951 | 0.941 | 0.958 |
| 2 | 0.824 | 0.778 | 0.748 | 0.708 | 0.663 |
| 3 | 0.510 | 0.530 | 0.497 | 0.447 | 0.445 |
| ⌊N/2⌋ | 0.321 | 0.255 | 0.237 | 0.168 | 0.154 |

W states stay at 82–89% of ceiling; half-filling falls to ~15%. An exact density matrix for
D12,6 (`bign/probe.csv`, 59 min) gives 0.260, matching the trajectory estimate (0.255).

**Teammates' builder (Aanha's, inside `tanish_submission.py`).** All exact and valid; hardening
961/969 (the same 8 impossible requests); worst build 30 s. Out of sample (seeds 3–5,
`vs_tanish_seeds345.csv`) it wins only where it won on the bench seeds: W states (+1–3%) and
n = ⌊N/2⌋ (+2–3%); it loses k = 2 (−15%) and k = 3–4 (−3 to −8%). At N = 11–15 and n = ⌊N/2⌋ it is
ahead by 3–13% except at N = 12 (tie). Two reasons:
- its W trees stay at 14 moments for every N (ours grow to 19–23);
- it packs single-qubit gates into CZ moments, which our noise model treats as free. Stratifying
  its D10,5 circuit (seed 0) costs 0.410 → 0.394 (`bign/mixed_probe.csv`, `experiments/mixed_moments.py`),
  so about half its edge there is this timing assumption. **Open question:** do mixed CZ + 1q
  moments behave on Willow as the model assumes? `_to_cz_phxz` deliberately stratifies.

**What did not work:**
- **Whole-state re-synthesis at N = 9–10** (`synth.py`, overnight, 24 workers, stopped): best
  exact circuits D9,2 33 CZ, D10,2 35, D9,4 98, D9,3 136, vs 27, 31, 63, 50 deployed.
  Ladder shapes became exact only for D10,2 (50 CZ). The random-start brickwork approach stops scaling at N ≈ 9.
- **Warm-start compression of Bryan's circuits** (`experiments/compress.py`): an exact U3
  re-parametrization of his D9,3, then greedy single-CZ deletion. Exactness was recovered 0 of
  8 times (best residual 1.6e-4); the split construction is locally tight.
- **ALAP scheduling** (keep qubits in |0⟩ longer): −1.4 to +0.1% E[score] under idle noise; removed.

**Bonus depth: what did not work** (Oct 25, before the levers in section 5):
- Givens-only brickwork (excitation-preserving, 2 CZ per gate) on 2D patches: D⁸₄ still at
  2e-3 infidelity after 13 rounds; D⁶₃ needs 7 rounds (16 gates) where Ry+CZ needs 7 layers.
- Ry+CZ brickwork on 2D patches instead of paths: more layers, not fewer (D⁶₃ 9–10 vs 7;
  D⁷₃ and D⁸₄ not exact by L = 13–14). Complex single-qubit layers (full PhasedXZ, 3× the
  parameters): the same minimal layer count (D⁶₃ 7). The minimal layer count is a property of
  the brickwork, not of the parameter budget, and it grows faster than N: from-scratch
  synthesis stops paying at N ≈ 8 regardless of ansatz.
- Divide-and-conquer beyond two arms: a weight split on a *superposed* unary register costs
  O(k) controlled rotations per thermometer bit, i.e. about as much as the B-E sweep it would
  replace; at N ≤ 15 the constants eat the asymptotic gain.
- Top-down thermometer (arm's own first sweep can start after two bits): exact, never shallower
  than bottom-up with the thermometer on the non-critical arm.
- **Re-synthesis of N ≤ 8 with 8 more seeds per shape** (`results/synth/smallN_seeds.log`) found
  fewer CZs for D⁵₂ (9 vs 10), D⁶₃ (14 vs 15) and D⁷₃ (22 vs 25), but on three calibrations
  (`synth_reseed.csv` vs `portfolio.csv`) none measures better: D⁶₃ is worse (median 0.91–0.95
  vs 0.96, λ 0.8–1.0 vs 0.25), D⁵₂ and D⁷₃ are ties within sampling noise while losing their
  robust angle picks. The hardware table was left as deployed; the same runs did shave bonus
  moments (D⁵₂ 15 → 13, D⁷₃ 23 → 21, via `_BONUS_SYNTH`).

## 7. Maintenance notes

- **Exports rewrite `main.py`** between `BEGIN`/`END` markers: `synth.py --export` → `_SYNTH`
  (applies `results/vqe/picks.json` only when structure and solution match);
  `scs_synth.py --export` → `_SCS`; `vqe.py apply` → `picks.json` (follow with `synth.py --export`);
  `tail_synth.py --export` → `_TAIL`; `synth.py --bonus-export` → `_BONUS_SYNTH` (bonus only).
  After any change to `main.py`, run `experiments/make_notebook.py --run`.
- **Synthesis result files are named by (case, shape, seed)**: a rerun with an already-used seed
  overwrites the earlier JSON (this happened to `D8_3_path_s1/s2` on Oct 25; the originals were
  restored from git and the reruns kept as `*_rerun.json`). Use fresh seeds for new runs.
- **Bonus section of `main.py`** (after `# ---- bonus track`) is the only place that may change
  the bonus entry; the hardware path is frozen and `tests/test_golden_small_n.py` guards it.
  Aanha's W tree is ported there with attribution (uniform costs, every root).
- **Bryan's code is vendored verbatim** with renames `_X → _BRYAN_X`, `_Y → _BRYAN_Y`,
  `create_dicke_state(_adaptive) → _bryan_create_dicke_state(_adaptive)`. Our `_X`/`_Y` are DD
  pulses.
- `evaluate.py --opt key=value` / `trajectory.py --opt` pass options to `main.build`:
  `bryan=false`, `synth=false`, `scs=false`, `tiny=false`, `w_tree=false`, `dd=false`.
- `tests/test_golden_small_n.py` freezes every N ≤ 8 circuit. Regenerate
  (`--regen`) only for a case that measured strictly better, with before/after numbers in the
  commit message.
- Parallel jobs: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (else ~15× slower), total processes
  ≤ vCPUs, and a separate `--out` CSV per concurrent run.
- **Assumptions behind big-N conclusions:** Tφ = 100 µs and 25/42 ns layer durations are
  assumed (T1 = 68 µs is the spec sheet). The judges' `max_correlator_size` and (N, n) set are
  unknown; we report M = N, and M = 4 is in every CSV.
