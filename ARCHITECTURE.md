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
 └─ build(..., w_tree, synth, scs, dd)                 defaults = the submission
     ├─ k = min(n, N−n); if n > N/2, prepare D_{N−n} and flip every qubit at the end
     ├─ k = 0  → product state (no gates, or X on all), best-readout qubits
     ├─ k = 1  → W broadcast tree
     ├─ (N, k) synthesized and cheaper than SCS → whole-state exact circuit, placed on chip
     ├─ N = 2  → W pair chosen by simulating every placement (predicted E[score])
     ├─ k ≤ 5  → Bärtschi–Eidenbenz from synthesized SCS blocks on a path
     ├─ else   → Bärtschi–Eidenbenz, routed (RouteCQC) on a low-error path
     ├─ compile to cirq.CZ + PhasedXZGate; every moment is all-CZ or all-1q
     └─ dynamical decoupling: XY4 in idle windows with ≥ 4 free 1q slots
```

The Dicke state is permutation-symmetric, so we return wherever the logical qubits end up.
Routing permutations never need undoing.

### Constructions

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
  - Results: D⁴₂ 5 CZ, D⁶₃ 15, D⁷₃ 25, D⁸₄ 27 (routed B-E needs 21, 80, 98, 162).
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
  - Status: deployed for every synthesized case (N ≤ 8).
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
- ≤ 1,750 moments and under 60 s build time. Builds currently take 0.1–3 s.
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
| D8,3 | 56 | whole-state synth | 123 → 43 | 0.851 | 1.054 | 0.1488 | 0.39 → 0.75 | 0.740 | 38% → 72% |
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

**Totals over these 25 cases:**

| | baseline | now | perfect state |
|---|---|---|---|
| Σ E[score] (objective) | 82.8 | **99.7** | 218.7 |
| … of which K ≥ 4 | 21.1 | **26.2** | 33.2 |
| … of which K ≤ 3 (W₂, W₃) | 61.7 | **73.5** | 185.5 |
| Σ median score | 22.2 | **26.9** | 33.1 |

## 5. Robustness, bonus, and known gaps

**Hardening** (`experiments/harden.py`, 969 builds, plus `tests/test_robustness.py`):
- Every N ≤ 12 with every n, on 6 calibrations (1–15% qubit dropout): all exact and valid.
- No crashes anywhere. The worst build takes 17.6 s.
- `NotImplementedError` only where the chip has no connected layout of size N (N ≥ 100 with
  dropouts).
- Over 1,750 moments only for N ≥ 18 with k ≥ 4; those would score ≈ 0 anyway.
- Calibration dicts with reversed or duplicated pair keys, list or numpy values, missing
  readout entries, or couplers to uncalibrated qubits all work.

**Bonus entry.** `create_dicke_state_adaptive` reuses the constructions on a uniform
calibration, without DD, packed for depth. It is not adaptive yet. `score_solution` over
2 ≤ N ≤ 15 gives 23.8.

**Known gaps, by expected value:**
1. **W₂/W₃ X-term floor.** These cases still hold most of the remaining expected gap: 73.5 of
   a possible 185.5. W₂ now picks its qubit pair by predicted E[score]: +9% expected, −5% median,
   a deliberate trade toward the objective. For W₃ the same selector gained nothing, so it is
   off there.
2. **D⁸₃ synthesis is stuck at 43 CZ** (D⁸₄ needs 27). The k=2 all-ones SCS block costs 7 CZ;
   another model reports a 4-CZ equivalent, which would save 3 CZ in every k ≥ 3 SCS circuit.
3. **Hardware gap.** Coherent errors, crosstalk, and the true T2* are unknown. For small K,
   1% over-rotation alone would cost about half the score. The hedges are the robust angle
   picks (all synthesized cases) and long-idle DD. Placement ranks every rate at
   value + 1σ, so qubits that only look good by chance are not preferred.
