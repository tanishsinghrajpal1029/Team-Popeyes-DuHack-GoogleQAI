# Handoff — Sat Oct 24, 2026, 6:05 PM EDT (moving to the A100 / 30-vCPU VM)

Deadline: **Sun Oct 25, 11:30 AM** (talk, repo, video). Branch to work on: **`richard`**
(pushed; `main` = the same code as of `69c73c1`, merged at Richard's request). Read
`ARCHITECTURE.md` for the design and the numbers, and `CLAUDE.md` for task and scoring.

## 0. Set up the VM (about 10 minutes)

```bash
git clone https://github.com/tanishsinghrajpal1029/Team-Popeyes-DuHack-GoogleQAI.git
cd Team-Popeyes-DuHack-GoogleQAI && git checkout richard
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow -r requirements.txt   # Willow105 needs the cirq 1.8 dev pin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # ALWAYS for parallel jobs (else ~15x slower)
.venv/bin/python -m pytest tests/ -q          # expect 256 passed in ~20 s
```

- **The GPU is not useful.** Every workload is small statevectors and latency-bound L-BFGS,
  so use the CPUs: `--workers 28` on 30 vCPUs.
- **Keep total processes ≤ number of vCPUs.**
- **Give each parallel `evaluate.py` run its own `--out` CSV.** `append_rows` rewrites the
  whole file, so concurrent writers clobber each other.

## 1. Where things stand

- **Submission:** `main.py` provides `create_dicke_state` and `create_dicke_state_adaptive`.
  It imports only cirq, cirq_google, numpy, networkx, scipy, and the standard library.
- **Dispatch** (`build()` in `main.py`):
  1. k = min(n, N−n); n > N/2 is handled as a complement (prepare D_{N−n}, flip all qubits).
  2. k = 2 with N ≥ 6, or k ≥ 3 with N ≥ 9 → Bryan's vendored solution (`_bryan_wins`).
     On any exception it falls back to ours.
  3. Otherwise:
     - k = 0 → product state;
     - N = 2 → W pair chosen by simulated E[score];
     - k = 1 → W broadcast tree (2N−3 CZ);
     - N ≤ 8 → whole-state synthesized circuit (`_SYNTH`, with robust angle picks);
     - k ≤ 5 → SCS-block Bärtschi–Eidenbenz (`_SCS`);
     - else → routed B-E.
  4. DD (XY4) is applied only to idle windows with ≥ 4 free single-qubit slots.
  5. Every placement uses error rate + 1σ.
- **Simulated results** (idle noise, 3 calibrations, M = N; `portfolio.csv`):
  - Σ E[score] **102.0**, Σ median **27.9** over the 25 cases N ≤ 10, 1 ≤ n ≤ N/2. The
    perfect-state median ceiling is 33.1; the readout-limited optimum is 32.4.
  - Bonus `score_solution` (N ≤ 15): **37.2**.
- **Four reference cases** (`four_cases_merged.csv`):

| case | construction | median | perfect median | % of perfect |
|---|---|---|---|---|
| D(5,2) | ours, 10 CZ | 1.069 | 1.197 | 89% |
| D(6,3) | ours, 15 CZ | 0.959 | 1.094 | 88% |
| D(8,4) | ours, 27 CZ | 0.790 | 1.024 | 77% |
| D(10,5) | Bryan's, 85 CZ | 0.391 | 1.006 | 39% |

- **Safety:** the hardening sweep (969 builds, N ≤ 30, up to 15% dropout) is all valid, with
  none over 1,750 moments and a worst build of 8.2 s. 256 tests pass.

## 2. The problem being worked on: big-N drop-off

**Diagnosis** (Bryan's circuits, which are what we deploy at N ≥ 9; `portfolio.csv`):

| case | CZ | duration | f | x (M=N) | median (M=N) | median (M=4) | correlator ratio at weights 2 … max |
|---|---|---|---|---|---|---|---|
| D9,3 | 50 | 1.83 µs | 0.777 | 0.225 | 0.651 | 0.702 | 0.81 … 0.67 |
| D9,4 | 63 | 2.37 µs | 0.734 | 0.523 | 0.504 | 0.613 | 0.76 … 0.56 |
| D10,3 | 59 | 1.83 µs | 0.748 | 0.252 | 0.619 | 0.671 | 0.80 … 0.65 |
| D10,4 | 77 | 2.37 µs | 0.693 | 0.618 | 0.481 | 0.602 | 0.75 … 0.52 |
| D10,5 | 85 | 2.64 µs | 0.668 | 1.020 | 0.391 | 0.564 | 0.72 … 0.45 |

- **One quantity, overall fidelity (~0.6–0.75), drives both terms.**
  - z ≈ 1/f.
  - x ≈ Σ over even weights of (1 − ratio)², so it grows with the number of even weights
    up to M. That's why D10,5 scores 0.39 at M = N but 0.56 at M = 4.
  - Imbalance is fine (χ²/(K−1) ≈ 1.02–1.09).
- **Gates and duration matter about equally.** For D10,5:
  - gate noise only → median 0.53;
  - idle T1/Tφ only → 0.59;
  - both → 0.375.
  For D9,3 the corresponding numbers are 0.79, 0.81, and 0.65. So the plan must cut **both
  CZ count and circuit duration**.
- **There's evidence of headroom.** At N ≤ 8, our whole-state synthesis beats Bryan's split
  roughly 2× in CZ (D8,4 27 vs 49, D7,3 25 vs 32, D6,3 15 vs 23). The readout-limited median
  at N = 9–10 is ~0.96–0.98, so these cases sit at 25–65% of what's achievable.

**Scope (Richard's call):** aim for N ≤ 15, because the bonus is bounded at 15 and the main
track is probably ~10–12. If N = 13–15 turns out poorly, fall back to 12, then 10. Run the
heavy jobs on the VM overnight.

## 3. Plan (in order)

### P0 — Freeze small N (do first, ~20 min)
- Add `tests/test_golden_small_n.py`. Snapshot, for every N ≤ 8 and 0 ≤ n ≤ N on calibration
  seeds 0–2, the built circuit's CZ count, moment count, qubit set, and a hash of
  `repr(circuit)` into `tests/golden_small_n.json` (generated from the current `richard`
  head). The test asserts nothing changed.
- Every big-N change must keep it green. Update the golden file only deliberately, for a case
  that measured strictly better, with before/after numbers in the commit message.

### P1 — Measure big N beyond 10 (~45 min)
- **N = 11–12:** the exact density-matrix path still works (4¹² entries, a few minutes per
  case). Run `experiments/evaluate.py --nmin 11 --nmax 12` with a separate `--out` per job,
  for a representative set of n (1, 2, 3, ⌊N/4⌋, ⌊N/2⌋).
- **N = 13–15:** build-only stats (CZ, moments, duration) plus a new trajectory evaluator,
  `experiments/trajectory.py`:
  - run `cirq.Simulator().run(circuit.with_noise(CalibratedNoise(...)), repetitions=10000)`
    in each basis (an H-equivalent PhasedXZ layer for X), apply readout flips classically,
    and score with `scoring.fast_overall_score`;
  - give each case its own process (about 10 min each at N = 15);
  - this yields single-draw scores, not the analytic E[score]. Run 3–5 draws for a spread.
- Output: a "score vs N" curve per k. That decides whether 13–15 is worth more effort.

### P2 — Whole-state re-synthesis for N = 9–10 (the main lever; start ASAP, runs 1–3 h)
```bash
.venv/bin/python -u experiments/synth.py --cases 9,2 9,3 9,4 10,2 10,3 10,4 10,5 \
    --shapes path ladder --seeds 0 1 --workers 28 > experiments/results/synth/run_n910.log 2>&1 &
.venv/bin/python -u experiments/synth.py --cases 8,3 --shapes path --seeds 1 2 3 --workers 3 \
    > experiments/results/synth/run_d83.log 2>&1 &   # D8,3 is stuck at 43 CZ (D8,4 needs 27)
```
- That's 28 + 3 jobs, or trim the seeds to fit 30 vCPUs. `synth.py` writes one JSON per
  (N, k, shape, seed) to `experiments/results/synth/`.
- Expect N = 10 to need ~20–30 CZ layers (`max_layers=40` should be enough). Pruning
  dominates the runtime.
- Then run `synth.py --export`, which picks the fewest-CZ entry per (N, k) and re-verifies it.
- **Important:** the current dispatch sends N ≥ 9 / k ≥ 2 to Bryan *before* the `_SYNTH`
  table is consulted, so new N = 9–10 entries do nothing until P4.

### P3 — Warm-start compression of Bryan's circuits (scales past N = 10; ~1.5 h dev + compute)
- New `experiments/compress.py`:
  1. Take Bryan's exact circuit without DD (`compile_layers(..., dd=False)` on the
     split/path ops for a fixed layout).
  2. Express it on its abstract shape, as relative grid coordinates.
  3. Parametrize every merged single-qubit slot as a U3 (complex; his gates use S and Rx,
     so the Ry-only synth ansatz will not do).
  4. Greedily delete CZs, re-optimizing all angles to infidelity < 1e-13 after each
     deletion, with a warm start and an exact backprop gradient (a complex version of
     `experiments/synth.Circuit`).
- Measured cost of one forward pass: 1.3 ms at N = 9 (50 CZ, 216 single-qubit gates) and
  2.8 ms at N = 10 (85 CZ, 371). That's ~8–17 s per re-optimization and ~1 h per case;
  N = 11–12 is roughly 2–4× slower.
- Export a `_COMPRESSED` table: shape coordinates, CZ layers, and U3 params per layer.
  Generalize `_place_synth` / `_synth_circuit` to U3.
- Targets: D9,3 … D10,5 first, then N = 11–12 at k = 3 … 6.

### P4 — Selection (~45 min)
- Replace the static `_bryan_wins` with a measured per-(N, k) choice among the exact
  candidates: `_SYNTH` (placed), `_COMPRESSED` (placed), Bryan's dynamic construction, and
  `_SCS`.
- Decide each winner with the harness (3 calibrations, idle noise, compared on median and
  E[score]). Encode it as a small table in `main.py`, falling back to the next candidate if
  placement fails.
- For N above anything measured, prefer Bryan's construction, which has shown the best
  scaling. Keep k = 1, N ≤ 8 untouched (the golden test enforces this).
- Optionally use Bryan's `estimate_log_fidelity` (already in `main.py`) as a build-time
  tie-breaker among candidates of similar CZ count.
- Run robust angle picks (`experiments/vqe.py spread/coherent/pick/apply`, then
  `synth.py --export`) on any new `_SYNTH` entries.

### P5 — N = 13–15 (stretch, only if P1 says it matters)
- A deeper divide-and-conquer: a 4-arm split (Bärtschi–Eidenbenz QCE 2022, arXiv:2207.09998)
  using synthesized "weight-distribution" blocks. Those blocks are isometries mapping a unary
  count to its hypergeometric split; `experiments/scs_synth.py` can be adapted to synthesize
  them.
- This halves the arm length again, so duration drops. Do it only if P1 shows N = 13–15
  scores worth fighting for.

### P6 — Verify and ship (~45 min, before the morning)
1. `pytest tests/` (including the golden test).
2. `experiments/harden.py --workers 28`, which must stay all-valid with no over-moment cases.
3. Re-evaluate the 25-case set plus N = 11–12 against `portfolio.csv`.
4. Update ARCHITECTURE.md and README.md with new rows and totals, plus `figures.py` if a
   figure changes.
5. Log the prompts and push `richard`. Merge to `main` only when Richard asks.

**Expected gain** if P2/P3 cut N = 9–10, k ≥ 3 circuits to ~55–65% of their current CZ count
and duration: D10,5 0.39 → ~0.55, i.e. about +0.6–0.8 summed over the five big cases (~+2.5%
of the total). The gain is larger if the judges go to N = 11–12.

## 4. Remaining deliverables (Sunday morning)

- Slides (7 min). Suggested story:
  1. Score-tail insight (`figures/score_tail.png`).
  2. Constructions: tree, synthesis, SCS, portfolio.
  3. Before/after (`ceiling_fraction*.png`, `cz_counts*.png`).
  4. Robustness: picks, DD, ±1σ, hardening.
  5. The honest hardware gap.
- ≤ 4-min video. Final README pass. **`prompts.md` must contain every prompt**, including
  the ones in this VM session.

## 5. Gotchas

- **Exports overwrite tables in `main.py`** between `BEGIN`/`END` markers:
  - `synth.py --export` → `_SYNTH` (applies `results/vqe/picks.json` only when the deployed
    structure and solution match);
  - `scs_synth.py --export` → `_SCS`;
  - `vqe.py apply` → `picks.json`, which needs a following `synth.py --export`.
- **Bryan's code is vendored verbatim**, with renames `_X → _BRYAN_X`, `_Y → _BRYAN_Y`,
  `create_dicke_state(_adaptive) → _bryan_create_dicke_state(_adaptive)`. Our `_X`/`_Y` are
  DD gates; don't mix them up.
- `evaluate.py --opt key=value` passes options to `main.build`. Use `bryan=false`,
  `synth=false`, `scs=false`, `tiny=false`, `dd=false`, or `alap=true` for ablations.
- `experiments/summarize.py` reports f, χ²/(K−1), x, loss, E[score], median, and % of
  ceiling.
- **The noise model's idle numbers are assumptions:** Tφ = 100 µs and 25/42 ns gate
  durations. T1 = 68 µs is from the spec sheet. Big-N conclusions depend on these.
- **Unknown judge parameters:** `max_correlator_size` (we report M = N; M = 4 is in the
  CSVs) and the (N, n) set.
- **Teammates' branches:** `origin/bryan` (already vendored) and `origin/Aanha` (not yet
  looked at).
