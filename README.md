# Dicke states on Google Willow — Team Popeyes (DuQuantum × Google Quantum AI)

`main.py` implements `create_dicke_state(N, n, cz_pauli_error_rate, sq_pauli_error_rate,
zero_errors, one_errors)`, returning an exact circuit (CZ + PhasedXZ only, on calibrated
qubits and couplers) that prepares |D^N_n⟩ on Willow 105, and the set of qubits it lives on.

## What it does

| case | construction | CZ count |
|---|---|---|
| n = 0 or n = N | product state (nothing, or X on all) | 0 |
| n = 1 or n = N−1 | broadcast tree of amplitude splits (first split 1 CZ, then 2-CZ Givens) | 2N − 3 |
| small N, 2 ≤ min(n, N−n) | exact Ry + CZ circuit found offline by numerical synthesis, placed at build time | e.g. D⁴₂: 5, D⁶₃: 15 |
| everything else | Bärtschi–Eidenbenz, routed on a low-error path with `cirq.RouteCQC` | O(nN) |

For n > N/2 we prepare D^N_{N−n} and flip every qubit. The Dicke state is
permutation-symmetric, so routing permutations never need undoing; we return wherever the
logical qubits end up. Every calibration-dependent choice (which qubits, which placement) is
made at build time from the calibration dicts. The objective includes calibrated gate and
readout errors, an idle-time term, and the χ² noncentrality caused by readout-asymmetry
spread between qubits.

## Why these choices (short version; details in CLAUDE.md)

- The judges' per-case score is 1/(z + x), and z is a χ² statistic, so it is heavy-tailed.
  Each case has a shot-noise ceiling: the median score of a perfect state ≈ K / median(χ²_{K−1}),
  with K = C(N, n). We report every result as a fraction of that ceiling.
- For small K, population imbalance is what destroys the score (χ² noncentrality
  λ = 10000·f·mean(δ²)). For larger n, the X term (coherence) is as large as the Z term, so
  CZ count is the main lever.

## Reproducing the numbers

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow -r requirements.txt  # Willow105 needs cirq 1.8 dev
.venv/bin/python -m pytest tests/ -q                     # exactness + rules, N <= 10, all n, 3 calibrations

# simulated scores (exact density matrix + 20k simulated 10k-shot runs), one CSV row per case
.venv/bin/python experiments/evaluate.py --out mine.csv --nmin 2 --nmax 8 --noise idle --repeats 20000
.venv/bin/python experiments/summarize.py mine.csv       # f, chi2/(K-1), x, loss, median, % of ceiling

# ablations: --opt passes options to main.build
.venv/bin/python experiments/evaluate.py --out abl.csv --label chain --opt w_tree=false --only-w --nmax 10
.venv/bin/python experiments/evaluate.py --out abl.csv --label B-E --opt synth=false --nmin 4 --nmax 6

# regenerate the synthesized-circuit table embedded in main.py
.venv/bin/python experiments/synth.py --cases 4,2 5,2 6,2 6,3 --shapes path ladder
.venv/bin/python experiments/synth.py --export
```

Calibrations are the notebook's fake-data generators with fixed seeds (`experiments/calib.py`,
seeds 0, 1, 2).

## Noise model (experiments/noise.py)

- Gate errors from the calibration dicts: depolarizing after each CZ and each non-virtual
  PhasedXZ (pure-Z gates are virtual: no error, no time).
- Idle decoherence each moment: T1 = 68 µs (Willow spec sheet), T_φ = 100 µs (assumed),
  25 ns single-qubit and 42 ns CZ layers (assumed).
- Optional quasi-static detuning (T2* = 20 µs, assumed) for the dynamical-decoupling study.
- Readout: asymmetric per-qubit bit flips from `zero_errors` / `one_errors`, in both bases.
- Not modeled: coherent gate errors, ZZ crosstalk, leakage, per-qubit T1 spread, drift.

## Layout

```
main.py                 submission: create_dicke_state + helpers (cirq, cirq_google, numpy, networkx, stdlib)
experiments/calib.py    notebook calibration generators (verbatim) + seeded wrapper
experiments/noise.py    calibrated noise model, detuning copies, readout confusion
experiments/scoring.py  judges' scorer (verbatim) + fast equivalent + score statistics + ceiling
experiments/validity.py challenge rules (stricter than Willow105.validate_circuit)
experiments/evaluate.py simulate + score + CSV
experiments/summarize.py seed-averaged, ceiling-normalized tables
experiments/synth.py    offline exact synthesis of small Dicke circuits
experiments/results/    every CSV behind the slides
tests/                  pytest suite
prompts.md              every AI prompt used (sponsor requirement)
```
