# CLAUDE.md — DuQuantum × Google Quantum AI Challenge (Standard Track)

Context for anyone (human or AI) working in this repo. Source of truth for the task
is `[standard_track] Google Quantum AI Challenge.ipynb`; this file summarizes it and
records our analysis and plan.

## Timeline

- Hackathon: DuQuantum, Duke. Work started Saturday, Oct 24, 2026.
- Hard deadline: **Sunday, Oct 25, 11:30 AM** — all three deliverables ready.
- Judging: 1.5 h session, 7-minute live talk to Google Quantum AI reps + DuQuantum judges.

## The task

Implement `create_dicke_state(num_qubits, num_excitations, cz_pauli_error_rate,
sq_pauli_error_rate, zero_errors, one_errors) -> (cirq.Circuit, set[cirq.GridQubit])`
that prepares the Dicke state |D^N_n⟩ (uniform, equal-phase superposition of all
N-bit strings with exactly n ones; n=1 is the W state) on Google's Willow chip.

Hard constraints (violations = circuit is skipped):

- Gates allowed: `cirq.CZ` only on pairs present in `cz_pauli_error_rate`;
  `cirq.PhasedXZGate` only on qubits present in `sq_pauli_error_rate`; optional
  mid-circuit `cirq.ResetChannel`. Nothing else. No classical conditioning.
- Noiseless simulation must give the exact state (fidelity 1, phases included) on the
  returned qubit set, or the function raises `NotImplementedError`.
- Build time < 60 s per call. Circuit ≤ 1750 moments.
- Imports: `cirq`, `cirq_google`, their dependencies (numpy is fine), Python stdlib.
- Device: `cirq_google.Willow105` (105-qubit grid). Use `Willow105.metadata.qubit_set`
  and `.qubit_pairs` for topology; `cirq_google.Willow105.validate_circuit(c)` for routing.
  The calibration dicts define which qubits/couplers are usable on judging day; some
  are missing (dropouts).

## How it is scored (hardware leaderboard)

Judges run our circuit on real Willow for undisclosed (N, n) pairs, 10,000 shots in
the Z basis and 10,000 in the X basis, compute `overall_score = 1/(z_term + x_term)`
per pair, and sum. Highest total wins.

- **Z term** = chi² / (K · f), where K = C(N,n), f = fraction of shots with exactly n
  ones, and chi² is the goodness-of-fit of the K valid-string counts against uniform.
- **X term** = Σ_k (relative error)² of the average weight-k parity correlator
  ⟨X_{i1}…X_{ik}⟩ vs. the ideal (0 for odd k, a closed form for even k), k = 1..M
  (M undisclosed). Absolute error is used when ideal is 0.

What this implies (derive these on a slide; numbers checked by Monte Carlo):

- **The score is heavy-tailed; small-K cases are worth far more than 1.** For a
  perfect state, chi² follows a χ²_{K−1} distribution, so z = chi²/(K·f) has mean
  ≈ (1 − 1/K)/f. But score = 1/(z+x), and E[1/χ²_{K−1}] = 1/(K−3), which gives
  E[score] ≈ K·f/(K−3) for K > 3. For K ≤ 3 the mean is capped only by the X term x.
  Mean score (f=0.9, x=1e-3, 10k shots):
  K=2 → 51, K=3 → 8.2, K=4 → 3.4, K=6 → 1.8, K=10 → 1.3, K≥20 → ≈1.
  For K=2: x=1e-4 → 164, x=1e-2 → 15. "≈1 per case" holds only for large K.
- **Systematic population imbalance kills the tail.** chi² becomes noncentral with
  λ = 10000·f·mean(δ²), where δ_i is the relative deviation of valid string i. This adds
  ≈ 10000·mean(δ²)/K to z. With δ_rms = 3%, the K=2 mean drops from 51 to 1.4 and the
  K=4 mean from 3.4 to 0.5. Target δ_rms ≲ 1% (λ ≲ 1).
- Leakage out of the n-excitation subspace costs only linearly (1/f).
- The X term is lenient for large K: a 20% shrink of a 0.5-valued correlator adds
  0.04. For K ≤ 3, though, x sets the cap on the lucky low-chi² draws, so coherence matters.
- K=1 cases (n=0 or n=N) have chi² ≡ 0, so their score is 1/x (huge). Never raise on them.
- **Priority order: valid on every (N,n) → small-K uniformity + X floor → leakage (f)
  → coherence at large N → bonus.** Never raise `NotImplementedError` if any valid
  layout exists: a noisy circuit still scores > 0.
- The target is permutation-symmetric and the returned qubits are a set, so routing
  permutations never need undoing. For n > N/2, prepare D_{N−n} and flip all qubits.
- Readout imbalance: δ_i ≈ −(d_i − mean d) with d = e1 − e0. A random layout gives
  δ_rms ≈ 0.6%, so for small K, pick readout-matched qubits. Pre-compensating amplitudes is
  forbidden because the noiseless state must be exact.

Sources of imbalance to watch: miscalibrated rotation angles, qubit-to-qubit readout
error differences (1→0 errors ≈ 2× the 0→1 rate), and noise-induced redistribution
inside the subspace. Excitation-preserving constructions (B-E, Givens trees) keep n
excitations in every branch for the whole circuit, so T1 exposure is automatically
balanced across branches. Keep this as a check, not a work item.

## Judging (beyond the score)

Deliverables on Devpost: (1) 7-min live presentation/demo, (2) full repo,
(3) ≤4-min video. (1) and (2) weigh most. Rubric emphasizes understanding of the
problem and theory, ownership of the thought process (explicitly: not an LLM's),
reproducible numerics showing the solution works, and what was learned.

**AI use is allowed but every prompt must be included in the submission.** Keep
`prompts.md` current. Design decisions are Richard's to make and defend; the AI's role
is scaffolding, tooling, and explanation.

## Approach

1. **Harness first.** Noise model from the calibration dicts (depolarizing after each
   CZ and single-qubit gate at the given rates; asymmetric bit-flip readout using
   `zero_errors`/`one_errors`), sampling in both bases (append H before measuring for
   X), the judges' `overall_score` verbatim, plus a decomposition: f, uniformity chi²,
   X error, CZ count, depth. Fixed seeds everywhere.
2. **Baseline circuit:** Bärtschi & Eidenbenz, "Deterministic Preparation of Dicke
   States" (FCT 2019, arXiv:1904.07358): O(N) depth, O(nN) gates, built from
   controlled-RY blocks. Gate (ii)_ℓ spans qubits m−ℓ, m−ℓ+1, m, so it is not
   nearest-neighbour. The paper's LNN version sifts qubit m with SWAPs (§3). We instead
   build the logical circuit, compile to CZ with
   `cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset())`, route with
   `cirq.RouteCQC` on the calibrated subgraph, and recompile. Verify exact for all
   N ≤ ~10, all n, and `Willow105.validate_circuit` passes.
3. **Improvements, one at a time, each with before/after numbers:**
   - Layout search: cost = Σ CZ error × uses + Σ single-qubit error × uses + readout
     error weighted by how often each qubit holds a 1 vs 0. Best fixed-length path
     (or subtree) on the usable grid via pruned DFS / beam search. Lognormal spread
     means best coupler ≈ several× better than median.
   - Dynamical decoupling on idle qubits (`cirq.add_dynamical_decoupling` or manual
     XY4). The simulator's Pauli model under-represents what DD fixes on real
     hardware; keep DD even if simulated gain is small.
   - Tree construction for n=1: split amplitude outward from a root. The first split
     costs 1 CZ (product input). Every later split costs exactly 2 CZ (a Givens rotation;
     1 CZ can't map a 2-dimensional input subspace). Total 2N−3 CZ, the same as a chain,
     but depth is ~log N rounds instead of N. The gain only shows up with idle noise in
     the simulator (T1 = 68 ± 13 µs per the Willow spec sheet).
   - DD only on qubits that are in superposition. Untouched |0⟩ qubits need none, so
     schedule ALAP to keep qubits in |0⟩ as long as possible.
   - Follow-up reference: Bärtschi & Eidenbenz, "Short-Depth Circuits for Dicke State
     Preparation" (QCE 2022, arXiv:2207.09998) — divide-and-conquer; check whether it
     maps onto a 2D patch.
   **Status (Sat Oct 24, afternoon; branch `richard`).** Implemented and measured
   (fractions are of the perfect-state median ceiling, simulated):
   - k = min(n, N−n) = 1: broadcast tree, 2N−3 CZ, 89–100% of ceiling for N ≤ 10.
   - N ≤ 8, k ≥ 2: whole-state exact Ry+CZ synthesis (`experiments/synth.py`), e.g.
     D⁶₃ goes from 80 to 15 CZ (48% → 85%). Angles are cached in `main.py` because
     synthesis takes minutes; placement is decided at build time.
   - k ≤ 4, larger N: B-E built from synthesized SCS blocks on a path
     (`experiments/scs_synth.py`), with no routing.
   - Otherwise routed B-E. DD (XY4) is applied only to long idle windows (Richard's call;
     see `experiments/results/dd*.csv`).
4. **Freeze late Saturday.** Bonus only if main track is solid.
5. **Sunday AM:** clean `main.py`, README, figures, slides, video, prompt log.

Do not try to invent a new Dicke construction in 24 h. Leverage is in the engineering
layer, measured empirically.

## Bonus track (not run on hardware)

`create_dicke_state_adaptive(num_qubits, num_excitations)` may use mid-circuit
measurement and classically conditioned gates. Scored on depth only: text says a point
per (N,n) to the shallowest submission for 2 ≤ N ≤ 15, 1 ≤ n ≤ ⌊N/2⌋; the shown
`score_solution` sums N·n/len(circuit) for N up to 20 (inconsistency — treat as
"shallow for large N·n wins"). Must produce the exact state on every measurement
branch (no post-selection). Willow readout-line rule: measuring any qubit on an
anti-diagonal (row + col = const) requires measuring all qubits on it — data qubits
and measured ancillas must sit on disjoint anti-diagonals (not checked by
`validate_circuit`). Idea: prepare Dicke states on blocks in parallel, fuse via
ancilla measurements + corrections. Expect log-ish depth, not constant.

## Known issues in the provided notebook

- `test_circuit`: `keep_indices = range(qubits)` crashes (`qubits` is a set). Use
  `range(len(qubits))`.
- Bonus range mismatch (N ≤ 15 in text vs. 20 in code), noted above.

## Repo layout

```
main.py            # ONLY the two required functions + their helpers. Submission file.
experiments/       # noise model, scoring harness, layout search, sweeps, plotting
figures/           # every plot used in slides, regenerated by scripts
tests/             # exactness checks, validate_circuit, moment/time budgets
prompts.md         # every AI prompt used (required by sponsor)
README.md          # how to reproduce every number in the slides
```

## Working conventions

- `main.py` must import only cirq / cirq_google / stdlib / numpy and run standalone.
- Every experiment: fixed RNG seed, parameters logged, result appended to a CSV in
  `experiments/results/`. Those numbers become the slides.
- Before claiming an improvement, show the decomposed metrics, not just the score.
- Cirq APIs shift between versions; verify a transformer/gateset name exists in the
  installed version before relying on it. `cirq_google.Willow105` exists only in the
  cirq 1.8 dev builds (pinned in `requirements.txt`), not in stable 1.7.0.
- Coupler keys from the notebook's generators are `tuple(frozenset)`, so their order is
  arbitrary. Always look up pairs order-independently.
- `Willow105.validate_circuit` accepts more gates (XPow, YPow, H, ...) than the challenge
  allows. Our own tests enforce the stricter whitelist: `cirq.CZ`, `PhasedXZGate`,
  `ResetChannel`.
- Keep `prompts.md` updated in the same commit as any AI-assisted change.
