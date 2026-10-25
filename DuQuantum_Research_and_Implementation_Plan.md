# DuQuantum: Research and Implementation Plan

Prepared October 24, 2026. Based on `DuQuantum_Google_Standard.ipynb`.

This document exports the research recommendations and implementation plan from the conversation. It proposes work; it does not report completed implementations or hardware benchmarks.

**The most promising strategy is to compare several exact circuit constructions, then optimize their placement and noise suppression on Willow.** The best theoretical depth will not necessarily produce the best hardware score.

The notebook imposes several decisive constraints: exact noiseless preparation, restricted CZ connectivity, only `PhasedXZGate` single-qubit gates, construction in under a minute, and at most 1,750 moments for the hardware track. The adaptive bonus needs a separate optimization strategy. Submission code may import only Cirq, cirq_google, their dependencies, and Python's standard library.

The reading list covers research available through October 24, 2026. Throughout, N denotes the number of qubits and k the excitation count.

**The research to prioritize is:**

1. **Yuan and Zhang — [Depth-Efficient Quantum Circuit Synthesis for Deterministic Dicke State Preparation](https://arxiv.org/abs/2505.15413), 2025.**

   This is the most directly relevant recent paper. It develops exact, deterministic constructions specifically for two-dimensional grids, without additional ancilla qubits. The central idea is to distribute excitations between blocks and prepare independent blocks in parallel.

   For an a × b grid, a ≤ b, it obtains depth O(k log(N/k) + b) in one regime and O(b) for sufficiently small k.

   **Use:** implement a parallel, grid-aware candidate.

   **Caution:** Willow's irregular boundary and missing couplers mean the rectangular-grid bounds are not guaranteed to survive placement and routing. Measure the compiled circuit.

2. **Raveh and Nepomechie — [Dicke states as matrix product states](https://arxiv.org/abs/2408.04729), 2024.**

   The paper expresses Dicke states as exact matrix product states, with bond dimension min(k, N−k) + 1. This compact representation describes how to prepare the state sequentially: when r excitations remain across m sites, the amplitude for emitting a 1 is √(r/m).

   **Use:** derive exact preparation angles, build an independent correctness reference, and investigate a sequential counter-register implementation.

   **Caution:** its circuit uses a multilevel ancilla. Encoding that ancilla into Willow qubits adds controls and routing; the supplied Cirq simulation is not directly a legal submission.

3. **Wang, Tan, Cong, and De Micheli — [Quantum State Preparation Using an Exact CNOT Synthesis Formulation](https://arxiv.org/abs/2401.01009), DATE 2024.**

   This work searches for preparation circuits with fewer CNOT gates. It reports a factor-of-two improvement for a particular Dicke-state example, suggesting that general constructions can leave substantial room for small-instance optimization.

   **Use:** develop a library of verified circuits for small (N, k) cases or recurring preparation blocks. Convert CNOTs into CZs and merge adjacent single-qubit rotations.

   **Caution:** perform expensive searches offline. The one-minute constructor should load and place templates, not solve a difficult synthesis problem. The reported improvement is not a universal guarantee.

4. **Evert et al. — [Syncopated Dynamical Decoupling for Suppressing Crosstalk in Quantum Circuits](https://arxiv.org/abs/2403.07836), 2024, revised 2025.**

   Dynamical decoupling inserts pulses during idle periods to suppress unwanted evolution. This paper studies coordinating different pulse patterns across neighboring qubits to suppress both decoherence and unwanted coupling.

   **Use:** compare staggered or syncopated patterns against ordinary decoupling. This is particularly relevant to the X-basis score, which detects coherence errors that Z-basis populations can miss.

   **Caution:** added pulses also have errors. Benefits depend on the device and timing; decoupling should be an experimentally evaluated option.

5. **Tong, Zhang, and Pokharel — [Empirical learning of dynamical decoupling on quantum processors](https://arxiv.org/abs/2403.02294), 2024 preprint, revised 2025.**

   This paper searches for decoupling strategies using feedback from actual circuit performance, rather than assuming one sequence works best everywhere.

   **Use:** search a small menu of pulse patterns and insertion rules during development, then retain the best policies for different circuit families.

   **Caution:** without hardware access, the selected policy is only as reliable as the noise models used to choose it. Avoid training exclusively against independent depolarizing noise.

6. **Huo et al. — [Revisiting Noise-adaptive Transpilation in Quantum Computing: How Much Impact Does it Have?](https://arxiv.org/abs/2507.01195), 2025.**

   Across several IBM processors, the authors find that calibration-aware mapping is not uniformly superior: lightweight or randomized alternatives can have comparable average performance, and calibration-driven choices can show substantial variability.

   **Use:** compare calibration-aware placement with topology-only and randomized placement. The notebook supplies uncertainty estimates, so test whether uncertainty-aware selection improves reliability.

   **Caution:** these are IBM results, not evidence that Willow calibration should be ignored. Their value here is motivating controlled comparisons instead of assuming the mapping heuristic is correct.

7. **Buhrman et al. — [State preparation by shallow circuits using feed forward](https://quantum-journal.org/papers/q-2024-12-09-1552/), Quantum, 2024.**

   This is the strongest starting point for the adaptive bonus. It develops exact preparation using nearest-neighbor gates, measurements, and classical corrections, including shallow W-state and low-weight Dicke-state constructions.

   **Use:** investigate binary-to-one-hot encoding, measurement-assisted cleanup, and teleportation of serial Clifford operations.

   **Caution:** shallow depth comes with substantial extra-qubit requirements. Willow's 105-qubit limit and diagonal readout restrictions must be checked explicitly; the paper's grid model does not automatically satisfy them.

Two newer results are useful mainly for identifying approaches that require further adaptation:

- **Yu et al., [Efficient Preparation of Dicke States](https://journals.aps.org/prl/abstract/10.1103/9gjk-rgql), 2026:** uses collective measurements and feedback with efficient expected preparation time. Finite truncation leaves failure branches, and collective Hamming-weight measurement is not equivalent to measuring every data qubit individually. It is not directly a bounded, deterministic solution to this notebook. [Open preprint](https://arxiv.org/abs/2411.03428).
- **Luo, Chen, and Ma, [Constant-depth adaptive preparation of Dicke and symmetric states](https://arxiv.org/abs/2608.01144), August 2026 preprint:** produces an exact state conditional on success, with significant ancilla overhead. Parallel repetition reduces failure probability but does not eliminate it. That distinction matters for the notebook's exactness requirement.

For a reliable baseline, retain the older **Bärtschi–Eidenbenz [2019 construction](https://arxiv.org/abs/1904.07358)**: exact, ancilla-free preparation with O(Nk) gates and linear depth on a nearest-neighbor line. Their **[2022 short-depth construction](https://arxiv.org/abs/2207.09998)** provides an intermediate parallel approach before implementing the more elaborate 2025 method.

**Turn this reading into the following implementation, testing, and evaluation plan.**

1. **Repair the evaluation harness and establish the rules.**

   The notebook contains issues that could invalidate comparisons:

   - `keep_indices = range(qubits)` raises an exception because `qubits` is a set; use `range(len(qubits))`.
   - Reset and adaptive circuits require checking the reduced output state or measurement branches. One sampled state-vector trajectory cannot establish deterministic correctness.
   - The bonus description says N ≤ 15 with points for the shallowest circuit, while its code uses N ≤ 20 and sums Nk/depth. Implement coverage through 20 and track both interpretations until clarified.
   - The standard test cases and maximum X-correlator order are unspecified; make both configurable.

   Preserve the official scoring calculation while recording undefined results, such as zero surviving weight-k shots, as evaluation failures.

2. **Implement a small portfolio with shared compilation and validation.**

   | Candidate | Research basis | Purpose |
   | --- | --- | --- |
   | Exact path-based preparation | Bärtschi–Eidenbenz 2019 | General fallback and correctness baseline |
   | Parallel block preparation | 2022 and Yuan–Zhang 2025 | Reduce depth using grid parallelism |
   | Specialized W-state circuits | Exact excitation splitting | Exploit the simpler k = 1 case |
   | Small-instance templates | Wang et al. 2024 | Reduce native gate count for common cases |
   | Sequential counter-register preparation | Raveh–Nepomechie 2024 | Independent alternative and reference |

   Keep logical construction, placement, native compilation, validation, and scoring separate. This lets each component improve without rewriting the entire solution.

   Use the complement identity below to reuse low-weight constructions for high-weight requests, counting the added gates:

   $$
   |D_k^{(N)}\rangle = X^{\otimes N}|D_{N-k}^{(N)}\rangle.
   $$

3. **Make placement and compilation reflect the actual hardware.**

   Construct the usable graph from the supplied calibration dictionaries. Compare:

   - topology-only placement;
   - placement minimizing estimated gate and readout error;
   - uncertainty-aware placement;
   - several randomized placements.

   Rank placements using the circuit's actual gate multiplicities. A low-error edge used repeatedly matters more than one used once. Include asymmetric readout errors: for an ideal Dicke state, each qubit has excitation probability k/N.

   Compile to full CZ gates and `PhasedXZGate` operations, then revalidate every edge and gate. Cirq's [`CZTargetGateset(allow_partial_czs=False)`](https://quantumai.google/reference/python/cirq/CZTargetGateset) is a useful starting point, but the challenge still needs its own stricter validator.

   Keep expensive searches outside the constructor. Use bounded candidate generation and leave a comfortable margin below 60 seconds.

4. **Establish exact correctness before evaluating noise.**

   For small sizes, test every excitation count. At larger sizes, cover k = 1, 2, floor(N/4), floor(N/2), and complements.

   Check:

   - target fidelity, including relative phases;
   - absence of unwanted entanglement with ancillas;
   - equivalence before and after compilation;
   - legal qubits, couplers, gates, and output-set size;
   - behavior under missing couplers and disconnected hardware;
   - construction time and compiled moment count.

   Numerical fidelity checks should support an analytic correctness argument. Variational circuits that merely approach the Dicke state do not meet the exactness requirement.

5. **Evaluate noise suppression through controlled comparisons.**

   Start with no decoupling, then compare ordinary and staggered sequences. Cirq supplies an [insertion transformer](https://quantumai.google/reference/python/cirq/add_dynamical_decoupling), but the resulting schedule still needs native-gate and exactness checks.

   Evaluate progressively richer noise models:

   - gate-dependent Pauli errors and asymmetric readout;
   - idle dephasing and amplitude damping;
   - coherent over-rotation and residual ZZ coupling;
   - calibration perturbations consistent with supplied uncertainties.

   The notebook's error rates do not determine all these noise parameters. Treat unmeasured coherence and crosstalk strengths as sensitivity scenarios.

   Compare every approach on the same calibration instances and noise seeds. Use small density-matrix simulations and larger trajectory or tensor-network simulations where feasible; the target state's compact MPS does not make arbitrary noisy circuits inexpensive to simulate.

6. **Measure performance using the official sampling budget and additional diagnostics.**

   Use **10,000 Z shots and 10,000 X shots** per evaluation, repeated across independent trials. Include ideal sampled Dicke states as a reference.

   There is a subtle scoring issue: writing S for total shots, m for weight-k shots, M = binomial(N, k), and c_i for their bitstring counts, the notebook's Z loss simplifies to

   $$
   L_Z = S\left[\sum_i(c_i/m)^2-\frac{1}{M}\right].
   $$

   For conditionally uniform samples,

   $$
   \mathbb{E}[L_Z\mid m] = \frac{S}{m}\left(1-\frac{1}{M}\right).
   $$

   Thus, even ideal finite-shot data typically has Z loss near **1** when M is large. Leakage affects this score indirectly through sampling, so track survival separately.

   This identity also enables a fast equivalent scorer using observed counts. Likewise, the averaged X correlators can be computed from Hamming-weight histograms using Krawtchouk polynomials, avoiding enumeration of every qubit subset. Both identities were checked against direct enumeration on small examples during the research review.

   | Metric | What it establishes |
   | --- | --- |
   | Official score distribution | Competition performance and variability |
   | Small-instance fidelity | Correctness and state quality |
   | Weight-k survival | Excitation loss or gain |
   | X-correlator error | Coherence preservation |
   | Native CZ count and moments | Physical resource cost |
   | Construction time and supported cases | Submission reliability |

   Choose candidates using held-out calibration instances. Report confidence intervals and nonfinite runs; do not select a winner from one favorable shot sample.

7. **Develop the adaptive bonus independently, starting with W states.**

   A concrete experiment inspired by the feedforward literature is:

   - Prepare an exact uniform superposition of indices j = 0, …, N−1.
   - Coherently decode each index into a one-hot output |e_j⟩.
   - Measure the index register in X.
   - Correct the resulting known phases with parallel single-qubit Z operations.

   The cleanup follows because measurement outcome s introduces phase (−1)^(s·j), removable on output qubit j. This is a proposed design to test, not a completed Willow implementation.

   Reserve appropriate readout diagonals for measured ancillas, verify every nonzero-probability branch, and count all preparation, routing, measurement, and correction moments. Retain the best nonadaptive circuit whenever the adaptive version is deeper.

   The final submission should select among verified candidates by (N, k) and calibration data, include reproducible benchmark settings, and preserve the AI prompts as the notebook requests.
