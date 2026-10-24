# pyright: reportUndefinedVariable=false
# Notebook fragment: executed inside the challenge notebook after the solution and the provided helper cells.
# %% [markdown]
# # DuQuantum noisy emulator and design experiments
#
# Hardware access is not available, so design choices were compared on a density-matrix emulator whose assumptions are
# explicit. Per moment it applies:
#
# * depolarising noise after every gate, with the calibration's Pauli error rate ($p/3$ per Pauli for 1q gates, $p/15$ for CZ);
# * amplitude damping ($T_1=68\,\mu$s) and white dephasing ($T_\phi=100\,\mu$s) on every qubit for the moment's duration;
# * optional **coherent** errors that dynamical decoupling targets: quasi-static detuning (a random frequency offset per qubit,
#   averaged over several realisations) and *spectator phases* (a fixed random $Z$ rotation on an idle qubit whenever a
#   neighbour takes part in a CZ);
# * independent asymmetric readout flips using the calibrated $0\to1$ and $1\to0$ rates.
#
# **Shot-noise-free score.** With 10,000 shots the official $L_Z$ fluctuates by about ±30%, which would swamp design differences.
# Writing $S$ for the shots, $f$ for the fraction of weight-$k$ shots, $M=\binom Nk$ and $q_i$ for the conditional distribution over
# the $M$ weight-$k$ strings, the notebook's $L_Z$ equals $S\big[\sum_i (c_i/m)^2-\frac1M\big]$ with $m=Sf$, so for multinomial counts
# $$\mathbb E[L_Z]=S\Big(\sum_iq_i^2-\frac1M\Big)+\frac{1-\sum_iq_i^2}{f}.$$
# The first term penalises non-uniformity (heavily, because it is multiplied by $S$), the second is $\approx1/f$: even a perfect
# state scores $\mathbb E[L_Z]=(1-1/M)$, so the ideal expected score is $M/(M-1)$. $L_X$ is evaluated at its exact (infinite-shot)
# correlators. Comparisons below use this expected score on identical calibrations.

# %%
EMU_T1 = 68e-6  # amplitude-damping time (s), typical published Willow value
EMU_TPHI = 100e-6  # white pure-dephasing time (s), an assumption
EMU_T_1Q, EMU_T_CZ = 25e-9, 42e-9  # moment durations (s)


class HardwareNoise:
    """Per-moment noise channels for the emulator (see the list above)."""

    def __init__(self, cal, detuning_hz=None, spectator=None):
        cz, sq = cal[0], cal[1]  # calibration dictionaries (means are used as the 'true' rates)
        self.cz_err = {frozenset(k): v[0] for k, v in cz.items()}  # CZ Pauli error per coupler
        self.sq_err = {q: v[0] for q, v in sq.items()}  # single-qubit Pauli error per qubit
        self.detuning_hz = detuning_hz or {}  # qubit -> quasi-static frequency offset (Hz)
        self.spectator = spectator or {}  # (idle qubit, active neighbour) -> Z phase per CZ moment
        self.nbrs = {}  # device adjacency (for spectator phases)
        for pair in self.cz_err:  # every coupler
            a, b = tuple(pair)  # endpoints
            self.nbrs.setdefault(a, set()).add(b)  # undirected ...
            self.nbrs.setdefault(b, set()).add(a)  # ... adjacency

    def moment_ops(self, moment, qubits):
        """Noise operations that follow `moment` on the simulated `qubits`."""
        dur = EMU_T_CZ if any(len(op.qubits) == 2 for op in moment) else EMU_T_1Q  # moment duration
        out = []  # noise operations
        for op in moment:  # gate errors
            if len(op.qubits) == 2:  # CZ
                out.append(cirq.depolarize(self.cz_err[frozenset(op.qubits)], n_qubits=2).on(*op.qubits))  # 2q Pauli noise
            else:  # PhasedXZ
                out.append(cirq.depolarize(self.sq_err[op.qubits[0]]).on(*op.qubits))  # 1q Pauli noise
        if self.spectator and dur == EMU_T_CZ:  # coherent spectator phases during CZ moments
            active = {q for op in moment for q in op.qubits}  # qubits inside CZs
            for q in qubits:  # idle qubits only
                if q not in active:  # spectator
                    phi = sum(self.spectator.get((q, w), 0.0) for w in self.nbrs.get(q, ()) if w in active)  # total phase
                    if phi:  # non-zero
                        out.append(cirq.rz(phi).on(q))  # coherent Z error
        g = 1 - math.exp(-dur / EMU_T1)  # amplitude-damping probability
        lam = 1 - math.exp(-2 * dur / EMU_TPHI)  # phase-damping parameter
        for q in qubits:  # idling noise on every qubit
            out.append(cirq.amplitude_damp(g).on(q))  # T1
            out.append(cirq.phase_damp(lam).on(q))  # T_phi
            if self.detuning_hz.get(q):  # quasi-static detuning
                out.append(cirq.rz(2 * math.pi * self.detuning_hz[q] * dur).on(q))  # coherent Z drift
        return out


def noisy_density_matrix(circuit, noise, qubits):
    """Final density matrix of `circuit` with noise inserted after every moment."""
    noisy = cirq.Circuit()  # noisy circuit
    for moment in circuit:  # original moments
        noisy.append(cirq.Moment(moment.operations))  # the gates
        extra = noise.moment_ops(moment, qubits)  # their noise
        while extra:  # pack noise into non-overlapping moments
            used, this, rest = set(), [], []  # greedy packing
            for op in extra:  # each channel
                if used.isdisjoint(op.qubits):  # fits into this noise moment
                    this.append(op)  # take it
                    used.update(op.qubits)  # occupy its qubits
                else:  # overlaps an earlier channel
                    rest.append(op)  # next noise moment
            noisy.append(cirq.Moment(this))  # one noise moment
            extra = rest  # remaining channels
    return cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(noisy, qubit_order=qubits).final_density_matrix


def _readout_channel(probs, n, e0, e1):
    """Exact outcome distribution after independent asymmetric bit flips (qubit 0 = most significant bit)."""
    p = np.asarray(probs, dtype=float).reshape([2] * n)  # one axis per qubit
    for q in range(n):  # flip each qubit independently
        m = np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]])  # m[observed, true]
        p = np.moveaxis(np.tensordot(m, p, axes=([1], [q])), 0, q)  # apply along axis q
    return p.reshape(-1)  # back to a vector


def expected_score(pz, px, n_qubits, k, e0, e1, shots=10_000, max_corr=4):
    """Shot-noise-free expectation of the official losses (formula above)."""
    pz = _readout_channel(pz, n_qubits, e0, e1)  # observed Z-basis distribution
    px = _readout_channel(px, n_qubits, e0, e1)  # observed X-basis distribution
    weights = np.array([bin(i).count('1') for i in range(2 ** n_qubits)])  # Hamming weights
    f = pz[weights == k].sum()  # fraction in the weight-k subspace
    q = pz[weights == k] / f  # conditional distribution
    M = math.comb(n_qubits, k)  # subspace size
    lz = shots * (np.sum(q ** 2) - 1 / M) + (1 - np.sum(q ** 2)) / f  # E[L_Z]
    lx = 0.0  # exact L_X
    comb = lambda a, b: math.comb(a, b) if (b >= 0 and a >= b) else 0  # binomial with zero outside range
    for c in range(1, max_corr + 1):  # correlator orders
        meas = 0.0  # average c-body correlator
        for idx in itertools.combinations(range(n_qubits), c):  # all subsets
            parity = np.zeros(2 ** n_qubits, dtype=int)  # parity of the subset per basis state
            for qb in idx:  # accumulate parity
                parity ^= (np.arange(2 ** n_qubits) >> (n_qubits - 1 - qb)) & 1  # bit of qubit qb
            meas += np.sum(px * (1 - 2 * parity))  # <prod X>
        meas /= math.comb(n_qubits, c)  # average
        expv = comb(c, c // 2) * comb(n_qubits - c, k - c // 2) / comb(n_qubits, k) if c % 2 == 0 else 0.0  # ideal value
        lx += (expv - meas) ** 2 / expv ** 2 if expv != 0 else (expv - meas) ** 2  # official weighting
    return float(1 / (lz + lx))  # expected official score


def emulate(circuit, qubits, k, cal, *, detuning_sigma_hz=0.0, n_samples=1, spectator_sigma=0.0, seed=1):
    """Expected official score of a returned circuit under the emulator."""
    rng = np.random.default_rng(seed)  # reproducible noise realisations
    qubits = sorted(qubits)  # measurement order
    n_q = len(qubits)  # number of data qubits
    spec = {}  # spectator phases
    if spectator_sigma:  # draw one fixed device realisation
        for pair in sorted(cal[0], key=str):  # every coupler
            a, b = tuple(pair)  # endpoints
            spec[(a, b)] = rng.normal(0, spectator_sigma)  # phase on a when b is active
            spec[(b, a)] = rng.normal(0, spectator_sigma)  # phase on b when a is active
    H = cirq.unitary(cirq.Circuit(cirq.H(q) for q in qubits))  # X-basis measurement rotation
    pz = np.zeros(2 ** n_q)  # Z-basis distribution
    px = np.zeros(2 ** n_q)  # X-basis distribution
    for _ in range(n_samples):  # average over quasi-static realisations
        det = {q: rng.normal(0, detuning_sigma_hz) for q in qubits} if detuning_sigma_hz else {}  # detunings
        rho = noisy_density_matrix(circuit, HardwareNoise(cal, det, spec), qubits)  # noisy state
        pz += np.real(np.diag(rho)) / n_samples  # Z-basis probabilities
        px += np.real(np.diag(H @ rho @ H.conj().T)) / n_samples  # X-basis probabilities
    e0 = np.array([cal[2][q][0] for q in qubits])  # readout 0 -> 1
    e1 = np.array([cal[3][q][0] for q in qubits])  # readout 1 -> 0
    return expected_score(pz, px, n_q, k, e0, e1)  # shot-noise-free official score


# %% [markdown]
# ### Design experiments
#
# The full runs (`tools/experiments.py` in the repository; two or three calibration seeds each) gave the following
# expected official scores (ideal values: $M/(M-1)$, e.g. 1.05 for $(6,3)$).
#
# **A. Construction family** (same calibration, best placement of each family):
#
# | $(N,n)$ | BE path: CZs / moments / score | Two-arm split: CZs / moments / score |
# | --- | --- | --- |
# | (6,2) | 18 / 31 / 0.878 | **15 / 17 / 0.943** |
# | (6,3) | 36 / 63 / 0.733 | **23 / 25 / 0.859** |
# | (8,2) | 26 / 43 / 0.794 | **23 / 23 / 0.876** |
# | (8,3) | 54 / 91 / 0.575 | **41 / 41 / 0.753** |
# | (8,4) | 77 / 117 / 0.502 | **49 / 49 / 0.699** |
#
# The split construction wins everywhere: fewer CZs *and* about half the depth.
#
# **B. Dynamical decoupling** (expected score without → with DD):
#
# | noise model | (6,1) | (6,3) | (8,2) |
# | --- | --- | --- | --- |
# | incoherent only | 1.117 → 1.118 | 0.868 → 0.859 | 0.885 → 0.876 |
# | + detuning, σ = 50 kHz | 1.113 → 1.114 | 0.847 → 0.848 | 0.880 → 0.873 |
# | + detuning, σ = 200 kHz | 1.036 → 1.041 | **0.288 → 0.579** | **0.644 → 0.812** |
# | + spectator phases, σ = 0.1 rad | 1.116 → 1.118 | **0.778 → 0.827** | **0.668 → 0.691** |
#
# DD is an insurance policy: when idle errors are purely incoherent it costs about 1% (the pulses themselves have errors),
# but when coherent idle errors are present, which is what the organisers warn about, it recovers large losses (×2 at
# 200 kHz detuning). We therefore keep DD on.
#
# **C. Placement strategy**, scored on a "true" device whose rates are the reported means plus Gaussian noise of the reported
# uncertainty (so the calibration is itself an estimate). Mean over 3 seeds × {(6,1), (6,3), (8,2)}:
#
# | strategy | mean expected score |
# | --- | --- |
# | calibration-aware, $\beta=1$ (submitted) | **0.944** |
# | calibration-aware, $\beta=0$ | 0.944 (identical choices: the fake data have uniform uncertainties) |
# | random choice among our candidates | 0.918 |
# | topology-only (ignores calibration) | 0.885 |
#
# Calibration-aware placement is clearly better than ignoring calibration, consistent with Huo et al.'s observation that
# randomised alternatives are competitive but not superior. The $\beta$ penalty only matters when uncertainties differ between
# qubits, as they will on real hardware.
#
# The cell below re-runs a light version of each experiment (about two minutes).

# %%
def _select(N, n, cal_obj, dd=True, kinds=None, rng=None):
    """Best (or random) candidate of the requested families under a given Calibration object."""
    cands = [(estimate_log_fidelity(c, cal_obj, qs, n / N), c, qs, kind)
             for c, qs, kind in generate_candidates(N, n, cal_obj, dd=dd) if not kinds or kind in kinds]  # scored
    return cands[rng.integers(len(cands))] if rng is not None else max(cands, key=lambda t: t[0])  # pick


_cal = _seeded_calibration(0)  # one reproducible device
_cobj = Calibration(*_cal)  # its pessimistic model
for _kind in ('path', 'split'):  # A. construction family on (6, 3)
    _, _c, _qs, _ = _select(6, 3, _cobj, kinds=[_kind])  # best candidate of this family
    print(f'(6,3) {_kind:5s}: CZs={sum(len(op.qubits) == 2 for op in _c.all_operations())}, '
          f'moments={len(_c)}, expected score={emulate(_c, _qs, 3, _cal):.3f}')
for _dd in (False, True):  # B. DD under strong quasi-static detuning on (6, 3)
    _, _c, _qs, _ = _select(6, 3, _cobj, dd=_dd)  # best candidate with / without DD
    print(f'(6,3) DD={_dd!s:5s}: incoherent={emulate(_c, _qs, 3, _cal):.3f}, '
          f'200 kHz detuning={emulate(_c, _qs, 3, _cal, detuning_sigma_hz=200e3, n_samples=6):.3f}')
_uniform = tuple({key: (1e-3, 0.0) for key in d} for d in _cal)  # C. topology-only calibration
for _name, _co in (('calibration-aware', _cobj), ('topology-only', Calibration(*_uniform, beta=0.0))):  # (6, 1)
    _, _c, _qs, _ = _select(6, 1, _co)  # chosen W-state circuit
    print(f'(6,1) {_name}: expected score={emulate(_c, _qs, 1, _cal):.3f}')
