"""Noisy-hardware emulator and official-score evaluation used to compare design choices.

The emulator is deliberately simple and *assumption-explicit*: every parameter that the
challenge's calibration data does not provide (T1, white dephasing, quasi-static detuning)
is an argument so that conclusions can be checked for sensitivity.
"""
import itertools
import math

import cirq
import numpy as np

T_1Q = 25e-9
T_CZ = 42e-9


# --- official scoring functions (verbatim logic from the challenge notebook) -------------------
def check_z_bitstring_distribution(bitstrings, num_excitations):
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == num_excitations)
    chi2 = 0
    ideal_prob = 1 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(range(n_qubits), num_excitations):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count
    return ideal_prob * chi2 / fraction_in_subspace


def get_correlator_average(bitstrings, correlator_size):
    correlator_sum = 0.0
    n_shots, n_qubits = bitstrings.shape
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1 - 2 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(bitstrings, num_excitations, max_correlator_size):
    comb = lambda n, k: math.comb(n, k) if (k >= 0 and n >= k) else 0
    n_shots, n_qubits = bitstrings.shape
    score = 0.0
    for c in range(1, max_correlator_size + 1):
        expected = (comb(c, c // 2) * comb(n_qubits - c, num_excitations - c // 2) / comb(n_qubits, num_excitations)
                    if c % 2 == 0 else 0.0)
        measured = get_correlator_average(bitstrings, c)
        score += (expected - measured) ** 2 / expected**2 if expected != 0 else (expected - measured) ** 2
    return score


def overall_score(z_bits, x_bits, num_excitations, max_correlator_size):
    return 1 / (check_z_bitstring_distribution(z_bits, num_excitations)
                + check_x_bitstring_distribution(x_bits, num_excitations, max_correlator_size))


# --- noise model ---------------------------------------------------------------------------------
class HardwareNoise:
    """Per-moment noise: gate depolarisation from calibration, T1/Tphi idling, quasi-static detuning."""

    def __init__(self, cz_err, sq_err, t1=68e-6, tphi=100e-6, detuning_hz=None, spectator=None):
        self.cz_err = {frozenset(k): v for k, v in cz_err.items()}
        self.sq_err = dict(sq_err)
        self.t1, self.tphi = t1, tphi
        self.detuning_hz = detuning_hz or {}
        self.spectator = spectator or {}  # (idle qubit, active neighbour) -> coherent Z phase per CZ moment
        self.nbrs = {}
        for pair in self.cz_err:
            a, b = tuple(pair)
            self.nbrs.setdefault(a, set()).add(b)
            self.nbrs.setdefault(b, set()).add(a)

    def moment_ops(self, moment, qubits):
        dur = T_CZ if any(len(op.qubits) == 2 for op in moment) else T_1Q
        out = []
        for op in moment:
            if len(op.qubits) == 2:
                out.append(cirq.depolarize(self.cz_err[frozenset(op.qubits)], n_qubits=2).on(*op.qubits))
            elif isinstance(op.gate, cirq.PhasedXZGate):
                out.append(cirq.depolarize(self.sq_err[op.qubits[0]]).on(*op.qubits))
        if self.spectator and dur == T_CZ:
            active = {q for op in moment for q in op.qubits}
            for q in qubits:
                if q in active:
                    continue
                phi = sum(self.spectator.get((q, w), 0.0) for w in self.nbrs.get(q, ()) if w in active)
                if phi:
                    out.append(cirq.rz(phi).on(q))
        g = 1 - math.exp(-dur / self.t1)
        lam = 1 - math.exp(-2 * dur / self.tphi)
        for q in qubits:
            out.append(cirq.amplitude_damp(g).on(q))
            out.append(cirq.phase_damp(lam).on(q))
            d = self.detuning_hz.get(q, 0.0)
            if d:
                out.append(cirq.rz(2 * math.pi * d * dur).on(q))
        return out


def final_density_matrix(circuit, noise, qubits):
    noisy = cirq.Circuit()
    for moment in circuit:
        noisy.append(cirq.Moment(moment.operations))
        extra = noise.moment_ops(moment, qubits)
        # split noise into moments that do not overlap
        while extra:
            used, this, rest = set(), [], []
            for op in extra:
                if used.isdisjoint(op.qubits):
                    this.append(op)
                    used.update(op.qubits)
                else:
                    rest.append(op)
            noisy.append(cirq.Moment(this))
            extra = rest
    sim = cirq.DensityMatrixSimulator(dtype=np.complex128)
    return sim.simulate(noisy, qubit_order=qubits).final_density_matrix


def sample_with_readout(probs, n_qubits, shots, e0, e1, rng):
    probs = np.clip(np.real(probs), 0, None)
    probs = probs / probs.sum()
    idx = rng.choice(len(probs), size=shots, p=probs)
    bits = ((idx[:, None] >> np.arange(n_qubits - 1, -1, -1)) & 1).astype(int)
    flip0 = rng.random(bits.shape) < e0[None, :]
    flip1 = rng.random(bits.shape) < e1[None, :]
    bits = np.where(bits == 0, flip0.astype(int), 1 - flip1.astype(int))
    return bits


def evaluate(circuit, qubits, num_excitations, cal, *, shots=10_000, max_corr=4, rng=None,
             t1=68e-6, tphi=100e-6, detuning_sigma_hz=0.0, n_detuning_samples=1, readout=True,
             spectator_sigma=0.0, spectator_seed=1234, true_cal=None):
    """Emulate Z- and X-basis experiments and return the official score and diagnostics."""
    cz, sq, e0d, e1d = true_cal if true_cal is not None else cal
    rng = rng or np.random.default_rng(0)
    spec = {}
    if spectator_sigma:
        srng = np.random.default_rng(spectator_seed)
        for pair in sorted(cz, key=str):
            a, b = tuple(pair)
            spec[(a, b)] = srng.normal(0, spectator_sigma)
            spec[(b, a)] = srng.normal(0, spectator_sigma)
    qubits = sorted(qubits)
    allq = sorted(circuit.all_qubits() | set(qubits))
    n = len(qubits)
    pz = np.zeros(2**n)
    px = np.zeros(2**n)
    hcirc = cirq.Circuit(cirq.H(q) for q in qubits)
    for s in range(n_detuning_samples):
        det = {q: rng.normal(0, detuning_sigma_hz) for q in allq} if detuning_sigma_hz else {}
        noise = HardwareNoise({k: v[0] for k, v in cz.items()}, {k: v[0] for k, v in sq.items()}, t1, tphi, det, spec)
        rho = final_density_matrix(circuit, noise, allq)
        keep = [allq.index(q) for q in qubits]
        if len(allq) > n:
            rho = cirq.partial_trace(rho.reshape([2] * (2 * len(allq))), keep).reshape(2**n, 2**n)
        pz += np.real(np.diag(rho)) / n_detuning_samples
        H = cirq.unitary(hcirc)
        px += np.real(np.diag(H @ rho @ H.conj().T)) / n_detuning_samples
    e0 = np.array([e0d[q][0] for q in qubits]) if readout else np.zeros(n)
    e1 = np.array([e1d[q][0] for q in qubits]) if readout else np.zeros(n)
    zb = sample_with_readout(pz, n, shots, e0, e1, rng)
    xb = sample_with_readout(px, n, shots, e0, e1, rng)
    lz = check_z_bitstring_distribution(zb, num_excitations)
    lx = check_x_bitstring_distribution(xb, num_excitations, max_corr)
    weights = np.array([bin(i).count('1') for i in range(2**n)])
    exp = expected_losses(pz, px, n, num_excitations, e0, e1, shots, max_corr)
    return {
        'score': 1 / (lz + lx), 'Lz': lz, 'Lx': lx,
        'pop_in_subspace': float(pz[weights == num_excitations].sum()),
        'fraction_in_subspace': float(np.mean(zb.sum(1) == num_excitations)),
        **exp,
    }


def _readout_channel(probs, n, e0, e1):
    """Exact distribution after independent asymmetric bit flips (qubit 0 = most significant bit)."""
    p = np.asarray(probs, dtype=float).reshape([2] * n)
    for q in range(n):
        m = np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]])  # m[out, in]
        p = np.moveaxis(np.tensordot(m, p, axes=([1], [q])), 0, q)
    return p.reshape(-1)


def expected_losses(pz, px, n, k, e0, e1, shots, max_corr):
    """Shot-noise-free expectation of the official losses (low-variance metric for comparisons).

    With S shots, m = S f of them of weight k, and conditional distribution q over the M weight-k
    strings, E[L_Z | m] = S (sum q^2 - 1/M) + (1 - sum q^2) / f.  The X loss is evaluated at its
    exact (infinite-shot) correlator values.
    """
    pz = _readout_channel(pz, n, e0, e1)
    px = _readout_channel(px, n, e0, e1)
    weights = np.array([bin(i).count('1') for i in range(2**n)])
    sub = pz[weights == k]
    f = sub.sum()
    q = sub / f
    M = math.comb(n, k)
    lz = shots * (np.sum(q**2) - 1 / M) + (1 - np.sum(q**2)) / f
    lx = 0.0
    comb = lambda a, b: math.comb(a, b) if (b >= 0 and a >= b) else 0
    for c in range(1, max_corr + 1):
        meas = 0.0
        for idx in itertools.combinations(range(n), c):
            parity = np.zeros(2**n, dtype=int)
            for qb in idx:
                parity ^= (np.arange(2**n) >> (n - 1 - qb)) & 1
            meas += np.sum(px * (1 - 2 * parity))
        meas /= math.comb(n, c)
        expv = comb(c, c // 2) * comb(n - c, k - c // 2) / comb(n, k) if c % 2 == 0 else 0.0
        lx += (expv - meas) ** 2 / expv**2 if expv != 0 else (expv - meas) ** 2
    return {'exp_Lz': float(lz), 'exp_Lx': float(lx), 'exp_score': float(1 / (lz + lx)), 'f_readout': float(f)}
