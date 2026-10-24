"""Noise model built from the calibration dicts, plus optional idle decoherence.

Gate noise (from the calibration data the judges hand us):
  * depolarize(p_pair, 2) after every CZ, where p_pair is the coupler's Pauli error rate;
  * depolarize(p_q) after every PhasedXZ with a nonzero X part. Pure-Z PhasedXZ gates are
    virtual (frame changes) on Google hardware, so they get no error and take no time.

Idle noise (not in the calibration data; parameters are assumptions, flagged as such):
  * amplitude damping with T1 (Willow spec sheet: 68 +- 13 us mean, QEC chip);
  * pure dephasing with T_phi (no published number; we assume 100 us);
  * quasi-static detuning: a random frequency offset per qubit per shot, which is the
    low-frequency noise that dynamical decoupling refocuses. It is sampled by
    `detuned_copies`, and the caller averages output distributions over the copies.
  Every moment lasts as long as its slowest gate. Assumed durations: 25 ns (1q), 42 ns (CZ).

Readout errors are applied classically to the exact distribution (`apply_readout`).
"""

import cirq
import numpy as np

T_1Q_NS = 25.0  # assumption: typical Google single-qubit microwave pulse
T_CZ_NS = 42.0  # assumption: typical Google CZ duration


def _is_virtual(gate: cirq.PhasedXZGate, atol: float = 1e-9) -> bool:
    x = gate.x_exponent % 2
    return min(x, 2 - x) < atol


def moment_duration_ns(moment: cirq.Moment) -> float:
    dur = 0.0
    for op in moment:
        if op.gate == cirq.CZ:
            dur = max(dur, T_CZ_NS)
        elif isinstance(op.gate, cirq.PhasedXZGate) and not _is_virtual(op.gate):
            dur = max(dur, T_1Q_NS)
    return dur


class CalibratedNoise(cirq.NoiseModel):
    def __init__(
        self,
        cz_pauli_error_rate: dict,
        sq_pauli_error_rate: dict,
        t1_us: float | None = None,
        tphi_us: float | None = None,
    ):
        self._cz = {frozenset(pair): v[0] for pair, v in cz_pauli_error_rate.items()}
        self._sq = {q: v[0] for q, v in sq_pauli_error_rate.items()}
        self._t1_ns = None if t1_us is None else 1e3 * t1_us
        self._tphi_ns = None if tphi_us is None else 1e3 * tphi_us

    def noisy_moment(self, moment: cirq.Moment, system_qubits):
        gate_noise = []
        for op in moment:
            if op.gate == cirq.CZ:
                gate_noise.append(cirq.depolarize(self._cz[frozenset(op.qubits)], 2).on(*op.qubits))
            elif isinstance(op.gate, cirq.PhasedXZGate) and not _is_virtual(op.gate):
                gate_noise.append(cirq.depolarize(self._sq[op.qubits[0]]).on(*op.qubits))
        out = [moment]
        if gate_noise:
            out.append(cirq.Moment(gate_noise))
        dur = moment_duration_ns(moment)
        if dur > 0 and self._t1_ns is not None:
            g = 1 - np.exp(-dur / self._t1_ns)
            out.append(cirq.Moment(cirq.amplitude_damp(g).on(q) for q in system_qubits))
        if dur > 0 and self._tphi_ns is not None:
            g = 1 - np.exp(-2 * dur / self._tphi_ns)
            out.append(cirq.Moment(cirq.phase_damp(g).on(q) for q in system_qubits))
        return out


def detuned_copies(
    circuit: cirq.Circuit, t2star_us: float, num_copies: int, seed: int = 0
) -> list[cirq.Circuit]:
    """Copies of `circuit` with a random static Z detuning per qubit (one draw per copy).

    The detuning is Gaussian with sigma = sqrt(2) / T2*, so an idling superposition loses
    coherence as exp(-(t / T2*)^2). The phase is applied after every moment,
    proportional to that moment's duration.
    """
    rng = np.random.default_rng(seed)
    qubits = sorted(circuit.all_qubits())
    sigma_per_ns = np.sqrt(2) / (1e3 * t2star_us)  # rad / ns
    copies = []
    for _ in range(num_copies):
        omega = dict(zip(qubits, rng.normal(0.0, sigma_per_ns, len(qubits))))
        moments = []
        for moment in circuit:
            moments.append(moment)
            dur = moment_duration_ns(moment)
            if dur > 0:
                moments.append(cirq.Moment(cirq.rz(omega[q] * dur).on(q) for q in qubits))
        copies.append(cirq.Circuit(moments))
    return copies


def apply_readout(probs: np.ndarray, e0: list[float], e1: list[float]) -> np.ndarray:
    """Apply independent asymmetric bit flips to a distribution over 2^N strings.

    probs is in cirq's big-endian order (qubit 0 = most significant bit), and e0[j] / e1[j]
    are the 0->1 / 1->0 error rates of qubit j.
    """
    N = len(e0)
    p = probs.reshape((2,) * N)
    for j in range(N):
        confusion = np.array([[1 - e0[j], e1[j]], [e0[j], 1 - e1[j]]])  # [read, true]
        p = np.moveaxis(np.tensordot(confusion, p, axes=([1], [j])), 0, j)
    return p.reshape(-1)
