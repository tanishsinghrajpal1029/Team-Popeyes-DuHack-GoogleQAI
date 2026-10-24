"""The challenge's hard constraints, checked the way the judges would.

`check_rules` is stricter than `cirq_google.Willow105.validate_circuit`, which also
accepts X/Y/H/Z-pow gates and measurements that the challenge does not allow.
"""

import itertools
import math

import cirq
import cirq_google
import numpy as np

MAX_MOMENTS = 1750


# ---- verbatim from the notebook (cell 4), with the keep_indices bug fixed --------------


def create_dicke_statevector(num_qubits: int, num_excitations: int) -> np.ndarray:
    psi = np.zeros(2**num_qubits)
    normalization = 1 / np.sqrt(math.comb(num_qubits, num_excitations))
    qubits = range(num_qubits)
    for excited_qubits in itertools.combinations(qubits, num_excitations):
        idx = sum(2**qubit_number for qubit_number in excited_qubits)
        psi[idx] = normalization
    return psi


def test_circuit(circuit: cirq.Circuit, num_excitations: int, qubits: set[cirq.GridQubit], sampler = cirq.Simulator()):
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(len(qubits))  # fixed: the notebook has range(qubits)
    num_qubits = len(qubits)
    psi_circuit_full = sampler.simulate(circuit, qubit_order=ordered_qubits).final_state_vector
    psi_circuit = cirq.sub_state_vector(
        state_vector=psi_circuit_full,
        keep_indices=keep_indices
    ) if len(all_qubits) > len(qubits) else psi_circuit_full
    assert np.isclose(np.abs(psi_circuit @ create_dicke_statevector(num_qubits, num_excitations)) ** 2, 1.0)


# ---- our checks -------------------------------------------------------------------------


def noiseless_fidelity(circuit: cirq.Circuit, qubits: set, num_excitations: int) -> float:
    """<D|rho|D> on the returned qubits, with ancillas traced out (complex128)."""
    data = sorted(qubits)
    anc = sorted(circuit.all_qubits() - set(qubits))
    sim = cirq.Simulator(dtype=np.complex128)
    psi = sim.simulate(circuit, qubit_order=data + anc).final_state_vector
    m = psi.reshape(2 ** len(data), 2 ** len(anc))
    d = create_dicke_statevector(len(data), num_excitations)
    v = d @ m
    return float(np.real(np.vdot(v, v)))


def check_rules(
    circuit: cirq.Circuit,
    qubits: set,
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict,
    sq_pauli_error_rate: dict,
    fidelity_atol: float = 1e-9,
) -> list[str]:
    """Return a list of violated rules (empty if the circuit is valid)."""
    problems = []
    pairs = {frozenset(p) for p in cz_pauli_error_rate}
    for op in circuit.all_operations():
        g = op.gate
        if isinstance(op, cirq.ClassicallyControlledOperation) or g is None:
            problems.append(f"disallowed operation {op!r}")
        elif isinstance(g, cirq.CZPowGate):
            if g != cirq.CZ:
                problems.append(f"partial CZ {op!r}")
            elif frozenset(op.qubits) not in pairs:
                problems.append(f"CZ on uncalibrated pair {op.qubits}")
        elif isinstance(g, (cirq.PhasedXZGate, cirq.ResetChannel)):
            if op.qubits[0] not in sq_pauli_error_rate:
                problems.append(f"gate on uncalibrated qubit {op!r}")
        else:
            problems.append(f"disallowed gate {op!r}")
    if len(circuit) > MAX_MOMENTS:
        problems.append(f"{len(circuit)} moments > {MAX_MOMENTS}")
    if len(qubits) != num_qubits:
        problems.append(f"returned {len(qubits)} qubits, expected {num_qubits}")
    if not set(qubits) <= set(sq_pauli_error_rate):
        problems.append("returned qubits outside the calibrated set")
    try:
        cirq_google.Willow105.validate_circuit(circuit)
    except ValueError as e:
        problems.append(f"validate_circuit: {e}")
    if not problems:
        fid = noiseless_fidelity(circuit, qubits, num_excitations)
        if abs(fid - 1) > fidelity_atol:
            problems.append(f"noiseless fidelity {fid:.12f}")
    return problems
