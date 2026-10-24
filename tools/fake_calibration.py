"""Copies of the notebook's fake-calibration generators (so tools can run outside Jupyter)."""
import cirq
import cirq_google
import numpy as np


def pick_single_qubit_dropouts(dropout_rate, rng):
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(median, spread, dropouts, uncertainty, rng):
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits)))))


def generate_two_qubit_pauli_error_rates(median=2.5e-3, spread=0.5, dropout_rate=0.01, uncertainty=5e-4, rng=None, single_qubit_dropouts=None):
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()
    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array([np.all([q not in single_qubit_dropouts for q in pair]) for pair in all_pairs])
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs)))))


def make_calibration(seed=0):
    rng = np.random.default_rng(seed)
    drop = pick_single_qubit_dropouts(0.01, rng)
    sq = generate_qubit_error_rates(4e-4, 0.5, drop, 2e-4, rng)
    cz = generate_two_qubit_pauli_error_rates(single_qubit_dropouts=drop, rng=rng)
    e0 = generate_qubit_error_rates(5e-3, 0.5, drop, 2e-4, rng)
    e1 = generate_qubit_error_rates(1e-2, 0.5, drop, 3e-4, rng)
    return cz, sq, e0, e1
