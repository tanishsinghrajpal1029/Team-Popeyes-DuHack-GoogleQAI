"""Fake Willow calibration data.

The three generators are copied verbatim from the challenge notebook (cell 11).
`make_calibration` reproduces the notebook's cell 12 with an explicit, seeded rng so
every experiment is reproducible.
"""

import cirq
import cirq_google
import numpy as np


# ---- verbatim from the notebook -------------------------------------------------------


def pick_single_qubit_dropouts(
    dropout_rate: float,
    rng: np.random.Generator = np.random.default_rng()
) -> set[cirq.GridQubit]:
    """Pick some qubits to be broken.

    Args:
        dropout_rate: The probability that a qubit is broken.
        rng: The pseudo-random number generator

    Returns:
        The broken qubits.
    """
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate fake single-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropouts: Qubits to exclude
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """

    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(
        zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits))))
    )


def generate_two_qubit_pauli_error_rates(
    median: float = 2.5e-3,
    spread: float = 0.5,
    dropout_rate: float = 0.01,
    uncertainty: float = 5e-4,
    rng: np.random.Generator = np.random.default_rng(),
    single_qubit_dropouts: set[cirq.GridQubit] = None,
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate fake two-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropout_rate: The probability of a broken coupler.
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator
        single_qubit_dropouts: The single-qubit dropouts to skip.

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()

    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array(
        [np.all([q not in single_qubit_dropouts for q in pair]) for pair in all_pairs]
    )
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(
        zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs))))
    )


# ---- seeded wrapper (same calls and parameters as notebook cell 12) -------------------

# Calibration seeds every experiment reports on. Fixed so slide numbers are reproducible.
BENCH_SEEDS = (0, 1, 2)


def make_calibration(seed: int, qubit_dropout_rate: float = 0.01) -> dict:
    """Notebook cell 12 with a seeded rng.

    Returns the four keyword arguments of `create_dicke_state`.
    """
    rng = np.random.default_rng(seed)
    single_qubit_dropouts = pick_single_qubit_dropouts(qubit_dropout_rate, rng=rng)
    sq_pauli_error_rate = generate_qubit_error_rates(
        4e-4, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    cz_pauli_error_rate = generate_two_qubit_pauli_error_rates(
        single_qubit_dropouts=single_qubit_dropouts, rng=rng
    )
    zero_errors = generate_qubit_error_rates(5e-3, 0.5, single_qubit_dropouts, 2e-4, rng=rng)
    one_errors = generate_qubit_error_rates(1e-2, 0.5, single_qubit_dropouts, 3e-4, rng=rng)
    return dict(
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )
