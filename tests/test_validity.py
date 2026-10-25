"""Every (N, n) with N <= 10 must yield a valid, exact circuit on several calibrations."""

import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]

from calib import BENCH_SEEDS, make_calibration  # noqa: E402
from main import create_dicke_state  # noqa: E402
from validity import check_rules  # noqa: E402
from validity import test_circuit as notebook_test_circuit  # noqa: E402

MAX_N = 10
CASES = [(N, n) for N in range(1, MAX_N + 1) for n in range(N + 1)]
CALS = {seed: make_calibration(seed) for seed in BENCH_SEEDS}


@pytest.mark.parametrize("seed", BENCH_SEEDS)
@pytest.mark.parametrize("N,n", CASES)
def test_valid_and_exact(seed, N, n):
    cal = CALS[seed]
    t0 = time.time()
    circuit, qubits = create_dicke_state(N, n, **cal)
    build_time = time.time() - t0
    assert build_time < 60, f"build took {build_time:.1f}s"
    problems = check_rules(
        circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"]
    )
    assert not problems, problems


@pytest.mark.parametrize("N,n", [(2, 1), (4, 2), (5, 3), (6, 1)])
def test_notebook_checker_accepts(N, n):
    circuit, qubits = create_dicke_state(N, n, **CALS[BENCH_SEEDS[0]])
    notebook_test_circuit(circuit, n, qubits)


def test_heavy_dropout_still_works():
    cal = make_calibration(7, qubit_dropout_rate=0.1)
    for N, n in [(3, 1), (6, 2), (8, 4)]:
        circuit, qubits = create_dicke_state(N, n, **cal)
        assert not check_rules(
            circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"]
        )


def _bonus_baseline() -> dict:
    """Moments per bonus case from the last notebook benchmark (experiments/bench_notebook.py)."""
    import csv

    path = os.path.join(ROOT, "experiments", "results", "notebook_bonus.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {(int(r["N"]), int(r["n"])): int(r["moments"]) for r in csv.DictReader(fh)}


BONUS_BASELINE = _bonus_baseline()


@pytest.mark.parametrize("N,n", [(N, n) for N in range(2, 16) for n in range(1, N // 2 + 1)])
def test_bonus_entry_exact_and_routed(N, n):
    """All 56 scored bonus cases: exact, routed, only allowed gates, built in time, no deeper
    than the committed benchmark (the bonus scores N*n / moments)."""
    from main import _uniform_calibration, create_dicke_state_adaptive
    from validity import noiseless_fidelity
    import cirq_google

    t0 = time.time()
    circuit, qubits = create_dicke_state_adaptive(N, n)
    assert time.time() - t0 < 60
    cirq_google.Willow105.validate_circuit(circuit)
    assert len(qubits) == N
    cal = _uniform_calibration()
    problems = check_rules(
        circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"], fidelity_atol=None
    )
    assert not problems, problems
    assert abs(noiseless_fidelity(circuit, qubits, n) - 1) < 1e-9
    if (N, n) in BONUS_BASELINE:
        assert len(circuit) <= BONUS_BASELINE[(N, n)], f"{len(circuit)} moments, benchmark had {BONUS_BASELINE[(N, n)]}"
