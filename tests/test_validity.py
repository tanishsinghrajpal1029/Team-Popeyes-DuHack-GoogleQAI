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
