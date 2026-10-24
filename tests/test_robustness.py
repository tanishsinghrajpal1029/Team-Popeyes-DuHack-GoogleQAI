"""Calibration data shaped differently from the notebook's generators must still work."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]

from calib import make_calibration  # noqa: E402
from main import create_dicke_state  # noqa: E402
from validity import check_rules  # noqa: E402

CASES = [(2, 1), (3, 2), (4, 2), (6, 3), (9, 2)]


def _valid(cal, N, n):
    circuit, qubits = create_dicke_state(N, n, **cal)
    return check_rules(circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"])


def _variant(name):
    cal = make_calibration(11)
    cz, sq, e0, e1 = (cal[k] for k in ("cz_pauli_error_rate", "sq_pauli_error_rate", "zero_errors", "one_errors"))
    if name == "reversed_pairs":
        cz = {(b, a): v for (a, b), v in cz.items()}
    elif name == "both_orientations":
        cz = {**cz, **{(b, a): (v[0] * 1.1, v[1]) for (a, b), v in list(cz.items())[::3]}}
    elif name == "lists_and_numpy":
        cz = {k: [np.float64(v[0]), np.float64(v[1])] for k, v in cz.items()}
        sq = {k: np.array(v) for k, v in sq.items()}
    elif name == "missing_readout":
        drop = sorted(e0)[::7]
        e0 = {k: v for k, v in e0.items() if k not in drop}
        e1 = {k: v for k, v in e1.items() if k not in drop[::2]}
    elif name == "coupler_to_uncalibrated_qubit":
        gone = sorted(sq)[::9]
        sq = {k: v for k, v in sq.items() if k not in gone}  # couplers still mention them
    return dict(cz_pauli_error_rate=cz, sq_pauli_error_rate=sq, zero_errors=e0, one_errors=e1)


@pytest.mark.parametrize("name", ["reversed_pairs", "both_orientations", "lists_and_numpy",
                                  "missing_readout", "coupler_to_uncalibrated_qubit"])
@pytest.mark.parametrize("N,n", CASES)
def test_odd_calibration_shapes(name, N, n):
    assert not _valid(_variant(name), N, n)


def test_numpy_integer_arguments():
    cal = make_calibration(0)
    circuit, qubits = create_dicke_state(np.int64(5), np.int32(2), **cal)
    assert not check_rules(circuit, qubits, 5, 2, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"])


@pytest.mark.parametrize("N,n", [(4, 5), (4, -1), (0, 0), (106, 1)])
def test_impossible_requests_raise_not_implemented(N, n):
    with pytest.raises(NotImplementedError):
        create_dicke_state(N, n, **make_calibration(0))


def test_very_sparse_device_does_not_crash():
    cal = make_calibration(3, qubit_dropout_rate=0.35)
    for N, n in [(3, 1), (5, 2), (8, 4)]:
        try:
            problems = _valid(cal, N, n)
        except NotImplementedError:
            continue
        assert not problems
