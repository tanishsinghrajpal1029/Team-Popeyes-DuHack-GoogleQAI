"""create_dicke_state: builds several exact candidate circuits and keeps the one with
the highest predicted success on the chip.

Candidates
  n = 1 or n = N - 1   splitting trees from several roots (w_tree.py)
  2 <= min(n, N - n)   two row split on several layouts (two_rows.py)
  small N              single chain baseline as an extra candidate (baseline.py)
  stored templates     shortest known exact circuits for small cases (templates.py)

Every candidate is compiled to CZ + PhasedXZ, scheduled as late as possible (qubits
stay in |0> where they do not decay), given decoupling pulses in idle slots, and
scored with a quick success estimate from the calibration data.
"""

import time

import cirq

import baseline
from w_tree import build_w_tree
from two_rows import build_two_rows
from decoupling import add_decoupling
from scheduling import alap, predict_success

try:
    import templates
except ImportError:
    templates = None

USE_DECOUPLING = True
_FLIP = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)


def _finish(raw_circuit, qubits, flip):
    c = raw_circuit
    if flip:
        c = c + cirq.Circuit(cirq.Moment(_FLIP.on(q) for q in qubits))
    c = baseline.compile_to_native(c)
    c = alap(c)
    if USE_DECOUPLING:
        c = add_decoupling(c)
    return c


def create_dicke_state(
    num_qubits,
    num_excitations,
    cz_pauli_error_rate,
    sq_pauli_error_rate,
    zero_errors,
    one_errors,
):
    N, n = num_qubits, num_excitations
    if N < 1 or n < 0 or n > N:
        raise NotImplementedError("Invalid (N, n).")
    errs = (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    start = time.time()

    k = min(n, N - n)
    flip = n != k
    if k == 0:
        return baseline.create_dicke_state(N, n, *errs)

    raw = []   # (circuit before compiling, qubit list)
    if templates is not None:
        try:
            raw += templates.candidates(N, k, *errs)
        except Exception:
            pass
    if k == 1:
        try:
            for c in build_w_tree(N, *errs, keep=4 if N <= 30 else 1):
                raw.append(c)
        except NotImplementedError:
            pass
    else:
        res = build_two_rows(N, k, *errs, keep=4 if N <= 16 else 1)
        if res:
            raw += res
    if N <= 8 or not raw:
        try:
            chain = baseline.find_path(N, *errs, frac_one=n / N,
                                       time_budget=10.0 if N <= 60 else 30.0)
            ops = [cirq.X(q) for q in chain[N - k:]]
            if N > 1:
                ops += baseline.dicke_unitary_ops(chain, k)
            raw.append((cirq.Circuit(ops), chain))
        except NotImplementedError:
            pass
    if not raw:
        raise NotImplementedError("No layout found.")

    best = None
    for c, qs in raw:
        fc = _finish(c, qs, flip)
        if len(fc) > 1750:
            continue
        score = predict_success(fc, qs, n, *errs)
        if best is None or score > best[0]:
            best = (score, fc, qs)
        if time.time() - start > 45:
            break
    if best is None:
        raise NotImplementedError("Circuit too deep.")
    return best[1], set(best[2])
