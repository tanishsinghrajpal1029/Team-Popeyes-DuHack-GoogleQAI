# pyright: reportUndefinedVariable=false
# Notebook fragment: executed inside the challenge notebook after the solution and the provided helper cells.
# %% [markdown]
# # DuQuantum verification
#
# Exactness is established in two layers, as the plan requires:
#
# 1. **Building blocks** are compared with their defining isometries on every input of their domain, *including the relative
#    phases* (all outputs must share one global phase). This checks derivations (a)–(f) of Section 1.
# 2. **Complete circuits** returned by `create_dicke_state` are simulated with the notebook's own `test_circuit` for every
#    $1\le N\le 10$ and every $0\le n\le N$, on two independent fake calibrations (the notebook's random one and a seeded one).
#    Each circuit is also checked against the hardware rules: only `CZ` on calibrated couplers, only `PhasedXZGate` on
#    calibrated qubits, homogeneous moments, at most 1750 moments, no ancillas, exactly $N$ returned qubits.
#
# The construction is analytic, so a numerical check at small $N$ plus the algebraic argument covers all $N$: no step depends on
# $N$ except through rotation angles computed in closed form.

# %%
def _instr_unitary(ops, qubits):
    """Exact unitary of an instruction list (tensor order = `qubits`)."""
    return compile_layers(ops, dd=True).unitary(qubit_order=qubits)  # compile WITH DD to prove DD is exact too


def _check_isometry(ops, qubits, mapping):
    """Assert that ops maps each input bitstring to its target state, all with one common global phase."""
    U = _instr_unitary(ops, qubits)  # full unitary
    ref = None  # common phase
    for bits, out in mapping:  # each domain element
        target = np.zeros(2 ** len(qubits), dtype=complex)  # target state vector
        for b, amp in out.items():  # fill amplitudes
            target[int(b, 2)] = amp  # big-endian bit order matches qubit_order
        overlap = np.vdot(target, U[:, int(bits, 2)])  # <target|U|input>
        assert abs(abs(overlap) - 1) < 1e-9, (bits, overlap)  # correct state ...
        ref = overlap if ref is None else ref  # remember the first phase
        assert abs(overlap - ref) < 1e-9, (bits, overlap, ref)  # ... with a common phase (relative phases exact)


_q = cirq.LineQubit.range(3)  # three test qubits
for _p in (0.1, 0.5, 0.9):  # several split probabilities
    _c, _s = math.sqrt(1 - _p), math.sqrt(_p)  # stay / move amplitudes
    _check_isometry(_transfer(_q[0], _q[1], _p), _q[:2], [('00', {'00': 1}), ('10', {'10': _c, '01': _s})])  # (c)
    _th = _split_angle(_p)  # rotation angle for probability p
    _check_isometry(_givens(_q[0], _q[1], _th), _q[:2], [('00', {'00': 1}), ('11', {'11': 1}),
                    ('01', {'01': _c, '10': _s}), ('10', {'10': _c, '01': -_s})])  # (d): full Givens rotation
    _check_isometry(_g3(*_q, _th), _q, [('000', {'000': 1}), ('011', {'011': 1}), ('111', {'111': 1}),
                    ('001', {'001': _c, '010': _s}), ('101', {'110': 1})])  # (e)
    _pn = min(1.0, _p + 0.05)  # a second, different probability for the top block
    _check_isometry(_top_block(*_q, _p, _pn), _q, [('000', {'000': 1}),
                    ('001', {'001': math.sqrt(1 - _pn), '010': math.sqrt(_pn)}),
                    ('011', {'011': _c, '110': _s})])  # (f)
print('building blocks (a)-(f): exact on their whole domains, including relative phases')

_grid = [cirq.GridQubit(r, c) for r in range(2) for c in range(6)]  # 2 x 6 patch for logical tests
for _N in range(2, 7):  # split construction on an explicit ladder, all k <= N/2
    for _k in range(1, _N // 2 + 1):  # every excitation number
        _right = [cirq.GridQubit(0, i) for i in range(_N - _N // 2)]  # r_1 ... r_b along row 0
        _left = [cirq.GridQubit(1, _k - v) for v in range(1, _N // 2 + 1)]  # l_v under r_{k+1-v}, continuing left
        _circ = compile_layers(split_ops(_left, _right, _k), dd=True)  # compile with DD
        test_circuit(_circ, _k, set(_left + _right))  # notebook's own checker (fixed version)
print('split construction: exact for all N <= 6')

# %%
import time as _time  # wall-clock timing of the full solution


def check_hardware_rules(circuit, qubits, cz_rates, sq_rates, num_qubits):
    """Assert the challenge's hardware requirements for a returned (circuit, qubits) pair."""
    pairs = {frozenset(p) for p in cz_rates}  # calibrated couplers
    assert len(qubits) == num_qubits, 'wrong number of qubits'  # exactly N data qubits
    assert circuit.all_qubits() <= set(qubits), 'unexpected ancilla'  # no extra qubits
    assert len(circuit) <= 1750, 'too many moments'  # hardware limit
    for moment in circuit:  # every moment
        assert len({len(op.qubits) for op in moment}) == 1, 'mixed moment'  # 1q-only or CZ-only
        for op in moment:  # every gate
            if len(op.qubits) == 2:  # two-qubit gate
                assert op.gate == cirq.CZ and frozenset(op.qubits) in pairs, op  # CZ on a calibrated coupler
            else:  # one-qubit gate
                assert isinstance(op.gate, cirq.PhasedXZGate) and op.qubits[0] in sq_rates, op  # native gate


def _seeded_calibration(seed):
    """A reproducible fake calibration using the notebook's generators."""
    rng = np.random.default_rng(seed)  # seeded generator
    drop = pick_single_qubit_dropouts(0.01, rng)  # broken qubits
    sq = generate_qubit_error_rates(4e-4, 0.5, drop, 2e-4, rng)  # single-qubit errors
    cz = generate_two_qubit_pauli_error_rates(single_qubit_dropouts=drop, rng=rng)  # CZ errors
    e0 = generate_qubit_error_rates(5e-3, 0.5, drop, 2e-4, rng)  # 0 -> 1 readout
    e1 = generate_qubit_error_rates(1e-2, 0.5, drop, 3e-4, rng)  # 1 -> 0 readout
    return cz, sq, e0, e1


_cals = {'notebook calibration': (cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors),
         'seeded calibration': _seeded_calibration(7)}  # two independent devices
_slowest = 0.0  # longest construction time
for _name, _cal in _cals.items():  # both calibrations
    for _N in range(1, 11):  # all small sizes
        for _n in range(0, _N + 1):  # all excitation numbers, including n > N/2 and the trivial ones
            _t = _time.monotonic()  # start timer
            _circ, _qs = create_dicke_state(_N, _n, *_cal)  # the submission function
            _slowest = max(_slowest, _time.monotonic() - _t)  # track the worst case
            check_hardware_rules(_circ, _qs, _cal[0], _cal[1], _N)  # hardware rules
            test_circuit(_circ, _n, _qs)  # exact state (notebook's checker)
    print(f'{_name}: all (N, n) with N <= 10 exact and hardware-valid')
print(f'slowest construction: {_slowest:.1f} s (limit 60 s)')

# %% [markdown]
# Resource summary on the notebook's calibration (CZ count, moments, construction time), including larger instances that
# are too big to simulate as state vectors here but use exactly the same, already verified, code paths.

# %%
print(f"{'N':>3} {'n':>3} {'CZs':>5} {'moments':>8} {'time(s)':>8}")
for _N, _n in [(4, 1), (4, 2), (6, 1), (6, 3), (8, 1), (8, 2), (8, 4), (10, 5), (12, 1), (12, 6), (16, 2), (20, 1), (20, 10), (40, 1), (100, 1)]:
    _t = _time.monotonic()  # start timer
    _circ, _qs = create_dicke_state(_N, _n, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)  # build
    _dt = _time.monotonic() - _t  # elapsed
    check_hardware_rules(_circ, _qs, cz_pauli_error_rate, sq_pauli_error_rate, _N)  # rules still hold
    _ncz = sum(1 for op in _circ.all_operations() if len(op.qubits) == 2)  # CZ count
    print(f'{_N:>3} {_n:>3} {_ncz:>5} {len(_circ):>8} {_dt:>8.1f}')
