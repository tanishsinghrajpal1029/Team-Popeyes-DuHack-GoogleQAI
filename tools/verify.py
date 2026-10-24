"""Exactness + hardware-rule verification of the solution on random fake calibrations."""
import sys, time
sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
import itertools, math
import cirq, numpy as np
import dicke_solution as D
from fake_calibration import make_calibration


def dicke_vector(n_qubits, k):
    psi = np.zeros(2**n_qubits)
    for exc in itertools.combinations(range(n_qubits), k):
        psi[sum(1 << (n_qubits - 1 - i) for i in exc)] = 1
    return psi / np.linalg.norm(psi)


def fidelity(circuit, qubits, k):
    qubits = sorted(qubits)
    psi = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits).final_state_vector
    return abs(np.vdot(dicke_vector(len(qubits), k), psi)) ** 2


def hardware_ok(circuit, qubits, cal):
    cz, sq = cal[0], cal[1]
    pairs = {frozenset(p) for p in cz}
    assert circuit.all_qubits() <= set(qubits) | set(), 'ancilla present'
    for m in circuit:
        kinds = {len(op.qubits) for op in m}
        assert len(kinds) == 1, 'mixed moment'
        for op in m:
            if len(op.qubits) == 2:
                assert op.gate == cirq.CZ and frozenset(op.qubits) in pairs, op
            else:
                assert isinstance(op.gate, cirq.PhasedXZGate) and op.qubits[0] in sq, op
    assert len(circuit) <= 1750


if __name__ == '__main__':
    worst = 0
    for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
        cal = make_calibration(seed)
        for N in range(1, 13):
            for n in range(0, N + 1):
                t = time.time()
                circ, qs = D.create_dicke_state(N, n, *cal)
                dt = time.time() - t
                worst = max(worst, dt)
                hardware_ok(circ, qs, cal)
                assert len(qs) == N
                f = fidelity(circ, qs, n)
                assert abs(f - 1) < 1e-8, (seed, N, n, f)
            print('seed', seed, 'N', N, 'ok', flush=True)
    print('all exact; slowest call %.1fs' % worst)
