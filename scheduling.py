"""Scheduling and a quick predicted success estimate used to pick between candidates.

alap   pushes every gate as late as possible, so qubits stay in |0>, where they
       do not decay, for as long as possible.
predict_success  rough probability that a shot comes out right: gate errors,
       decay and dephasing while a qubit is active, and readout errors.
"""

import math

import cirq

T1_US = 68.0
T2_US = 100.0
DT_CZ_US = 0.042
DT_SQ_US = 0.025


def alap(circuit):
    ops = list(circuit.all_operations())
    rev = cirq.Circuit(reversed(ops))      # earliest placement of the reversed list
    return cirq.Circuit(reversed(list(rev)))


def _pair_err(cz, a, b):
    v = cz.get((a, b)) or cz.get((b, a))
    return v[0] + v[1]


def predict_success(circuit, qubits, n, cz, sq, z0, o1):
    N = len(qubits)
    logp = 0.0
    dur = []
    for mom in circuit:
        has2 = False
        for op in mom:
            if len(op.qubits) == 2:
                has2 = True
                logp += math.log(max(1e-9, 1 - 1.25 * _pair_err(cz, *op.qubits)))
            else:
                q = op.qubits[0]
                logp += math.log(max(1e-9, 1 - sum(sq[q])))
        dur.append(DT_CZ_US if has2 else DT_SQ_US)
    # time each qubit spends after its first gate
    first = {}
    for t, mom in enumerate(circuit):
        for op in mom:
            for q in op.qubits:
                first.setdefault(q, t)
    total = sum(dur)
    suffix = [0.0] * (len(dur) + 1)
    for t in range(len(dur) - 1, -1, -1):
        suffix[t] = suffix[t + 1] + dur[t]
    f = n / N if N else 0
    rate = f / T1_US + 1 / (2 * T2_US)     # decay only hurts the part that is 1
    for q, t in first.items():
        logp -= suffix[t] * rate
    for q in qubits:
        e = (1 - f) * sum(z0.get(q, (0.05, 0))) + f * sum(o1.get(q, (0.05, 0)))
        logp += math.log(max(1e-9, 1 - e))
    return math.exp(logp)
