"""Exact Dicke state preparation on Willow along a single snake path of qubits.

Method: Bartschi Eidenbenz (2019) Dicke unitary, rewritten so every gate acts on
neighbouring qubits of a single path ("snake") through the chip.

Picture: the n ones start packed at one end of the path. We then decide the
qubits one at a time. When r ones are still left among the j undecided qubits,
the current qubit keeps a 1 with amplitude sqrt(r/j). If it gives its 1 away,
the 1 is passed back to the edge of the packed block, using only nearest
neighbour gates.
"""

import itertools
import math

import cirq
import cirq_google
import numpy as np


# ---------------------------------------------------------------------------
# Error dictionary helpers
# ---------------------------------------------------------------------------

def _pair_key(a, b):
    return frozenset((a, b))


def _usable_graph(cz_pauli_error_rate, sq_pauli_error_rate):
    """Adjacency of qubits that have working single qubit gates and couplers."""
    good = set(sq_pauli_error_rate)
    adj = {q: {} for q in good}
    for pair, (err, _unc) in cz_pauli_error_rate.items():
        a, b = tuple(pair)
        if a in good and b in good:
            adj[a][b] = err
            adj[b][a] = err
    return adj


def _qubit_cost(q, zero_errors, one_errors, sq_pauli_error_rate, frac_one):
    z = zero_errors.get(q, (0.05, 0))[0]
    o = one_errors.get(q, (0.05, 0))[0]
    s = sq_pauli_error_rate.get(q, (0.01, 0))[0]
    return (1 - frac_one) * z + frac_one * o + 4 * s


# ---------------------------------------------------------------------------
# Choosing a path of qubits
# ---------------------------------------------------------------------------

def find_path(num_qubits, cz_pauli_error_rate, sq_pauli_error_rate,
              zero_errors, one_errors, frac_one=0.5, time_budget=10.0):
    """Find a low error simple path of num_qubits qubits on the usable graph."""
    import time

    adj = _usable_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    qcost = {q: _qubit_cost(q, zero_errors, one_errors, sq_pauli_error_rate, frac_one)
             for q in adj}
    if num_qubits == 1:
        best = min(adj, key=lambda q: qcost[q])
        return [best]

    deadline = time.time() + time_budget
    best_path, best_cost = None, float("inf")

    def path_cost(p):
        c = sum(qcost[q] for q in p)
        c += sum(2.5 * adj[p[i]][p[i + 1]] for i in range(len(p) - 1))
        return c

    # starts ordered by local quality
    starts = sorted(adj, key=lambda q: qcost[q] + min(adj[q].values(), default=1))
    for s in starts:
        if time.time() > deadline:
            break
        # randomized greedy depth first search with Warnsdorff tie breaking
        for attempt in range(3):
            rng = np.random.default_rng(hash((s, attempt)) % (2 ** 32))
            path = [s]
            used = {s}
            stack = [iter(())]

            def options(q):
                opts = [r for r in adj[q] if r not in used]
                # prefer few onward options (keeps the path from boxing itself in),
                # then low error
                return sorted(opts, key=lambda r: (
                    sum(1 for t in adj[r] if t not in used),
                    qcost[r] + 2.5 * adj[q][r] + 1e-3 * rng.random()))

            stack = [iter(options(s))]
            steps = 0
            while stack and len(path) < num_qubits:
                steps += 1
                if steps > 20000 or time.time() > deadline:
                    break
                nxt = next(stack[-1], None)
                if nxt is None:
                    stack.pop()
                    used.discard(path.pop())
                    continue
                path.append(nxt)
                used.add(nxt)
                stack.append(iter(options(nxt)))
            if len(path) == num_qubits:
                c = path_cost(path)
                if c < best_cost:
                    best_cost, best_path = c, list(path)
    if best_path is None:
        raise NotImplementedError("No path of the requested length found.")
    return best_path


# ---------------------------------------------------------------------------
# Logical gates (later compiled to CZ + PhasedXZ)
# ---------------------------------------------------------------------------

def _cry(control, target, theta):
    """Exact controlled Ry(theta) with 2 CZs."""
    return [
        cirq.ry(theta / 2).on(target),
        cirq.CZ(control, target),
        cirq.ry(-theta / 2).on(target),
        cirq.CZ(control, target),
    ]


def _cnot(control, target):
    return [cirq.H(target), cirq.CZ(control, target), cirq.H(target)]


def _mux_ry_two_controls(b, c, a, f00, f01, f10, f11):
    """Exact Ry on target a with angle f[b][c] (uniformly controlled rotation).

    a must neighbour both b and c. Uses 4 CZs.
    Sequence in time: Ry(t1) CX(c,a) Ry(t2) CX(b,a) Ry(t3) CX(c,a) Ry(t4) CX(b,a).
    """
    # parity of fired controls before each rotation: p1=0, p2=c, p3=c^b, p4=b
    rows = []
    rhs = []
    for (bb, cc, f) in [(0, 0, f00), (0, 1, f01), (1, 0, f10), (1, 1, f11)]:
        par = [0, cc, cc ^ bb, bb]
        rows.append([(-1) ** p for p in par])
        rhs.append(f)
    t = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
    ops = []
    ops.append(cirq.ry(t[0]).on(a))
    ops += _cnot(c, a)
    ops.append(cirq.ry(t[1]).on(a))
    ops += _cnot(b, a)
    ops.append(cirq.ry(t[2]).on(a))
    ops += _cnot(c, a)
    ops.append(cirq.ry(t[3]).on(a))
    ops += _cnot(b, a)
    return ops


def _split_angle(c, j):
    """Angle so the wall pair becomes sqrt(c/j)|01> + sqrt((j-c)/j)|10>."""
    alpha = math.sqrt(c / j)
    beta = math.sqrt((j - c) / j)
    return 2 * math.atan2(beta, alpha)


def _plain_unit(a, b, phi):
    """Pair (a, b) holding 00, 01 or 11. 01 -> cos|01> + sin|10>."""
    ops = []
    ops += _cnot(a, b)
    ops += _cry(b, a, phi)
    ops += _cnot(a, b)
    return ops


def _controlled_unit(c, a, b, phi):
    """Triple (c, a, b) on a line. If c is 0 this acts as the plain unit;
    if c is 1 (a hole is being carried through), 01 on (a, b) becomes 10."""
    ops = []
    ops += _cnot(a, b)
    ops += _mux_ry_two_controls(b, c, a, 0.0, 0.0, phi, math.pi)
    ops += _cnot(a, b)
    return ops


def dicke_unitary_ops(chain, max_count):
    """Bartschi Eidenbenz Dicke unitary on a nearest neighbour chain.

    chain[0..m-1] are BE positions 1..m. Maps the unary state with l ones at the
    END of the chain (positions m-l+1..m) to D(m, l), for all l <= max_count.
    """
    m = len(chain)
    pos = lambda p: chain[p - 1]  # BE position p -> qubit
    ops = []
    for j in range(m, 1, -1):
        k = min(max_count, j - 1)
        for p in range(j - k, j):
            c = j - p  # number of ones to the right of the wall if the wall sits at p
            phi = _split_angle(c, j)
            if p == j - k or p == 1:
                ops += _plain_unit(pos(p), pos(p + 1), phi)
            else:
                ops += _controlled_unit(pos(p - 1), pos(p), pos(p + 1), phi)
    return ops


# ---------------------------------------------------------------------------
# Compilation to the allowed gate set
# ---------------------------------------------------------------------------

def compile_to_native(circuit):
    """Rewrite into CZ + PhasedXZ only, merge single qubit gates."""
    c = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.CZTargetGateset(allow_partial_czs=False))
    c = cirq.merge_single_qubit_gates_to_phxz(c)
    c = cirq.drop_negligible_operations(c)
    c = cirq.drop_empty_moments(c)
    # make sure every single qubit op is a PhasedXZGate
    out = []
    for op in c.all_operations():
        if len(op.qubits) == 1:
            out.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(op)).on(*op.qubits))
        else:
            out.append(op)
    return cirq.Circuit(out)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

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

    flip = n > N - n          # build the smaller case, flip all qubits at the end
    k = N - n if flip else n

    chain = find_path(N, cz_pauli_error_rate, sq_pauli_error_rate,
                      zero_errors, one_errors, frac_one=n / N)

    ops = []
    # start: k ones packed at the end of the chain
    for q in chain[N - k:]:
        ops.append(cirq.X(q))
    if 0 < k < N:
        ops += dicke_unitary_ops(chain, k)
    if flip:
        ops += [cirq.X(q) for q in chain]
    if not ops:
        ops = [cirq.I(q) for q in chain]

    circuit = compile_to_native(cirq.Circuit(ops))
    # keep every output qubit in the circuit
    missing = [q for q in chain if q not in circuit.all_qubits()]
    if missing:
        circuit.append(cirq.PhasedXZGate(x_exponent=0, z_exponent=0,
                                         axis_phase_exponent=0).on_each(*missing))
    if len(circuit) > 1750:
        raise NotImplementedError("Circuit too deep.")
    return circuit, set(chain)
