import itertools
import math
import time

import cirq
import cirq_google
import networkx as nx
import numpy as np

# Dicke state preparation on Google Willow.
# For each (N, n) several exact candidate circuits are built (stored short templates
# for small cases, splitting trees for n = 1, a two row split for larger n, a single
# chain fallback), each is placed on low error qubits from the calibration data,
# scheduled as late as possible, given dynamical decoupling pulses, and the candidate
# with the highest predicted success is returned.




# ---------------------------------------------------------------------------
# Error dictionary helpers
# ---------------------------------------------------------------------------

def _pair_key(a, b):
    return frozenset((a, b))


def _usable_graph(cz_pauli_error_rate, sq_pauli_error_rate):
    """Adjacency of qubits that have working single qubit gates and couplers."""
    good = set(sq_pauli_error_rate)
    adj = {q: {} for q in good}
    for pair, (err, unc) in cz_pauli_error_rate.items():
        a, b = tuple(pair)
        if a in good and b in good:
            # pessimistic estimate: a coupler that only looks good by chance is not trusted
            adj[a][b] = err + unc
            adj[b][a] = err + unc
    return adj


def _qubit_cost(q, zero_errors, one_errors, sq_pauli_error_rate, frac_one):
    z = sum(zero_errors.get(q, (0.05, 0)))
    o = sum(one_errors.get(q, (0.05, 0)))
    s = sum(sq_pauli_error_rate.get(q, (0.01, 0)))
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
                if steps > 400000 or time.time() > deadline:
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
    # 5 CZ total: CNOT, a 3 CZ multiplexed rotation on a, CNOT.
    # In the frame after the first CNOT, a must get Ry(phi) when (b, c) = (1, 0),
    # flip from |0> to |1> when (b, c) = (1, 1), and stay put when b = 0.
    u = -(phi + math.pi) / 4
    v = (math.pi - phi) / 4
    ops = []
    ops += _cnot(a, b)
    ops += [cirq.ry(u).on(a), cirq.CZ(c, a), cirq.ry(v).on(a), cirq.CZ(a, b),
            cirq.ry(-v).on(a), cirq.CZ(c, a), cirq.ry(-u).on(a)]
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

def _baseline_create_dicke_state(
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

    if k == 0:
        # product state, no entangling gates and no path needed: pick the best N qubits
        good = sorted(sq_pauli_error_rate, key=lambda q: _qubit_cost(
            q, zero_errors, one_errors, sq_pauli_error_rate, n / N))
        if len(good) < N:
            raise NotImplementedError("Not enough working qubits.")
        chosen = good[:N]
        x = 1 if n == N else 0
        circuit = cirq.Circuit(cirq.Moment(cirq.PhasedXZGate(
            x_exponent=x, z_exponent=0, axis_phase_exponent=0).on(q) for q in chosen))
        return circuit, set(chosen)

    chain = find_path(N, cz_pauli_error_rate, sq_pauli_error_rate,
                      zero_errors, one_errors, frac_one=n / N,
                      time_budget=10.0 if N <= 60 else 30.0)

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






def _grow_tree(root, N, adj, qcost):
    """Broadcast style growth. Returns rounds: list of [(parent, child), ...]."""
    used = {root}
    holders = [root]
    rounds = []
    while len(used) < N:
        G = nx.Graph()
        for h in holders:
            for r in adj[h]:
                if r not in used:
                    # higher weight is better: prefer good couplers and good qubits
                    w = 1.0 - (2.5 * adj[h][r] + qcost[r])
                    G.add_edge(("h", h), ("f", r), weight=w)
        if G.number_of_edges() == 0:
            return None
        m = nx.max_weight_matching(G, maxcardinality=True)
        pairs = []
        for a, b in m:
            if a[0] == "f":
                a, b = b, a
            pairs.append((a[1], b[1]))
        need = N - len(used)
        if len(pairs) > need:
            pairs.sort(key=lambda p: 2.5 * adj[p[0]][p[1]] + qcost[p[1]])
            pairs = pairs[:need]
        for _, c in pairs:
            used.add(c)
        holders += [c for _, c in pairs]
        rounds.append(pairs)
    return rounds


def _tree_cost(root, rounds, adj, qcost):
    c = qcost[root]
    for rnd in rounds:
        for p, ch in rnd:
            c += qcost[ch] + 2 * 2.5 * adj[p][ch]
    c += 0.004 * len(rounds) * sum(1 + len(r) for r in rounds) ** 0  # depth penalty
    return c + 0.01 * len(rounds)


def build_w_tree(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors,
                 one_errors, num_roots=25, keep=1):
    trees = _choose_trees(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors,
                          one_errors, num_roots, keep)
    return [_tree_circuit(root, rounds) for _, root, rounds in trees]


def _choose_trees(N, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors,
                  one_errors, num_roots, keep):
    adj = _usable_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    qcost = {q: _qubit_cost(q, zero_errors, one_errors, sq_pauli_error_rate, 1.0 / N)
             for q in adj}
    roots = sorted((q for q in adj if adj[q]),
                   key=lambda q: qcost[q] + min(adj[q].values()))[:num_roots]
    found = []
    for r in roots:
        rounds = _grow_tree(r, N, adj, qcost)
        if rounds is None:
            continue
        found.append((_tree_cost(r, rounds, adj, qcost), r, rounds))
    if not found:
        raise NotImplementedError("No tree found.")
    found.sort(key=lambda t: t[0])
    return found[:keep]


def _tree_circuit(root, rounds):

    # subtree sizes
    children = {root: []}
    for rnd in rounds:
        for p, c in rnd:
            children.setdefault(p, []).append(c)
            children.setdefault(c, [])

    size = {}

    def sz(v):
        if v not in size:
            size[v] = 1 + sum(sz(c) for c in children[v])
        return size[v]

    sz(root)
    # remaining share owned by each node (itself plus children not yet split off)
    remaining = dict(size)

    moments = []
    first = True
    for rnd in rounds:
        m1, m2, m3, m4, m5 = [], [], [], [], []
        for p, c in rnd:
            frac = size[c] / remaining[p]          # share handed to the child
            theta = 2 * math.asin(math.sqrt(frac))
            if first:
                # root and first child both start in |0>:
                # child = Ry(theta)|0>, then root = NOT child
                m1.append(cirq.ry(theta).on(c))
                m2.append(cirq.H(p))
                m3.append(cirq.CZ(c, p))
                m4.append(cirq.PhasedXZGate.from_matrix(
                    cirq.unitary(cirq.X) @ cirq.unitary(cirq.H)).on(p))
            else:
                # controlled Ry on a fresh |0> target (1 CZ), then CNOT child -> parent
                m1.append(cirq.ry(-theta / 2).on(c))
                m2.append(cirq.CZ(p, c))
                m3.append(cirq.ry(theta / 2).on(c))
                m3.append(cirq.H(p))
                m4.append(cirq.CZ(c, p))
                m5.append(cirq.H(p))
            remaining[p] -= size[c]
        if first:
            moments += [cirq.Moment(m1 + m2), cirq.Moment(m3), cirq.Moment(m4)]
            first = False
        else:
            moments += [cirq.Moment(m1), cirq.Moment(m2), cirq.Moment(m3),
                        cirq.Moment(m4), cirq.Moment(m5)]
    qubits = [root] + [c for rnd in rounds for _, c in rnd]
    return cirq.Circuit(moments), qubits






# ---------------------------------------------------------------------------
# Layout search
# ---------------------------------------------------------------------------

def _straight_ladders(adj, n):
    """All placements of two neighbouring straight rows of n qubits with working rungs.

    Returns (top, bottom) where top[i] is adjacent to bottom[i] (same column).
    """
    out = []
    qs = set(adj)
    for q in qs:
        for d, perp in (((0, 1), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (-1, 0)), ((1, 0), (0, -1))):
            top = [q + d_scaled for d_scaled in []]  # placeholder, rebuilt below
            top = []
            bot = []
            ok = True
            for i in range(n):
                t = q.__class__(q.row + d[0] * i, q.col + d[1] * i)
                b = q.__class__(t.row + perp[0], t.col + perp[1])
                if t not in qs or b not in qs or b not in adj[t]:
                    ok = False
                    break
                if i > 0 and (t not in adj[top[-1]] or b not in adj[bot[-1]]):
                    ok = False
                    break
                top.append(t)
                bot.append(b)
            if ok:
                out.append((top, bot))
    return out


def _extend(start, length, adj, qcost, used, deadline):
    """Self avoiding path of `length` new qubits continuing from `start`."""
    if length == 0:
        return []
    best = None
    for attempt in range(4):
        rng = np.random.default_rng(attempt)
        path = []
        local_used = set(used)

        def options(q):
            opts = [r for r in adj[q] if r not in local_used]
            return sorted(opts, key=lambda r: (
                sum(1 for t in adj[r] if t not in local_used),
                qcost[r] + 2.5 * adj[q][r] + 1e-3 * rng.random()))

        stack = [iter(options(start))]
        prev = [start]
        steps = 0
        while stack and len(path) < length:
            steps += 1
            if steps > 5000 or time.time() > deadline:
                break
            nxt = next(stack[-1], None)
            if nxt is None:
                stack.pop()
                if path:
                    local_used.discard(path.pop())
                    prev.pop()
                continue
            path.append(nxt)
            local_used.add(nxt)
            prev.append(nxt)
            stack.append(iter(options(nxt)))
        if len(path) == length:
            c = sum(qcost[q] for q in path)
            if best is None or c < best[0]:
                best = (c, list(path))
    return None if best is None else best[1]


def find_two_rows(N, n, mA, mB, adj, qcost, time_budget=15.0, keep=1):
    deadline = time.time() + time_budget
    ladders = _straight_ladders(adj, n)

    def lcost(top, bot):
        c = sum(qcost[q] for q in top + bot)
        c += sum(2.5 * adj[t][b] for t, b in zip(top, bot))
        c += sum(2.5 * (adj[top[i]][top[i + 1]] + adj[bot[i]][bot[i + 1]]) for i in range(n - 1))
        return c

    ladders.sort(key=lambda tb: lcost(*tb))
    found = []
    for top, bot in ladders[:60]:
        if time.time() > deadline:
            break
        # A: a1 = top[0] ... an = top[n-1], continues from top[n-1]
        # B: b1 = bot[n-1] ... bn = bot[0],  continues from bot[0]
        A = list(top)
        B = list(reversed(bot))
        used = set(A) | set(B)
        extA = _extend(A[-1], mA - n, adj, qcost, used, deadline)
        if extA is None:
            continue
        used |= set(extA)
        extB = _extend(B[-1], mB - n, adj, qcost, used, deadline)
        if extB is None:
            continue
        A += extA
        B += extB
        cost = lcost(top, bot) + sum(qcost[q] for q in extA + extB)
        found.append((cost, A, B))
        if len(found) >= max(8, 3 * keep):
            break
    if not found:
        return None
    found.sort(key=lambda t: t[0])
    return [(A, B) for _, A, B in found[:keep]]


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

def _split_ops(A, B, n, mA, mB):
    """Thermometer on a1..an plus rungs. Column i (1 based): a_i and b_(n+1-i)."""
    N = mA + mB
    P = [math.comb(mA, j) * math.comb(mB, n - j) / math.comb(N, n) for j in range(n + 1)]
    ge = [sum(P[j:]) for j in range(n + 2)]   # P(j >= i)
    le = [sum(P[:i + 1]) for i in range(n + 1)]  # P(j <= i)

    def ang(p):
        p = min(max(p, 0.0), 1.0)
        return 2 * math.asin(math.sqrt(p))

    s = (n + 1) // 2  # start column, generate outward both ways
    ops = []
    a = lambda i: A[i - 1]
    # start column: marginal
    ops.append(cirq.ry(ang(ge[s])).on(a(s)))
    # right of start: x_i = 0 if x_(i-1) = 0, else 1 with P(j >= i | j >= i-1)
    right = []
    for i in range(s + 1, n + 1):
        th = ang(ge[i] / ge[i - 1]) if ge[i - 1] > 0 else 0.0
        right.append((i, i - 1, -th / 2, th / 2))
    # left of start: x_i = 1 if x_(i+1) = 1, else 1 with P(j = i | j <= i)
    left = []
    for i in range(s - 1, 0, -1):
        th = ang(P[i] / le[i]) if le[i] > 0 else 0.0
        left.append((i, i + 1, (th - math.pi) / 2, (th + math.pi) / 2))
    for seq in (right, left):
        for (i, ctrl, alpha, beta) in seq:
            ops.append(cirq.ry(alpha).on(a(i)))
            ops.append(cirq.CZ(a(ctrl), a(i)))
            ops.append(cirq.ry(beta).on(a(i)))
    # rungs: b_(n+1-i) = NOT x_i
    for i in range(1, n + 1):
        b = B[n - i]
        ops.append(cirq.X(b))
        ops += _cnot(a(i), b)
    return ops


def build_two_rows(N, n, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors,
                   keep=1):
    """Requires 1 <= n <= N/2. Returns (circuit, qubit list) or None if no layout fits."""
    mB = N // 2
    mA = N - mB
    if n > mB:
        return None
    adj = _usable_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    qcost = {q: _qubit_cost(q, zero_errors, one_errors, sq_pauli_error_rate, n / N)
             for q in adj}
    layouts = find_two_rows(N, n, mA, mB, adj, qcost, keep=keep)
    if layouts is None:
        return None
    out = []
    for A, B in layouts:
        ops = _split_ops(A, B, n, mA, mB)
        # finish each row: baseline expects the ones at the END of the chain list
        ops += dicke_unitary_ops(list(reversed(A)), min(n, mA))
        ops += dicke_unitary_ops(list(reversed(B)), min(n, mB))
        out.append((cirq.Circuit(ops), A + B))
    return out



_X = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
_Y = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0.5)


def add_decoupling(circuit, min_gap=2):
    """Fill idle stretches of every qubit (after its first gate) with X X or X Y X Y.

    Idle stretches run until the end of the circuit, since the measurement comes
    right after.
    """
    moments = [list(m.operations) for m in circuit]
    busy = {}
    for t, mom in enumerate(moments):
        for op in mom:
            for q in op.qubits:
                busy.setdefault(q, []).append(t)
    T = len(moments)
    for q, times in busy.items():
        times = sorted(set(times))
        # gaps after each busy moment
        bounds = list(zip(times, times[1:] + [T]))
        for start, end in bounds:
            gap = end - start - 1          # free moments strictly between
            if gap < min_gap:
                continue
            seq = [_X, _Y, _X, _Y] if gap >= 4 else [_X, _X]
            L = len(seq)
            # evenly spaced pulse positions inside the free window
            slots = [start + 1 + int((i + 0.5) * gap / L) for i in range(L)]
            for s, g in zip(slots, seq):
                moments[s].append(g.on(q))
    return cirq.Circuit(cirq.Moment(m) for m in moments)




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


TEMPLATES = {
    "4,2": {
        "m": 5,
        "cells": [(0, 0), (0, 1), (1, 1), (1, 0)],
        "edges": [(0, 1), (0, 3), (1, 2), (3, 2)],
        "seq": [3, 2, 1, 0, 3],
        "params": [0.9152251869199917, -1.570796325403217, 2.0900648210551114, 1.8692011249470026, -1.570796325562261, 2.3655201874146266, -2.4179334405501467, -1.6246094335940167, -2.197358915225356, -0.5760806525818848, 2.410567616738059, -1.6924249148248318, -1.7956944685060041, -1.06279538834646, -2.158352099813202, -2.2140202840636007, 1.2166626717544191, -0.9091641032693085, 1.0167743919391317, -1.6722008125477743, 2.8431878576638736, -0.651597662858857, -1.5707963267137959, 1.1685646075017446, 2.652700642515031, 8.791651371615145e-10, 2.997493840136398, -0.7369791770867095, -1.5707963258859357, 1.7743621060038888, 4.144244618119247, 1.570796328026762, -0.2822343628593521, 1.4364014618324719, -1.3113045784793899e-09, 0.12027243612363181, 1.0026519639928766, -1.5707963241025313, 0.736979177858213, 1.2691374818911962, 3.1415926554193554, -2.2235094747305855],
    },
    "5,2": {
        "m": 10,
        "cells": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        "edges": [(0, 1), (0, 3), (1, 2), (1, 4), (3, 4)],
        "seq": [3, 2, 4, 0, 1, 0, 4, 3, 1, 4],
        "params": [1.0978818961939096, -0.7853981242917613, 1.4727555223511677, 1.3757814194126186, -0.9170366410619162, 1.7773964788258096, 2.926994429124498, -1.5707963040032937, -0.9655221139379317, 0.796969491675068, 0.6154798365710585, 1.9088611740202805, 0.9749007888954939, 2.4764731195617986, 2.086792667558297, 0.16814061795965682, 1.102983753098459, -0.8222705778608492, 1.5183442695097957, 2.9331039340755223, 0.9950968113067814, -2.2225822412902643, 1.055003266210752e-07, -0.8266343538038219, 0.3460466163492173, 1.5707962439153007, 0.21459822409394888, 1.0233899020030446, 0.6154797435120007, -0.7962892193437947, -3.1563444034644546, 1.5707960378431718, -0.007921372872869046, 3.3783045745121156, 1.570894713468651, -2.6685800860476085, 0.9629152381633073, 1.57079631071484, 1.6631556468854818, 1.0249551738824412, -1.5709348523232949, 0.5486862494919814, -0.6822053057212604, -1.5707963153536546, 0.5472672091202356, -1.7330199705327285, -1.5707963208246396, -1.0246777971383083, 2.9713176656497753, -3.1415926639658607, 1.778396019855926, -0.10054050223156384, 2.3561944349949893, 0.681813355783217, 0.47724778407631113, -0.7853981431521005, -3.1276252747892106, 1.1666465249312237, 1.5707963109177905, 2.9115862150569454, -2.2698559679921844, 6.705351474039285e-09, -0.0016836599857636219, -0.40442749188590676, 1.5707963006124381, 1.7330200043981783, 1.4971098712803268, 3.141592638313141, 2.368445670371039, 0.09929133328881226, 3.1415926979055717, 1.2750682106928366, -3.517720427914767, -3.358185736798283e-08, 0.19547327980386328],
    },
    "6,2": {
        "m": 15,
        "cells": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        "edges": [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)],
        "seq": [5, 1, 5, 4, 6, 4, 5, 2, 3, 0, 6, 1, 3, 4, 1],
        "params": [1.2278237895533397, 1.401178952441533, -2.7971731079931152, -2.5808665155802815, 1.5707963344531495, 2.3075858676623557, -2.302523838510664, 1.5707962901614496, -0.6545982908253885, -2.118287085663263, -0.45244323044809076, -0.9086714810104131, -4.0774646534769685, 1.4859614921461801, -0.8289020056393221, 2.4829013281105987, 1.5707963599748902, -1.5320151967703777, 1.453504434531185, -4.552213382295043, 2.44602334321934, 2.241999021287189, 1.1051961447881224, 1.4155261252380038, 1.942191141590358, -2.475483813008775, 1.9963999541010988, -1.8153612762882163, -1.6325335641088896, -0.8354025860910187, 2.888566821858411, 0.859519513675752, 2.519530923746934, -2.6918006043651137, -2.6348527828796633, -0.530288420449832, -0.031998291676036614, -1.8888729826046289, -0.3096573741564422, -1.7932385610057182, -1.963318048625143, 0.6028245962491279, -0.4446398191320723, -0.7853981786596648, 0.2868477619827955, -0.1544448782912649, -2.186276035512955, -2.7892455073449574, -1.7593069259144232, 1.5707964014514053, 2.092779927097489, 0.49633270849886335, 1.5707963807323797, 2.5106393259099633, -1.4188917551422757, -1.57079631237123, -1.1777392583960136, -2.605196610913039, 4.440195390281019, 2.2983149472950113, 2.5995270063708524, -1.5707963210396247, -3.7023188065502763, 1.51198075264584, 1.570796337621507, -1.693658105011778, -2.2114944561961374, 1.5707963178935218, -0.7173583394512172, 0.16912117937826204, -1.5707963482373573, -1.2828003532518621, -1.0348867239002373, -1.5707963511524992, -1.3674305728941436, -0.38974245103793564, 2.9467459649812683e-08, -2.424364869282272, 1.278496800532173, -3.141592613922145, -0.9952732781983203, 3.3468985622889664, 1.5707963506896416, -2.0671290550595534, -1.0856447003477343, -2.3561944888228528, 2.605683015157323, 0.7623685122886664, 1.570796350559726, 0.6334935986189051, -0.34739482161901575, 1.570796321655559, -1.2575835317253838, -0.7541170489891756, 1.1361168153401097e-08, 0.2628866377054098, -1.52817417510431, 1.5707963148155355, 1.6296119139426695, -1.8458725245988963, 3.141592669244836, 2.618588390205919, 3.3219912440709796, -1.5707963287100164, 1.0856446449391086, 0.1803986058817658, -1.5707962967666567, -2.3331648546130555],
    },
    "6,3": {
        "m": 14,
        "cells": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        "edges": [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)],
        "seq": [1, 2, 3, 6, 4, 5, 6, 4, 5, 6, 0, 3, 2, 5],
        "params": [-0.004358709043469499, -1.5707963280527075, 2.1646975265774966, 0.69052433826127, -1.4870179470153946, 2.1129187488917247, -2.6894721547872655, -0.7687799642392928, 2.672643051737504, 1.843160338226393, -1.5707963250015535, 0.12061533467884525, -0.7203065711589474, -2.255515554279712, -1.1523598426037729, -0.07536876045467923, -1.5707963847870503, 2.7631927091716424, -0.921409959511693, 1.3787311015441353, 2.437941252853558, 1.9003876674110678, 1.3597030391806944, 2.071534049549365, -1.3687913033373285, 1.948614278091, 1.8131899942308005, 1.5068387806050083, -0.5419279901627104, -1.43341971289844, 1.8552564157112017, 0.7853981799774187, 0.1034326672828162, 0.3245717925172843, -1.5707965112445277, -0.8504896831131051, -1.7553704098846912, 2.186276085949012, -0.6052804699646529, -1.3689612566630083, 1.5707963538630667, 0.07536880918777478, -0.5990504118756445, 1.8925468830959664, 3.0654892348700193, -1.0810624751546987, 0.9553166780067274, -1.4919227247967852, -3.096535663616525, -1.570796324778838, -2.390107791807479, 1.027401578436607, -1.5707963319509268, 0.9699723486586119, -2.208853910024467, -1.570796346732418, -2.598197908409128, 2.2746113343957597, -1.5707963310662383, -2.8459282677185844, 1.9951728548161158, -1.5707963322841036, 0.5990503619565999, -1.2534375342732498, 3.1415926276652546, -0.573961686242868, 1.586520023435268, 1.5707960164674422, -1.6158533118724603, -3.1116907476219002, -1.5707963378760128, -0.9327387570582413, -1.1124263210456944, 0.758673164741977, -0.02990197929149771, 1.4480070100233413, 1.5707963341829758, -1.7583017947930724, -1.6935856734954495, 2.329469431042431, 1.5109236723221127, 3.7033178073888386, 1.2844876246135925e-08, 3.5335844608605527, -1.1421604475949083, 1.5707963278762889, 3.474212003097007, -1.394733530546847, -1.5707963744861881, -0.45836987289150216, -2.4964677985562322, 5.508105106444252e-08, -1.196550057693644, 1.4480070083985028, -1.5707963644129843, 1.1464198044195317, 3.2388748403708822, -1.611814724257924e-08, 2.9057974451167095, 3.0188033320109473, 1.570796311367528, 3.7242030349228847],
    },
}





def _su2(a, b, c):
    return np.array([[np.exp(-0.5j * (a + c)) * np.cos(b / 2), -np.exp(-0.5j * (a - c)) * np.sin(b / 2)],
                     [np.exp(0.5j * (a - c)) * np.sin(b / 2), np.exp(0.5j * (a + c)) * np.cos(b / 2)]])


def _symmetries(cells):
    out = []
    for flip in (False, True):
        for rot in range(4):
            pts = []
            for (r, c) in cells:
                if flip:
                    c = -c
                for _ in range(rot):
                    r, c = c, -r
                pts.append((r, c))
            out.append(pts)
    return out


def _template_candidates(N, k, cz, sq, z0, o1, keep=3):
    key = f"{N},{k}"
    if key not in TEMPLATES:
        return []
    t = TEMPLATES[key]
    cells, edges, seq, params = t["cells"], t["edges"], t["seq"], t["params"]
    used = {}
    for e in seq:
        used[e] = used.get(e, 0) + 1
    adj = _usable_graph(cz, sq)
    qcost = {q: _qubit_cost(q, z0, o1, sq, k / N) for q in adj}
    places = []
    seen = set()
    for pts in _symmetries(cells):
        for anchor in adj:
            r0, c0 = anchor.row - pts[0][0], anchor.col - pts[0][1]
            qs = [cirq.GridQubit(r0 + r, c0 + c) for (r, c) in pts]
            if any(q not in adj for q in qs):
                continue
            ok = True
            cost = sum(qcost[q] for q in qs)
            for e, cnt in used.items():
                a, b = qs[edges[e][0]], qs[edges[e][1]]
                if b not in adj[a]:
                    ok = False
                    break
                cost += 1.25 * cnt * adj[a][b]
            if not ok:
                continue
            sig = (frozenset(qs), tuple(qs))
            if sig in seen:
                continue
            seen.add(sig)
            places.append((cost, qs))
    places.sort(key=lambda x: x[0])
    out = []
    for _, qs in places[:keep]:
        ops = []
        p = 0
        for i in range(N):
            ops.append(cirq.MatrixGate(_su2(*params[p:p + 3])).on(qs[i]))
            p += 3
        for e in seq:
            i, j = edges[e]
            ops.append(cirq.CZ(qs[i], qs[j]))
            ops.append(cirq.MatrixGate(_su2(*params[p:p + 3])).on(qs[i]))
            ops.append(cirq.MatrixGate(_su2(*params[p + 3:p + 6])).on(qs[j]))
            p += 6
        out.append((cirq.Circuit(ops), qs))
    return out






USE_DECOUPLING = True
_FLIP = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)


def _finish(raw_circuit, qubits, flip):
    c = raw_circuit
    if flip:
        c = c + cirq.Circuit(cirq.Moment(_FLIP.on(q) for q in qubits))
    c = compile_to_native(c)
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
        return _baseline_create_dicke_state(N, n, *errs)

    raw = []   # (circuit before compiling, qubit list)
    if True:
        try:
            raw += _template_candidates(N, k, *errs)
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
            chain = find_path(N, *errs, frac_one=n / N,
                                       time_budget=10.0 if N <= 60 else 30.0)
            ops = [cirq.X(q) for q in chain[N - k:]]
            if N > 1:
                ops += dicke_unitary_ops(chain, k)
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


# ---------------------------------------------------------------------------
# Bonus. Reuses the main construction on uniform calibration data.
# ---------------------------------------------------------------------------

def _ideal_errors():
    qs = list(cirq_google.Willow105.metadata.qubit_set)
    pairs = [tuple(p) for p in cirq_google.Willow105.metadata.qubit_pairs]
    cz = {p: (2.5e-3, 0.0) for p in pairs}
    sq = {q: (4e-4, 0.0) for q in qs}
    z0 = {q: (5e-3, 0.0) for q in qs}
    o1 = {q: (1e-2, 0.0) for q in qs}
    return cz, sq, z0, o1


def create_dicke_state_adaptive(num_qubits, num_excitations):
    global USE_DECOUPLING
    saved = USE_DECOUPLING
    USE_DECOUPLING = False
    try:
        circuit, qubits = create_dicke_state(num_qubits, num_excitations, *_ideal_errors())
    finally:
        USE_DECOUPLING = saved
    return circuit, qubits
