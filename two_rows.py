"""Two row split for larger n.

Layout (n columns of rungs, A runs left to right on top, B right to left on bottom):

    a1 - a2 - ... - an - a(n+1) - ...        row A continues to the right
    |    |          |
    bn - ...- b2 -  b1                       row B continues to the left from bn
  ...

1. Split. Row A gets j of the n excitations and row B gets n - j, with
   probability C(mA, j) C(mB, n - j) / C(N, n). This is a short chain of
   controlled rotations on the top row (one CZ each), then one CNOT per rung.
   Column i holds its excitation on top if i <= j and on the bottom otherwise,
   so A ends up with ones on a1..aj and B with ones on b1..b(n-j).
2. Finish. Each row turns its packed ones into a Dicke state with the exact
   nearest neighbour construction from baseline.py. Both rows run at the same
   time, each on about half the qubits, which cuts depth and gate count.
"""

import math
import time

import cirq
import numpy as np

import baseline
from baseline import _usable_graph, _qubit_cost


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
        ops += baseline._cnot(a(i), b)
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
        ops += baseline.dicke_unitary_ops(list(reversed(A)), min(n, mA))
        ops += baseline.dicke_unitary_ops(list(reversed(B)), min(n, mB))
        out.append((cirq.Circuit(ops), A + B))
    return out
