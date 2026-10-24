"""Shortest known exact circuits for small cases, found offline by numerical search.

Each template is a fixed CZ sequence on a small patch of grid cells plus the exact
single qubit gates. At run time the patch is placed on the chip in every rotation,
reflection and position that the working couplers allow, and the best placements are
returned as candidates.
"""

import cirq
import numpy as np

from baseline import _usable_graph, _qubit_cost
from templates_data import TEMPLATES


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


def candidates(N, k, cz, sq, z0, o1, keep=3):
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
