"""W state (n = 1) on Willow with a splitting tree.

The single excitation starts on one qubit and is split outward. In every round
each qubit that still owns part of the excitation hands a share to one fresh
neighbour, so the number of qubits involved roughly doubles per round and the
depth grows like log N instead of N.

Each split costs 2 CZ gates, except the very first one, which costs 1.
Total CZ count is 2N - 3.
"""

import math

import cirq
import networkx as nx
import numpy as np

from baseline import _usable_graph, _qubit_cost


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
