"""Dicke-state preparation on Google Willow (DuQuantum x Google Quantum AI, standard track).

Only cirq, cirq_google, their dependencies (numpy, networkx) and the standard library are
imported, as the challenge requires.

Construction (baseline): Baertschi & Eidenbenz, "Deterministic Preparation of Dicke
States" (arXiv:1904.07358). We build the logical circuit, place it on a low-error path of
calibrated qubits, route it with cirq.RouteCQC, and compile to CZ + PhasedXZ. The Dicke
state is permutation-symmetric, so wherever routing leaves the logical qubits is a valid
answer; we just return that set.
"""

import math

import cirq
import networkx as nx
import numpy as np


# ---- calibration graph ----------------------------------------------------------------


def _usable_graph(cz_pauli_error_rate: dict, sq_pauli_error_rate: dict) -> nx.Graph:
    """Calibrated couplers whose endpoints both have single-qubit calibrations."""
    g = nx.Graph()
    for pair, (err, _) in cz_pauli_error_rate.items():
        a, b = tuple(pair)  # keys are tuple(frozenset): order is arbitrary
        if a in sq_pauli_error_rate and b in sq_pauli_error_rate:
            g.add_edge(a, b, err=err)
    return g


def _qubit_cost(q, sq_pauli_error_rate, zero_errors, one_errors, p_one: float, n_sq: float = 1) -> float:
    """Expected per-qubit error: readout weighted by how often the qubit reads 1, plus
    `n_sq` single-qubit gates."""
    e0 = zero_errors.get(q, (0.5, 0))[0]
    e1 = one_errors.get(q, (0.5, 0))[0]
    return (1 - p_one) * e0 + p_one * e1 + n_sq * sq_pauli_error_rate[q][0]


def _best_path(g: nx.Graph, length: int, node_cost: dict, beam: int = 256) -> list | None:
    """Beam search for a simple path of `length` qubits minimizing edge + node cost."""
    frontier = sorted(((node_cost[q], [q]) for q in g.nodes), key=lambda t: t[0])[:beam]
    for _ in range(length - 1):
        grown = []
        seen = set()
        for cost, path in frontier:
            used = set(path)
            for nb in g.neighbors(path[-1]):
                if nb in used:
                    continue
                key = (frozenset(used | {nb}), nb)
                if key in seen:
                    continue
                seen.add(key)
                grown.append((cost + g.edges[path[-1], nb]["err"] + node_cost[nb], path + [nb]))
        if not grown:
            return None
        grown.sort(key=lambda t: t[0])
        frontier = grown[:beam]
    return frontier[0][1]


def _connected_patch(g: nx.Graph, size: int, node_cost: dict) -> list | None:
    """Greedy connected set of `size` qubits, grown from the cheapest qubit outward."""
    for comp in sorted(nx.connected_components(g), key=len, reverse=True):
        if len(comp) < size:
            return None
        start = min(comp, key=node_cost.get)
        chosen, frontier = [start], set(g.neighbors(start))
        while len(chosen) < size:
            nxt = min(frontier, key=lambda q: node_cost[q])
            chosen.append(nxt)
            frontier |= set(g.neighbors(nxt))
            frontier -= set(chosen)
        return chosen
    return None


# ---- W state (n = 1): broadcast tree of amplitude splits ------------------------------

# Idle-error model for layout decisions. The calibration data has no T1/T2, so we use
# the Willow spec-sheet T1 (68 us) and an assumed T_phi (100 us). One tree round is
# 2 CZ layers (42 ns) + 2 single-qubit layers (25 ns). A W state carries exactly one
# excitation in every branch, so its idle loss per round is independent of N.
_ROUND_NS = 2 * 42 + 2 * 25
_IDLE_RATE_PER_NS = 1 / 68e3 + 1 / 100e3


def _readout_lambda(qubits: list, zero_errors, one_errors, n: int, f: float) -> float:
    """Chi^2 noncentrality from readout alone: a valid string x survives readout with
    relative weight ~ exp(-sum_{i in x} (e1_i - e0_i)), so its spread over random
    n-subsets is n (N-n)/(N-1) var(e1 - e0)."""
    N = len(qubits)
    if N < 2:
        return 0.0
    d = np.array([one_errors[q][0] - zero_errors[q][0] for q in qubits])
    return 1e4 * f * n * (N - n) / (N - 1) * float(np.var(d))


def _grow_tree(g: nx.Graph, root, N: int, node_cost: dict):
    """Each round, every qubit holding amplitude may split it onto one fresh neighbour.
    Greedy: take the cheapest (edge + new-qubit) splits first. Returns (rounds, cost)
    where rounds is a list of [(parent, child), ...]."""
    nodes, used, rounds, cost = [root], {root}, [], node_cost[root]
    while len(nodes) < N:
        candidates = sorted(
            (2 * g.edges[p, c]["err"] + node_cost[c], p, c)
            for p in nodes for c in g.neighbors(p) if c not in used
        )
        busy, splits = set(), []
        for c_cost, p, c in candidates:
            if p in busy or c in busy:
                continue
            splits.append((p, c))
            busy |= {p, c}
            used.add(c)
            cost += c_cost
            if len(nodes) + len(splits) == N:
                break
        if not splits:
            return None, math.inf
        nodes += [c for _, c in splits]
        rounds.append(splits)
    cost -= g.edges[rounds[0][0]]["err"] if rounds else 0  # the first split is 1 CZ
    return rounds, cost


def _cheap_connected_sets(g: nx.Graph, N: int, node_cost: dict, cz_per_edge: float, keep: int):
    """Beam search over connected qubit sets; cost = node costs + cheapest attaching edge."""
    beam = max(16, 4000 // N)
    frontier = sorted(((node_cost[q], frozenset([q])) for q in g.nodes), key=lambda t: t[0])[:beam]
    for _ in range(N - 1):
        grown = {}
        for cost, S in frontier:
            for v in {u for s in S for u in g.neighbors(s)} - S:
                e = min(g.edges[v, u]["err"] for u in g.neighbors(v) if u in S)
                c2 = cost + node_cost[v] + cz_per_edge * e
                S2 = S | {v}
                if c2 < grown.get(S2, math.inf):
                    grown[S2] = c2
        if not grown:
            return []
        frontier = sorted(((c, S) for S, c in grown.items()), key=lambda t: t[0])[:beam]
    return [S for _, S in frontier[:keep]]


def _best_w_tree(g, N, node_cost, zero_errors, one_errors):
    """Pick a cheap qubit set first, then the best broadcast tree inside it."""
    best, best_cost = None, math.inf
    for S in _cheap_connected_sets(g, N, node_cost, cz_per_edge=2, keep=8):
        sub = g.subgraph(S)
        roots = sorted(S, key=lambda q: max(nx.single_source_shortest_path_length(sub, q).values()))
        for root in roots[:12]:
            rounds, cost = _grow_tree(sub, root, N, node_cost)
            if rounds is None:
                continue
            cost += len(rounds) * _ROUND_NS * _IDLE_RATE_PER_NS
            lam = _readout_lambda(list(S), zero_errors, one_errors, 1, math.exp(-cost))
            cost += math.log1p(lam / (N - 1))
            if cost < best_cost:
                best, best_cost = (root, rounds), cost
    return best


def _w_tree_circuit(root, rounds: list) -> cirq.Circuit:
    """Amplitude starts on the root; a split hands a child the share of its subtree."""
    children = {}
    for r in rounds:
        for p, c in r:
            children.setdefault(p, []).append(c)

    size = {}

    def subtree(v):
        size[v] = 1 + sum(subtree(c) for c in children.get(v, []))
        return size[v]

    remaining = {root: subtree(root)}  # amplitude^2 (in units of 1/N) still held by v
    moments = []
    for i, r in enumerate(rounds):
        ops = []
        for p, c in r:
            theta = math.asin(math.sqrt(size[c] / remaining[p]))
            remaining[p] -= size[c]
            remaining[c] = size[c]
            if i == 0:  # product input: Ry, CNOT, X gives cos|10> + sin|01> with 1 CZ
                ops += [cirq.ry(2 * theta).on(c), cirq.CNOT(c, p), cirq.X(p)]
            else:  # cirq.givens(-t): |10> -> cos t |10> + sin t |01>
                ops.append(cirq.givens(-theta).on(p, c))
        moments.append(cirq.Circuit(ops))
    return sum(moments, cirq.Circuit())


# ---- Baertschi-Eidenbenz logical circuit ------------------------------------------------


def _ccry(phi: float, c1, c2, t) -> list:
    """Doubly-controlled Ry(phi) with 4 CNOTs: X Ry(a) X = Ry(-a) makes the Ry's cancel
    unless both controls are 1, in which case they add to Ry(4a) = Ry(phi)."""
    a = phi / 4
    return [
        cirq.ry(a).on(t), cirq.CNOT(c1, t),
        cirq.ry(-a).on(t), cirq.CNOT(c2, t),
        cirq.ry(a).on(t), cirq.CNOT(c1, t),
        cirq.ry(-a).on(t), cirq.CNOT(c2, t),
    ]


def _scs(q: list, m: int, k: int) -> list:
    """Split & Cyclic Shift SCS_{m,k} on the 1-indexed logical qubits q[1..m] (paper 2.2)."""
    ops = []
    # gate (i) on (m-1, m)
    a, b = q[m - 2], q[m - 1]
    ops += [cirq.CNOT(a, b), cirq.ry(2 * math.acos(math.sqrt(1 / m))).on(a).controlled_by(b), cirq.CNOT(a, b)]
    # gates (ii)_l on (m-l, m-l+1, m)
    for l in range(2, k + 1):
        t, c = q[m - l - 1], q[m - l]
        ops += [cirq.CNOT(t, b)]
        ops += _ccry(2 * math.acos(math.sqrt(l / m)), b, c, t)
        ops += [cirq.CNOT(t, b)]
    return ops


def _bartschi_eidenbenz(N: int, k: int) -> tuple[cirq.Circuit, list]:
    """U_{N,k} |0^{N-k} 1^k> = |D^N_k> on LineQubits."""
    q = cirq.LineQubit.range(N)
    ops = [cirq.X(q[i]) for i in range(N - k, N)]
    for m in range(N, k, -1):
        ops += _scs(q, m, k)
    for m in range(k, 1, -1):
        ops += _scs(q, m, m - 1)
    return cirq.Circuit(ops), q


# ---- compilation helpers -----------------------------------------------------------------


def _to_cz_phxz(circuit: cirq.Circuit) -> cirq.Circuit:
    """Compile to cirq.CZ + PhasedXZGate and drop global phases."""
    c = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    c = cirq.Circuit(
        cirq.Moment(op for op in m if not isinstance(op.gate, cirq.GlobalPhaseGate)) for m in c
    )
    c = cirq.merge_single_qubit_gates_to_phxz(c)
    return cirq.drop_empty_moments(cirq.drop_negligible_operations(c))


def _phxz(op: cirq.Operation) -> cirq.Operation:
    return cirq.PhasedXZGate.from_matrix(cirq.unitary(op)).on(*op.qubits)


# ---- entry point ------------------------------------------------------------------------


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[
        tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]
    ],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.
        cz_pauli_error_rate: Measured CZ Pauli error rates and their statistical uncertainties.
        sq_pauli_error_rate: Measured single-qubit Pauli error rates and their statistical uncertainties.
        zero_errors: Measured parallel 0 -> 1 readout error rates and their statistical uncertainties.
        one_errors: Measured 1 -> 0 readout error rates and their statistical uncertainties.

    Returns:
        A Cirq circuit to produce Dicke state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested (N, n) combination is not implemented.
    """
    return build(
        num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
        zero_errors, one_errors,
    )


def build(
    num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
    zero_errors, one_errors, *, w_tree: bool = True,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """`create_dicke_state` with switches for the experiments (defaults = submission)."""
    N, n = int(num_qubits), int(num_excitations)
    if N < 1 or not 0 <= n <= N:
        raise NotImplementedError(f"invalid (N, n) = ({N}, {n})")
    flip = n > N - n  # D^N_n = X^N D^N_{N-n}: prepare the lighter one
    k = N - n if flip else n
    node_cost = {
        q: _qubit_cost(q, sq_pauli_error_rate, zero_errors, one_errors, n / N)
        for q in sq_pauli_error_rate
    }

    if k == 0:  # product state: |0...0> or |1...1>
        qubits = sorted(node_cost, key=node_cost.get)[:N]
        if len(qubits) < N:
            raise NotImplementedError("not enough calibrated qubits")
        ops = [cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0).on(q) for q in qubits] if flip else []
        return cirq.Circuit(ops), set(qubits)

    g = _usable_graph(cz_pauli_error_rate, sq_pauli_error_rate)

    if k == 1 and w_tree:
        tree_cost = {  # a tree node receives ~4 non-virtual single-qubit gates
            q: _qubit_cost(q, sq_pauli_error_rate, zero_errors, one_errors, n / N, n_sq=4)
            for q in sq_pauli_error_rate
        }
        tree = _best_w_tree(g, N, tree_cost, zero_errors, one_errors)
        if tree is not None:
            root, rounds = tree
            logical = _w_tree_circuit(root, rounds)
            qubits = {root} | {c for r in rounds for _, c in r}
            if flip:
                logical.append(cirq.X.on_each(*sorted(qubits)))
            return _to_cz_phxz(logical), qubits

    layout = _best_path(g, N, node_cost) or _connected_patch(g, N, node_cost)
    if layout is None:
        raise NotImplementedError("no connected layout of the requested size")

    logical, lq = _bartschi_eidenbenz(N, k)
    if flip:
        logical.append(cirq.X.on_each(*lq))
    logical = _to_cz_phxz(logical)

    device = g.subgraph(layout).copy()
    router = cirq.RouteCQC(device)
    initial = cirq.HardCodedInitialMapper(dict(zip(lq, layout)))
    routed, initial_map, swap_map = router.route_circuit(logical, initial_mapper=initial)
    circuit = _to_cz_phxz(routed)
    final_qubits = {swap_map[initial_map[q]] for q in lq}
    return circuit, final_qubits
