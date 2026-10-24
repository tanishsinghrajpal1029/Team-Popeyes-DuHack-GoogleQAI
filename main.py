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


def _qubit_cost(q, sq_pauli_error_rate, zero_errors, one_errors, p_one: float) -> float:
    """Expected per-qubit error: readout weighted by how often the qubit reads 1."""
    e0 = zero_errors.get(q, (0.5, 0))[0]
    e1 = one_errors.get(q, (0.5, 0))[0]
    return (1 - p_one) * e0 + p_one * e1 + sq_pauli_error_rate[q][0]


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
