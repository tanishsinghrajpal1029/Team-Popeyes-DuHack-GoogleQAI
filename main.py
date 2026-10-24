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
import time

import cirq
import networkx as nx
import numpy as np


# Only attempt the second routing variant if the first finished within this many seconds,
# so the 60 s build limit holds with a wide margin on slower machines.
_SECOND_TRY_BUDGET_S = 10.0


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


def _best_path(g: nx.Graph, length: int, node_cost: dict, beam: int = 256,
               edge_weight: float = 1.0, keep: int = 1):
    """Beam search for a simple path of `length` qubits minimizing
    edge_weight * (CZ error) + node cost. Returns the best path, or the best `keep`."""
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
                step = edge_weight * g.edges[path[-1], nb]["err"] + node_cost[nb]
                grown.append((cost + step, path + [nb]))
        if not grown:
            return None
        grown.sort(key=lambda t: t[0])
        frontier = grown[:beam]
    return frontier[0][1] if keep == 1 else frontier[:keep]


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
    d = np.array([one_errors.get(q, (0.5, 0))[0] - zero_errors.get(q, (0.5, 0))[0] for q in qubits])
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


# ---- small cases: numerically synthesized exact circuits ----------------------------------
#
# For small (N, k) we embed exact Ry + CZ circuits found offline by experiments/synth.py
# (gradient optimization to infidelity < 1e-13, then greedy CZ pruning). They depend only
# on a layout shape (grid coordinates), never on calibration data; the calibration-dependent
# part, where to put the shape, is decided below at build time. Synthesis itself takes
# minutes per case, far over the 60 s build budget, which is why the result is cached here.

_GRID_SYMMETRIES = (
    lambda r, c: (r, c), lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (-r, -c),
    lambda r, c: (c, r), lambda r, c: (c, -r), lambda r, c: (-c, r), lambda r, c: (-c, -r),
)


def _place_synth(g, entry, n, sq_pauli_error_rate, zero_errors, one_errors):
    """Best placement of a synthesized shape on calibrated qubits and couplers."""
    coords = entry["coords"]
    N = len(coords)
    th = np.array(entry["params"]).reshape(-1, N)
    ry_count = (np.abs(np.sin(th / 2)) > 1e-9).sum(axis=0)  # non-trivial rotations per qubit
    uses = {}
    for layer in entry["cz_layers"]:
        for e in layer:
            uses[tuple(e)] = uses.get(tuple(e), 0) + 1
    usable = set(g.nodes)
    best, best_cost = None, math.inf
    for sym in _GRID_SYMMETRIES:
        pts = [sym(r, c) for r, c in coords]
        r0, c0 = pts[0]
        for anchor in usable:
            qs = [cirq.GridQubit(anchor.row + r - r0, anchor.col + c - c0) for r, c in pts]
            if not all(q in usable for q in qs) or not all(g.has_edge(qs[i], qs[j]) for i, j in uses):
                continue
            cost = sum(u * g.edges[qs[i], qs[j]]["err"] for (i, j), u in uses.items())
            cost += sum(
                _qubit_cost(q, sq_pauli_error_rate, zero_errors, one_errors, n / N, n_sq=ry_count[i])
                for i, q in enumerate(qs)
            )
            lam = _readout_lambda(qs, zero_errors, one_errors, n, math.exp(-cost))
            cost += math.log1p(lam / max(math.comb(N, n) - 1, 1))
            if cost < best_cost:
                best, best_cost = qs, cost
    return best


def _synth_circuit(entry, qs) -> cirq.Circuit:
    N = len(qs)
    th = np.array(entry["params"]).reshape(-1, N)
    moments = []
    for l in range(len(th)):
        moments.append(cirq.Moment(cirq.ry(th[l, i]).on(qs[i]) for i in range(N)))
        if l < len(entry["cz_layers"]):
            moments.append(cirq.Moment(cirq.CZ(qs[i], qs[j]) for i, j in entry["cz_layers"][l]))
    return cirq.Circuit(moments)


# BEGIN SYNTH TABLE (generated by experiments/synth.py --export; do not edit)
_SYNTH = {
    (4, 2): dict(  # 5 CZ, ladder, seed 0
        coords=[(0, 0), (1, 0), (0, 1), (1, 1)],
        cz_layers=[[(2, 3)], [(0, 2), (1, 3)], [(0, 1), (2, 3)]],
        params=[-2.4121959773832304, -0.19449143864768134, 1.9811211708815972, 0.813964899962581, -2.3001930008562295, -1.3763048881299909, 3.6840084710212055, -1.6347988157547007, -1.8677748517379763e-09, 1.6459326957102467e-09, 1.5707963271470289, -1.5707963258105022, 3.141592654529376, -1.570796328472846, 2.5770715684959696e-09, 1.570796325412249],
    ),
    (5, 2): dict(  # 10 CZ, path, seed 0
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        cz_layers=[[(0, 1), (2, 3)], [(1, 2)], [(2, 3)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1)]],
        params=[0.6183268576904242, 4.492265691117577, -0.8209378413355491, 0.6574833346606679, -0.6939678699158748, -0.43206140822478645, -3.1185870481778513, 3.1869665948668033, -1.983118445993127, 0.8613739578281225, 0.12409010769983507, -2.934922120891745, -2.229854361923873, -2.690058891780023, 1.3996791502766723, -0.3463281577298319, -2.3929465700149395, 2.617993879617297, -3.6052402650923736, 0.003711092141899059, -4.659959297015562, 0.6154797074967451, 2.589272463940698e-09, 1.570796317632, 0.0329888580686312, -0.25885703062599014, 0.7853981619369151, 1.5707963202639785, -1.570796331127954, -1.6037851879018736, -2.0973374610153246, -1.570796322384344, 3.141592661916084, 2.2535503248077753e-08, -1.1061837893983615, -3.1415926646213137, 1.5707963192449865, -1.570796314178292, -1.282448586953362e-08, 1.1061837826845407],
    ),
    (6, 2): dict(  # 18 CZ, path, seed 0
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        cz_layers=[[(0, 1), (2, 3)], [(1, 2), (3, 4)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(0, 1), (2, 3)]],
        params=[-1.9337932255602202, 2.650215813244823, 1.0492421095569429, 1.966635779192254, 2.242817164651964, 0.39971805956555817, -2.0052506379994646, -1.920110291355524, 0.5910060105573643, -5.6528891690880405, 2.4695718231542476, -2.5162185471522123, 2.080526756530937, 1.1824497076580225, -1.6267632110877428, -0.48213158527426153, -8.671601293822655e-08, 0.5457042019892089, -0.4796655976500496, -0.5064446397756512, -2.1028320474366398, -4.2302573737922, 0.9553164890492081, -6.041705363656901, -3.430870176364036, -3.926990793557645, -2.553590043258465, 2.7344400970610088, 1.5707962685773413, -1.4384841492343092, 1.7017603027682535, -0.7853981697435274, -1.5707963150740005, 2.8198421155105406, -2.472628604053983, 1.6505724968326048, 3.0540167861939387, 3.851565811625154, 1.419742968176338, 0.32175063315616664, -1.5707963198923012, -2.350568753450995, -2.7850075411558457, 1.3860565947744303, -2.675094574257822, -3.1415926649076367, -1.5707961113501914, -0.3370340820232944, 2.741619433291046, -0.525233427548667, -3.608090726612401, 3.1415926505021523, 0.7446906940902297, -0.943700609172635, 1.149633759193378e-08, -1.5707963135229008, 0.15105336009150372, -1.5707963190356238, -2.315487117418924, -0.2900616861350798],
    ),
    (6, 3): dict(  # 15 CZ, ladder, seed 0
        coords=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        cz_layers=[[(0, 1), (2, 3), (4, 5)], [(0, 2), (1, 3)], [(3, 5)], [(2, 3)], [(0, 2), (1, 3)], [(2, 4), (3, 5)], [(0, 1), (2, 3)], [(0, 2), (1, 3)]],
        params=[0.7851664747438014, -1.7710156329330253, -2.1185388505219698, -1.8671133966187958, -1.5707963275045767, 1.5707963256046655, 1.179027566593969, 0.5604597447908155, -2.079661915538599, 2.0877329669549036, -2.459430144929798, -1.1908031396858503, 2.353694854176512, -2.8723578127057365, 2.4947967041170873, -2.6179938716603455, -0.16225239605727232, 2.9702705943893166, -3.0798344420293646, -0.8015105929607077, 0.031316237377117216, 8.456895202595918e-09, 1.4670255092459061, -2.0445397170983315e-09, 2.296935928754977, -4.180113230291891, 5.667705597872043, -1.570796326356758, -1.6021361104209528, 1.065986195764983, 1.3170388159091704, -0.013898220187408068, -3.926990795306432, -1.5707963170027488, -1.7469247107956127, 2.0756064583417775, -0.5316406507884522, 1.5846945494841258, 4.71238897189064, -6.156903272848264e-09, 3.1415926504475498, 6.601882229324128e-09, 1.5707963272468566, -2.868178657040069e-09, -8.888359152269209e-09, -1.5707963217006877, -0.9230877005638871, -0.3167438770138414, -1.5707963403380412, -1.5707963490341266, -3.1415926405387524, 3.141592671192763, 0.9230877046026618, -1.2540524601828866],
    ),
}
# END SYNTH TABLE


# ---- larger cases: B-E from synthesized nearest-neighbour SCS blocks ----------------------
#
# SCS_{m,k} acts on the k+1 consecutive line qubits m-k .. m, and its input is always a
# sorted string |0^{w-l} 1^l> on that window. experiments/scs_synth.py finds exact Ry + CZ
# circuits for these small isometries on a path, so the whole B-E circuit becomes
# nearest-neighbour on a line: no routing. Keys are (m, k, all_ones); all_ones marks the
# second-group blocks SCS_{m,m-1}, m <= K, whose window can hold |1^m>.


def _scs_blocks(N: int, K: int) -> list[tuple[int, int, bool]]:
    return [(m, K, False) for m in range(N, K, -1)] + [(m, m - 1, True) for m in range(K, 1, -1)]


def _scs_cz_count(N: int, K: int) -> int | None:
    keys = _scs_blocks(N, K)
    if not all(key in _SCS for key in keys):
        return None
    return sum(sum(map(len, _SCS[key][0])) for key in keys)


def _scs_circuit(N: int, K: int, path: list) -> cirq.Circuit:
    """U_{N,K} |0^{N-K} 1^K> = |D^N_K> on `path` (path[0] is B-E qubit 1).

    Operations are appended in order with cirq's EARLIEST insertion, so consecutive
    blocks overlap in time wherever their qubits allow (the paper's "pushed stairs").
    """
    ops = [cirq.X(q) for q in path[N - K:]]
    for m, k, all_ones in _scs_blocks(N, K):
        layers, params = _SCS[(m, k, all_ones)]
        window = path[m - k - 1:m]
        th = np.array(params).reshape(-1, k + 1)
        for l in range(len(th)):
            ops += [cirq.ry(th[l, i]).on(window[i]) for i in range(k + 1)]
            if l < len(layers):
                ops += [cirq.CZ(window[i], window[j]) for i, j in layers[l]]
    return cirq.Circuit(ops)


# BEGIN SCS TABLE (generated by experiments/scs_synth.py --export; do not edit)
_SCS = {}
# END SCS TABLE


# ---- compilation helpers -----------------------------------------------------------------


def _to_cz_phxz(circuit: cirq.Circuit) -> cirq.Circuit:
    """Compile to cirq.CZ + PhasedXZGate and drop global phases."""
    # CZTargetGateset already merges single-qubit runs into PhasedXZ and drops negligible ops
    c = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    c = cirq.drop_empty_moments(cirq.Circuit(
        cirq.Moment(op for op in m if not isinstance(op.gate, cirq.GlobalPhaseGate)) for m in c
    ))
    # hardware-friendly layering: every moment is either all CZ or all single-qubit
    if any(any(op.gate == cirq.CZ for op in m) and any(len(op.qubits) == 1 for op in m) for m in c):
        c = cirq.stratified_circuit(c, categories=[lambda op: op.gate == cirq.CZ])
    return c


# ---- dynamical decoupling --------------------------------------------------------------------

_X = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
_Y = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0.5)


def _is_virtual_z(op: cirq.Operation) -> bool:
    if not isinstance(op.gate, cirq.PhasedXZGate):
        return False
    x = op.gate.x_exponent % 2
    return min(x, 2 - x) < 1e-9


def _add_dd(circuit: cirq.Circuit) -> cirq.Circuit:
    """Echo idle windows of qubits that may be in superposition.

    A qubit is protected from its first non-virtual single-qubit gate (before that it sits
    in |0>, which dephasing cannot touch) to the end of the circuit, including the trailing
    window before measurement. Pulses go only into single-qubit moments. Windows with >= 4
    free slots get X-Y-X-Y, with 2-3 slots X-X; both multiply to identity up to a global
    phase, so the noiseless state is unchanged.
    """
    moments = [list(m) for m in circuit]
    one_q_layer = [all(len(op.qubits) == 1 for op in m) for m in circuit]
    for q in sorted(circuit.all_qubits()):
        busy = [any(q in op.qubits for op in m) for m in circuit]
        active = [i for i, m in enumerate(circuit) for op in m
                  if q in op.qubits and len(op.qubits) == 1 and not _is_virtual_z(op)]
        if not active:
            continue
        i = active[0] + 1
        while i < len(circuit):
            if busy[i]:
                i += 1
                continue
            j = i
            while j < len(circuit) and not busy[j]:
                j += 1
            slots = [t for t in range(i, j) if one_q_layer[t]]
            seq = [_X, _Y, _X, _Y] if len(slots) >= 4 else [_X, _X] if len(slots) >= 2 else []
            for s, gate in enumerate(seq):  # evenly spaced over the window
                moments[slots[round((s + 0.5) * len(slots) / len(seq) - 0.5)]].append(gate.on(q))
            i = j
    return cirq.Circuit(cirq.Moment(m) for m in moments)


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
    zero_errors, one_errors, *, w_tree: bool = True, synth: bool = True,
    scs: bool = True, dd: bool = False,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """`create_dicke_state` with switches for the experiments (defaults = submission)."""
    circuit, qubits = _build_core(
        num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
        zero_errors, one_errors, w_tree=w_tree, synth=synth, scs=scs,
    )
    return (_add_dd(circuit) if dd else circuit), qubits


def _build_core(
    num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
    zero_errors, one_errors, *, w_tree: bool, synth: bool, scs: bool,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
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

    scs_cz = _scs_cz_count(N, k) if scs else None
    synth_entry = _SYNTH.get((N, k)) if synth else None
    if synth_entry is not None and (scs_cz is None or sum(map(len, synth_entry["cz_layers"])) <= scs_cz):
        qs = _place_synth(g, synth_entry, n, sq_pauli_error_rate, zero_errors, one_errors)
        if qs is not None:
            logical = _synth_circuit(synth_entry, qs)
            if flip:
                logical.append(cirq.X.on_each(*qs))
            return _to_cz_phxz(logical), set(qs)

    if scs_cz is not None:
        paths = _best_path(g, N, node_cost, edge_weight=scs_cz / (N - 1), keep=16)
        if paths:
            K = math.comb(N, n)

            def total(cost_path):
                cost, path = cost_path
                lam = _readout_lambda(path, zero_errors, one_errors, n, math.exp(-cost))
                return cost + math.log1p(lam / (K - 1))

            path = min(paths, key=total)[1]
            logical = _scs_circuit(N, k, path)
            if flip:
                logical.append(cirq.X.on_each(*path))
            return _to_cz_phxz(logical), set(path)

    layout = _best_path(g, N, node_cost) or _connected_patch(g, N, node_cost)
    if layout is None:
        raise NotImplementedError("no connected layout of the requested size")

    logical, lq = _bartschi_eidenbenz(N, k)
    if flip:
        logical.append(cirq.X.on_each(*lq))

    # Routing is chaotic: routing the raw CNOT/CRy circuit or a CZ-compiled one wins on
    # different calibrations (by up to ~30% CZ). Try both while time allows; keep the
    # one with fewer CZs. The raw variant goes first because it is the faster one.
    device = g.subgraph(layout).copy()
    start = time.monotonic()
    best = None
    for precompile in (False, True):
        if best is not None and time.monotonic() - start > _SECOND_TRY_BUDGET_S:
            break
        router = cirq.RouteCQC(device)
        initial = cirq.HardCodedInitialMapper(dict(zip(lq, layout)))
        source = _to_cz_phxz(logical) if precompile else logical
        routed, initial_map, swap_map = router.route_circuit(source, initial_mapper=initial)
        circuit = _to_cz_phxz(routed)
        key = (sum(op.gate == cirq.CZ for op in circuit.all_operations()), len(circuit))
        if best is None or key < best[0]:
            best = (key, circuit, {swap_map[initial_map[q]] for q in lq})
    return best[1], best[2]
