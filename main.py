"""Dicke-state preparation on Google Willow (DuQuantum x Google Quantum AI, standard track).

Only cirq, cirq_google, their dependencies (numpy, networkx) and the standard library are
imported, as the challenge requires.

Construction (baseline): Baertschi & Eidenbenz, "Deterministic Preparation of Dicke
States" (arXiv:1904.07358). We build the logical circuit, place it on a low-error path of
calibrated qubits, route it with cirq.RouteCQC, and compile to CZ + PhasedXZ. The Dicke
state is permutation-symmetric, so wherever routing leaves the logical qubits is a valid
answer; we just return that set.
"""

import heapq
import math
import time

import cirq
import cirq_google
import networkx as nx
import numpy as np


# Only attempt the second routing variant if the first finished within this many seconds,
# so the 60 s build limit holds with a wide margin on slower machines.
_SECOND_TRY_BUDGET_S = 10.0


# ---- calibration graph ----------------------------------------------------------------


def _robust(v) -> float:
    """Reported error rate plus one standard uncertainty, used for every placement decision.

    Picking the lowest reported rates favours qubits that only look good by chance (their
    true error regresses toward the median). Ranking by value + 1 sigma hedges that; with
    the notebook's uniform uncertainties it shifts all costs equally and changes nothing.
    Accepts (value, sigma) or a bare value.
    """
    try:
        return float(v[0]) + max(float(v[1]), 0.0)
    except (TypeError, IndexError):
        return float(v)


def _usable_graph(cz_pauli_error_rate: dict, sq_pauli_error_rate: dict) -> nx.Graph:
    """Calibrated couplers whose endpoints both have single-qubit calibrations."""
    g = nx.Graph()
    for pair, v in cz_pauli_error_rate.items():
        a, b = tuple(pair)  # keys are tuple(frozenset): order is arbitrary
        if a in sq_pauli_error_rate and b in sq_pauli_error_rate:
            g.add_edge(a, b, err=_robust(v))
    return g


def _qubit_cost(q, sq_pauli_error_rate, zero_errors, one_errors, p_one: float, n_sq: float = 1) -> float:
    """Expected per-qubit error: readout weighted by how often the qubit reads 1, plus
    `n_sq` single-qubit gates."""
    e0 = _robust(zero_errors.get(q, (0.5, 0)))
    e1 = _robust(one_errors.get(q, (0.5, 0)))
    return (1 - p_one) * e0 + p_one * e1 + n_sq * _robust(sq_pauli_error_rate[q])


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


# ---- tiny W states (K <= 3): choose qubits by predicted expected score --------------------
#
# For K <= 3 the judged score 1/(z + x) is so heavy-tailed that its expectation is set by
# the X-term floor x and the imbalance lambda, not by CZ count. With at most 3 qubits we can
# afford to simulate every placement exactly (gate depolarizing + readout, from the
# calibration dicts) and pick the one with the highest predicted E[score].

_PAULIS = [np.array(m, dtype=complex) for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]])]


def _embed(mats: dict, N: int) -> np.ndarray:
    """Kronecker product placing 2x2 `mats[i]` on logical qubit i (qubit 0 most significant)."""
    out = np.array([[1.0 + 0j]])
    for i in range(N):
        out = np.kron(out, mats.get(i, np.eye(2)))
    return out


def _tiny_program(N: int, rounds: list, flip: bool):
    """Compile one tree shape on LineQubits once; return it plus a noisy-simulation program."""
    lq = cirq.LineQubit.range(N)
    logical = _w_tree_circuit(lq[rounds[0][0][0]], [[(lq[p], lq[c]) for p, c in r] for r in rounds])
    if flip:
        logical.append(cirq.X.on_each(*lq))
    compiled = _to_cz_phxz(logical)
    steps = []
    for op in compiled.all_operations():
        idx = tuple(q.x for q in op.qubits)
        u = cirq.Circuit(op).unitary(qubit_order=lq)
        if op.gate == cirq.CZ:
            paulis = [_embed({idx[0]: a, idx[1]: b}, N) for a in [np.eye(2)] + _PAULIS
                      for b in [np.eye(2)] + _PAULIS][1:]
            steps.append((u, "cz", idx, paulis))
        elif not _is_virtual_z(op):
            steps.append((u, "sq", idx, [_embed({idx[0]: p}, N) for p in _PAULIS]))
        else:
            steps.append((u, None, idx, []))
    return compiled, lq, steps


def _expected_score(K: int, f: float, lam: float, x: float, exact: bool) -> float:
    """E[1/(z + x)] with z = chi^2/(K f), chi^2 ~ noncentral chi^2(K-1, lam).

    exact=False uses closed forms for lam = 0 (fast ranking); exact=True integrates."""
    from scipy import integrate, special, stats

    a = K * f * x  # E[score] = K f E[1/(chi^2 + a)]
    if not exact:
        if K == 2:
            return K * f * math.sqrt(math.pi / (2 * a)) * special.erfcx(math.sqrt(a / 2))
        return K * f * 0.5 * math.exp(a / 2) * special.exp1(a / 2)
    dist = stats.ncx2(K - 1, lam) if lam > 1e-12 else stats.chi2(K - 1)
    t_max = math.sqrt(dist.isf(1e-14))
    val, _ = integrate.quad(lambda t: dist.pdf(t * t) * 2 * t / (t * t + a), 0, t_max,
                            points=[min(math.sqrt(a), t_max / 2)], limit=500)
    return K * f * val


def _predict_tiny(steps, phys, n, cz_err, sq_err, zero_errors, one_errors, exact, shots=10_000):
    N = len(phys)
    rho = np.zeros((2**N, 2**N), dtype=complex)
    rho[0, 0] = 1
    for u, kind, idx, paulis in steps:
        rho = u @ rho @ u.conj().T
        if kind is None:
            continue
        p = cz_err[frozenset(phys[i] for i in idx)] if kind == "cz" else sq_err[phys[idx[0]]]
        rho = (1 - p) * rho + (p / len(paulis)) * sum(P @ rho @ P for P in paulis)
    h = _embed({i: np.array([[1, 1], [1, -1]]) / math.sqrt(2) for i in range(N)}, N)
    probs = []
    for r in (rho, h @ rho @ h):
        pr = np.real(np.diag(r)).reshape((2,) * N)
        for j, q in enumerate(phys):  # readout confusion, [read, true]
            e0, e1 = _robust(zero_errors.get(q, (0.5, 0))), _robust(one_errors.get(q, (0.5, 0)))
            conf = np.array([[1 - e0, e1], [e0, 1 - e1]])
            pr = np.moveaxis(np.tensordot(conf, pr, axes=([1], [j])), 0, j)
        probs.append(pr.reshape(-1))
    pz, px = probs
    weight = np.array([bin(i).count("1") for i in range(2**N)])
    pv = pz[weight == n]
    f = float(pv.sum())
    lam = shots * f * float(np.mean((pv / pv.mean() - 1) ** 2))
    qw = np.bincount(weight, weights=px, minlength=N + 1)
    x = 0.0
    for k in range(1, N + 1):  # mean X term, including its shot-noise floor
        v = np.array([sum((-1) ** j * math.comb(w, j) * math.comb(N - w, k - j) for j in range(k + 1))
                      for w in range(N + 1)]) / math.comb(N, k)
        m = float(qw @ v)
        var = float(qw @ v**2) - m * m
        ideal = (math.comb(k, k // 2) * math.comb(N - k, n - k // 2) / math.comb(N, n)
                 if k % 2 == 0 and 0 <= n - k // 2 <= N - k else 0.0)
        x += ((m - ideal) ** 2 + var / shots) / (ideal**2 if ideal else 1.0)
    return _expected_score(math.comb(N, n), f, lam, x, exact)


def _tiny_candidates(g: nx.Graph, N: int):
    """(logical rounds, physical qubits by logical index) for every placement of N <= 3."""
    if N == 2:
        for a, b in g.edges:
            yield [[(0, 1)]], [a, b]
            yield [[(0, 1)]], [b, a]
        return
    for mid in g.nodes:  # N == 3: every path a - mid - c
        nbrs = sorted(g.neighbors(mid))
        for i, a in enumerate(nbrs):
            for c in nbrs[i + 1:]:
                yield [[(0, 1)], [(0, 2)]], [mid, a, c]  # star from the middle, either order
                yield [[(0, 1)], [(0, 2)]], [mid, c, a]
                yield [[(0, 1)], [(1, 2)]], [a, mid, c]  # chain from either end
                yield [[(0, 1)], [(1, 2)]], [c, mid, a]


def _best_tiny_w(g, N, n, flip, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
    cz_err = {frozenset(p): _robust(v) for p, v in cz_pauli_error_rate.items()}
    sq_err = {q: _robust(v) for q, v in sq_pauli_error_rate.items()}
    programs, scored = {}, []
    for rounds, phys in _tiny_candidates(g, N):
        key = str(rounds)
        if key not in programs:
            programs[key] = _tiny_program(N, rounds, flip)
        e = _predict_tiny(programs[key][2], phys, n, cz_err, sq_err, zero_errors, one_errors, exact=False)
        scored.append((e, key, phys))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best = max(scored[:10], key=lambda t: _predict_tiny(
        programs[t[1]][2], t[2], n, cz_err, sq_err, zero_errors, one_errors, exact=True))
    compiled, lq, _ = programs[best[1]]
    mapping = dict(zip(lq, best[2]))
    return compiled.transform_qubits(lambda q: mapping[q]), set(best[2])


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
    (4, 2): dict(  # 5 CZ, ladder, seed 0, robust pick 18
        coords=[(0, 0), (1, 0), (0, 1), (1, 1)],
        cz_layers=[[(2, 3)], [(0, 2), (1, 3)], [(0, 1), (2, 3)]],
        params=[1.011641720900928, 2.2789188858281637, -1.3076572937407227, 2.3795057930799706, 3.70074725852797, -0.7081225567239268, -1.6779636626810785, -0.8638085259288678, 1.7818695622763414e-09, 5.417358508585396e-10, 1.5707963256719608, -1.570796329932613, 1.232338693447e-09, -1.5707963271329526, -3.1415926551997733, 4.712388979190364],
    ),
    (5, 2): dict(  # 10 CZ, path, seed 0, robust pick 21
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        cz_layers=[[(0, 1), (2, 3)], [(1, 2)], [(2, 3)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1)]],
        params=[-1.4986414490951898, 0.6030549441059361, -1.694517047389452, 0.46750258492084046, -0.8868353061110757, 0.9118787801784288, 1.27621343166358, 1.0501511920838598, 0.038332717492985925, 4.127597365561725, -4.15047590148269, -0.37135563950150685, 0.9117382703191945, -1.0233121926372912, 4.207942601792098, -4.17328935685691, -0.5839610094147459, -2.6179938817221795, -0.46364762627110256, 0.4052769752256716, -2.8563973746921527, -0.6154797200760713, 7.31177774405034e-10, -1.5707963198525992, 2.468324523175826, 0.5451618742927558, -0.7853981653743284, 1.5707963246198553, -0.8492082787191461, 1.9359403955115857, 0.2402362793754006, 1.5707963323593293, -1.152017897434777, -3.531489246951883, 1.0362492938856247, -3.426946963739398e-09, 1.5707963252379087, 2.7228142396304356, -0.3316914655960666, -0.7281252343604439],
    ),
    (6, 2): dict(  # 18 CZ, path, seed 0, robust pick 19
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        cz_layers=[[(0, 1), (2, 3)], [(1, 2), (3, 4)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(0, 1), (2, 3)]],
        params=[1.273401929989079, -2.6629953412693688, -1.1939606465323103, 4.176766894705059, -1.7829991268996401, -1.0753845379217315, 0.3912099294907602, 1.2573895570428093, -0.9141970221022485, -0.7894541023186473, 0.2122028170155825, -2.988012893451692, -0.24130623722819666, -1.959929601761315, -0.45178167614039333, 1.2833688120374722, -1.2012224064022992e-07, -0.6489914531278403, 2.8314469771168516, 3.6471817311943973, 1.039931446636215, 0.2874275436518202, -4.092375402160591, -0.37280878090051606, 2.6832296672128617, 2.3571525770817834, -2.5534368387727473, -0.8210874460269298, 1.5707962714355483, 2.315713695205924, -0.17159733097131105, 2.3571527864420037, 1.5707963170140482, 2.820087997027582, -0.6679354206263197, 0.2915723909618664, 0.3459405490564516, -3.1886386324184737, -5.083238801751641e-06, -0.32502165430141255, -1.5707963269518088, 0.9194467459990205, 0.9131317817015843, 0.9289964056245835, 1.918517887380368, 2.4560572830361436e-08, 1.563735176654994, -1.4608111378352324, 0.48332093592281383, -2.452746759761478, 1.2250008480283063, -0.0019275463714309416, -2.6380347126710237, 3.2230017455792863, -1.296261642185438e-08, 1.5707963165289673, 4.712394058952568, -1.570796364013428, 1.067238622003698, 2.9501983416037136],
    ),
    (6, 3): dict(  # 15 CZ, ladder, seed 0, robust pick 11
        coords=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        cz_layers=[[(0, 1), (2, 3), (4, 5)], [(0, 2), (1, 3)], [(3, 5)], [(2, 3)], [(0, 2), (1, 3)], [(2, 4), (3, 5)], [(0, 1), (2, 3)], [(0, 2), (1, 3)]],
        params=[-1.6833327905971032, -3.9131196713627574, 4.341168673558379, -1.0678724970825328, 4.712388982885306, -4.712388975292479, 0.43163538346296787, 1.1809666058926551, 1.6346909844915862, 1.5114228726759555, 4.21962563962444, 1.3749485264472727, 1.547568542350625, -1.9420186765959466, -1.6806071783873842, 0.5235987841617172, 5.524967593479281, 0.22694055202490632, 5.791065438109848, 2.094040365347798, -2.0764651884616643, -3.141592647509924, 3.2689490957552283, 1.1696749274549658, 3.656940298418632, 1.418774641266783, -0.6154796992737697, 1.57079632908446, 3.906002231953504, -1.6094251340194683, 5.911045383838438, -0.09474504892640277, 3.9269907883821156, -4.712388982345858, 3.531900420964907, -2.701842427932343, 2.728334403932701, 1.665541375198194, -4.712388976756994, 3.584454232746003e-08, 2.747662515563989, -1.0099745059033434, 1.5707963223033918, -6.283185308515168, 6.283185304954134, 4.712388982499544, 1.3652406440829166, -1.9717065658794286, 1.570796316203271, -1.570796310237535, -8.492225472465768e-09, 6.283185295438324, 2.1702821826282004, -1.7307079337981268],
    ),
    (7, 2): dict(  # 20 CZ, path, seed 0, robust pick 4
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
        cz_layers=[[(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (5, 6)], [(0, 1), (2, 3)], [(3, 4)], [(4, 5)], [(5, 6)]],
        params=[2.967492661493879, 0.8168458820013998, -2.003723848363806, 4.312730799132894, -0.9107507716748718, 1.4940346153324664, -0.6120009416648662, 2.8753747264975495, -3.1770972222804117, 0.3192704873557996, 1.879184760054684, -0.8494593213126443, -3.001719359170047, 2.534520093205146, -1.1304784574292615, 0.6154796831604005, -1.5126116121812235e-08, 4.100791818808925, -0.558419780783166, 0.9633819976315638, -0.014033797866221822, 4.210026020199953, 3.0821935805591766e-08, 2.180479957818448, 2.617993858138423, 2.617993857490405, -4.069420820163932, -2.5239650524795563, 2.0731592505063188, 1.5707962635903796, -2.526113022899635, -1.0471976366568048, 5.309838121475653e-08, 0.9553165610879187, -3.1336973621667483, 0.2117599888347925, -0.4560721879121676, 4.096909329643039, -0.6838504514887151, 1.8056762124565968, 0.6154796857319087, 2.1783807331212452, -0.3989773148753558, 2.3561944916708253, 1.5707963069052584, -2.45774225340186, 2.127827760458616, -2.2869103661449515, -0.41622352641856536, 0.849645597348783, -3.163548500474424, -2.306106083585746, 1.5707963243299767, 2.349681269715352, -3.21690816902943, 1.3569837256186912, -1.5875069698559126, -2.8644218675817177, 2.332200464511252, 1.7381127737396762, 1.570796341743864, -3.1355613212616857, 0.7262571333561135, -0.2573963970453183, -2.3705718732797414, 2.091469814931959, 2.289657494222605, -1.0684090142361817, -1.5707963285657756, 2.2599735129214156, 2.5660541740450897, -2.5970320611391657, 2.594824777677355, -2.4569739585348316, 2.6392053295406135, 1.570796301362198, -3.1415926732597668],
    ),
    (7, 3): dict(  # 25 CZ, path, seed 0, robust pick 13
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
        cz_layers=[[(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)], [(2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)], [(0, 1)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(0, 1), (4, 5)]],
        params=[-0.3942314280927663, 1.4092262234909678, 2.5385912449389814, 0.8702059606520871, 0.6487688377703466, 1.2737463439107866, 1.9828598153297126, -1.8043094634563046, -0.10145935997514405, -0.6391434117148288, 2.6334610467336432, 1.3674001167779066, 1.1492499702137444, -2.8333352848561235, 1.0140502909364142, 4.851352548047241, 2.551639092085599, 2.6870542599761844, -1.531892087279405, -3.9327395798784464, -2.100602228503966, 3.497022069033054, -1.5533503001156914e-08, 1.976904916524363, -2.3547256087627524, 1.5707963305705859, 2.328667685775536, -1.7613112860988238, 3.613401877140848, 2.1091378833128034, -1.2401026586424566, 3.1415926927756277, 0.9553166020516288, 1.5707963234284212, -2.828336223425421, 2.8344549587996832, 1.0324547646279119, 0.5752675381520481, 5.144270899290797e-08, -1.5707963544151784, -0.9165071171804843, -0.31325641108912194, 1.6661751043135566, -1.078930635817034, 1.7318282249116903, -3.841211615494095, 0.04461094346511278, -1.617337574616888, -2.7181960383259276, -2.7614149038680127, 1.5707963588427158, 2.468978769044101, -4.012770026905119, 0.29589799861053134, -0.7388568891100873, -0.42339662833402436, -1.2399857777587897, -3.926990826891757, 3.259904301411565e-08, -6.521426334902808e-08, 1.1478933643403952, -2.7228828964553564, -1.8588125380322826, -2.7171955269594563, 3.92699081588842, -4.712388984421516, 3.42347480976285e-08, -1.1478933901648036, 0.04119851838543039, -1.585650124864282, 1.146399240285612, -1.5707963280630206, -0.5593017029595194, -0.12587204601869884, -1.911305274073867, -2.9472117032301988, 0.4453862890888238, 3.1415926642838246, -1.570796328806121, -1.0114946294955436, 1.6966683619492111, 4.88968465336644e-09, -6.283185325678817, 1.4282800879394413],
    ),
    (8, 2): dict(  # 26 CZ, path, seed 0, robust pick 15
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        cz_layers=[[(2, 3), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(2, 3), (6, 7)], [(1, 2)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (4, 5), (6, 7)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (5, 6)], [(0, 1), (4, 5), (6, 7)]],
        params=[-1.747091435851154, 1.3650438966653893, -1.4963156882850905, 0.7243754809984432, -1.5707964486317998, 1.5707967144379076, 1.5707963250708792, 1.5707963140450014, -2.8727113975652174, -2.9687515255419883, -3.2156355428303773, -0.7711331145531166, -2.667826536462889, 2.829960688705391, 1.0133799837386552, 2.7585709201560156, -0.10280586172410847, 0.4803497430455318, -0.724607804230786, -2.4797469148653417, -1.6916262713862011, 1.7446967904483326, 0.6154797610523011, 0.9404390249538708, 0.18525438473856937, -2.018234645804437, 1.4945634073743364, 0.787941171661171, 0.5796932975028568, -1.0630548480984767, -2.4407821249660384, 2.080665086948523, 0.7802819949728967, 2.0943950804617053, 2.29867877120356, -1.2515886812207433, 0.32653479788033807, 2.4599506594817933, -2.213504370338546, -0.5281477605201444, 0.009788489860277406, -2.186276016330939, -1.4274485566246515, -0.4241943790539373, 5.759586419007105, -4.712389079908457, -1.6288987943988045, 2.841017326072248, -2.1960645371412717, -2.526112972386863, 1.7452773306202503, 0.733461854349365, -1.0471977353031687, 0.6154794753837496, -0.6154796829156034, 1.8896505216829342, -2.022360551046738, 0.19123498417378482, -1.2747959688866757, -3.875054534689223, 1.5707962700262, -0.4100195899810998, 0.08178840926145188, -1.0258666611839367, -1.9091759058927387, 0.17872120720750784, -0.4704812243526994, 1.5707963262955833, 2.8034974131303967, 1.3653360795356821, -2.7248405599107683, -1.544119701620624, -1.9101773492711813, -2.7261507342033724, 1.5707963201747963, -0.8164117758785704, 1.2709388392898626, 1.520251745024208e-10, -0.49854054289071703, 2.7226794918235093, 0.3439266395111363, -1.5707963238353995, -2.573461711934341, 0.12875433355091292, -1.2709389551799208, -1.8290929670694593e-08, -2.3561946328910452, 2.2035002657418508, -4.195444697288995e-08, 1.5707962892950067, 1.0026653601802733, 2.2584537394976825, -1.2327011127206535, 1.5707963040481836, 1.5707962805424485, 4.712389256241399],
    ),
    (8, 3): dict(  # 43 CZ, path, seed 0, robust pick 2
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        cz_layers=[[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)], [(2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3), (4, 5), (6, 7)], [(1, 2), (5, 6)], [(2, 3), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3), (4, 5)], [(3, 4), (5, 6)], [(0, 1), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3)]],
        params=[-1.4528572997252809, 1.6939772098043697, 1.9262873874942834, -1.6925572510654292, 2.2125195123530297, 1.4432215887314443, -1.4546771756226995, 1.6341048544231525, -0.22651675507784388, -0.8108278044615322, 1.882978597136051, 0.8622657596163803, 2.296580987075838, -1.0773803660419103, 0.4366878315036302, -0.7575944030329999, 2.9667375171411225, -2.4690583830172494, -1.191201287384057e-08, -0.2591889038072575, 1.2569096870390104, 2.3654749055468707, -2.7762951741351856, -1.6480755433945689, 0.08389752218718916, -3.814126949681976, 1.5707963316686613, -1.1622069848356023, -2.3378124104335596, 2.370638864232938, -1.805638217590833, 0.8854396014517993, -2.0568682131118896, -3.1415927980820175, -0.5309868090034835, -1.1774428735778373, 0.2383793019006771, -3.83260654748873, -4.749938585314228, -1.052870527690528, 2.3103783279462937, 3.1415926330798967, 3.1098524976377657, -0.6347288264336309, -3.157371375154515, -1.7585448968094595, -3.1415928577697394, 2.7307813312823, -0.5031433283414898, -1.497816298145078, -3.141592672027004, 2.205525147436584, 1.5793073544037086, -0.2982929562772877, -1.4824047400423204, -0.7563870533550071, -0.6355191015137842, -3.4981962094023373, 1.7667980694982486, -1.4173921378014023, -3.0957319945588955, 3.1597759335896063, -2.048984377808955, 1.0031490085872614, -1.0413054952292278, -1.5707963174182402, 2.0562182661662722e-08, 0.9786326943754086, -1.5707963297288574, -1.0469757669665862e-07, 0.9734640662055081, -0.8798256738799023, -0.10686526163011476, 5.6671417383888585, -0.6452230793010015, 1.5707962327636695, -6.044537776914961, -1.570796324001922, -4.164366844936398, -0.58233274050579, 0.1068651749811387, -1.7141262774601425e-08, 0.7853982202936503, -1.0312183659906449, 6.284715306014242, -2.0980660678725027e-09, -0.3524146891174211, 1.634253862883702, 2.260592449436984, -0.08776307929396919, -0.8964735655653115, 3.141592748724993, -2.931345223488456e-07, -3.1415930127282183, 1.923211013060196, 1.2475423281385833, 2.500695544972951, 1.2229806023138918, 2.8806816284308, 1.5012009982465802, 3.200036885598149e-07, -1.750033994459817e-08, 2.9042166349316685, 1.8680521647452333, 2.747291158251342, -1.5707963330615746, -1.9842079900968084, -1.5012006897811816, 1.446448263527206, 3.3180684294729814, -0.6511152489333101, 3.0692170038525233, -1.176494724639539, -1.5707963259754543, -8.538205518028015e-08, -3.141592695255856, 4.742387529607343, -0.27562881331612815, 0.6188011336038999, -2.5616890029049535, 1.570796285504707, -1.5707963351730927, -4.712388915728165, 0.20422051000965505, 1.5423276884559203, -1.295167479952093, -2.1895974341752096, 2.634064663997086],
    ),
    (8, 4): dict(  # 29 CZ, path, seed 0, robust pick 14
        coords=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        cz_layers=[[(3, 4)], [(2, 3), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(2, 3)], [(1, 2), (3, 4)], [(0, 1), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3), (4, 5), (6, 7)], [(1, 2), (3, 4), (5, 6)], [(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5), (6, 7)]],
        params=[1.547198989998157, -2.192123647098601, 0.06931157035242458, -2.127957126320808, 2.2833469410568434, -2.160483728229487, -2.7860358403476355, 1.7852074422747406, -0.8574739597764105, 0.6732315697380621, -1.6577635220233697, -1.4207147786755805, 3.8473144649058844, -1.7857426185475427, 1.21523948413521, -3.3560037816655597, -0.004198826183240813, -3.1934968805335133, -1.4355442890070365, -2.237433838909259, -0.9223453434117104, 2.336959039600489, 1.634173992579475, -1.5525024825750688, -1.7170495692696344, -1.3141585233950228, -1.5707963180564533, -0.8506827351245877, -0.348160400452928, 3.309528269803486, 0.39064960802909826, 1.9141753132392516, 2.343156708961155, -3.3982304470059996, 1.5707963164646408, -3.1415926408592867, -0.23984220891096922, 2.079142535865502, -3.019238460414366, 1.4026127068746377, 0.692700772820962, -3.1415926318648713, 0.4711905796767751, -1.3604062178756855, -2.4911019318437515, -1.9141460693156949, 0.4423128178197566, -1.2121834937704705, -1.3811631605860242, -1.5707963082774687, 1.0996057789909608, -3.351982773047053, 0.8196319107198208, 1.237864244954955, 1.570796343955774, 2.7996051877212733, 0.42584652912954774, 0.6895745041167302, -3.141592646249463, -1.5707963083947118, -4.712388980666592, -1.3110794700647877e-08, 0.7853981674979944, -1.228808816525811, -2.0845470123503893, 2.3561944790627387, -1.5747251394914847e-08, -0.785398166751132, -4.712388978720002, -1.5707963266774692, 3.926990816494695, 1.60698247705685, 3.655343348116879, 1.5707963276661134, 4.5388134591135784e-09, 1.570796329583965, 4.460134156418878, -0.31265938414584604, -2.3409508365436533, -0.001007751407567284, -1.1380244189286717, 0.24497881352567927, -1.5707963125445377, -0.9762019528890976, 1.5280363069761873, 0.7580899024028405, -2.0994244961237554, 2.1363876711803926, 1.1380244321383224, -1.5707963197573402, 1.0691935657763076, -1.0096401934073185, -1.5280362997714414, -3.899682555550743, -0.27201364791671406, -2.171566084946386, -1.5707963148500337, 1.2687064319744044e-08, -1.0691935839103623, -1.1557505083119262, 4.460134159231326, -1.258136945539457, 1.5707963262061964, -7.49252850561493e-09],
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
_SCS = {  # (m, k, all_ones): (cz_layers, ry_params)
    (2, 1, False): ([[(0, 1)], [(0, 1)]], [-1.015321455826436, 0.9012076585199794, 0.7853981635004875, -2.1584696466585975, -0.7290898508023025, 1.5707963268232388]),
    (2, 1, True): ([[(0, 1)], [(0, 1)]], [1.5707963299106977, -1.5707963279692112, 0.7853981644905792, 0.7853981648548624, -1.5707963324863208, 1.5707963356010937]),
    (3, 1, False): ([[(0, 1)], [(0, 1)]], [-1.1407597698521028, 0.8649861674793102, 0.9553166183850991, -2.025741846810719, -0.7639646880798547, 1.570796325631784]),
    (3, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.5261129440759578, -3.1415926534463203, 1.6814398415163123, 1.8673550539072388, 1.5707963269906793, -3.252236168399352, -2.482834760860102, 0.6154797037939795, 2.6934577818389718, 1.4461898972389164, 1.570796322171429, -1.7381411647522957, 1.6954027561629759, -1.5707963268805634, 1.5707963267467884]),
    (3, 2, True): ([[(0, 1)], [(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.5707964203929847, 1.5707963520373425, -0.7875640413941856, 2.526112939928778, 0.6154797147869595, 3.442482558804403, 1.0254108620531002, -1.570796367896398, 2.0574703112574455, 0.5453853558120263, -1.8785361722138427, 2.530702774328467, -3.5450922942447773, -0.3077398612692013, -0.9599064343508606, 0.4034996247001735, -0.3077398564289521, -2.2855112763759493, 0.7539365622784483, 1.8785361808190313, -2.426877804746498, 2.3876561191795918, 3.1415927900973726, -1.5707963312735973]),
    (4, 1, False): ([[(0, 1)], [(0, 1)]], [-1.2067991147336417, 0.8501571332801536, 1.0471975503230786, -1.955939742194762, -0.7784186580454592, 1.5707963250791273]),
    (4, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.3561944902018714, -3.141592653201629, 1.681439841579331, 1.7823958256956658, 1.570796327062821, -3.252236168336336, -2.567793989071676, 0.5235987756088417, 2.7393982488821935, 1.4461898972013847, 1.5707963267701406, -1.6922006977090764, 1.6954027561254483, -1.5707963268858744, 1.5707963268049172]),
    (4, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.5235987756403431, 1.5241303672196429e-10, -1.5707963268101248, 1.570796326790054, 1.5284495953370514, 1.5707963266253244, 2.6179938771048707, -0.3973794920781327, -2.05204837083282, 2.795357843898402, -1.5707963267929774, 1.4445770418768198, 2.7795191979441385, 1.5243320518143768, 1.570796326754649, -3.956499180968387, 0.3620734557846727, -3.188056931174852, 3.1415926535875043, 1.9930037750916123, -2.9766722307666393, -2.0099596830663353, 1.570796328158474, 1.052883329402526, 2.9766722311973584, -1.3226001086956851e-08, 1.5707963268001304, 0.12521391232496118, -3.2577292833979974, -1.0203875102664337e-08, -1.570796325149596, -2.0735106129714755, 0.11613662974040719, -1.570796326802337, 1.5707963267949383, -2.6388783674678624]),
    (4, 3, True): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(2, 3)]], [1.5707963062814767, -1.5707962713225228, 1.5915914333336953e-08, 1.570796227073341, -1.7720651735309385, 0.7853981867001699, -1.5707963511249923, -0.22775788406567657, -0.5841292782624524, 1.0154709729356895e-08, -0.7853981708967535, -0.4267415368013041, 1.6797599165278145, 2.3561945140333775, -3.14159262901271, 0.06740666596816665, -1.6797598588283682, 1.5707962893558238, 0.5235986287973314, 2.157888978693121, -3.1291894908699924, 0.30773987999020186, -1.047197590649111, -1.1319250404546353, -1.5831994422047118, -0.3077398688417215, -3.1415926888392445, -3.5804639918184415, -2.351240053941933, -0.3077398722401282, -3.1415926669134913, 4.650550553507826, -5.502741620140648, 0.30773989386597483, 0.7853981174131546, 0.06183841104645399, 3.307166940539704, 9.271642357478542e-10, 0.7853981113726165, 1.8846489422097708, -0.5529118002335035, -8.825476369880942e-09, -2.356194441066823, -1.8846490607070039, 5.099726468331428, -3.141592663128518, -2.3561944279409666, 7.026400268227146e-09]),
    (5, 1, False): ([[(0, 1)], [(0, 1)]], [-1.2493722484973575, 0.8420415032801906, 1.1071487289052535, -1.9109407348473444, -0.7863771580214348, 1.5707963302956998]),
    (5, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.2555155297783154, -3.1415926528393414, 1.6814398416230167, 1.7320563455920457, 1.5707963272402872, -3.2522361682926526, -2.618133469175297, 0.463647609041793, 2.7693738320971537, 1.4461898970645548, 1.5707963266562575, -1.6622251144941151, 1.6954027559886222, -1.570796326979636, 1.5707963268268799]),
    (5, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.684719200760806, 3.89382212395982e-10, -1.5707963267475615, 1.5707963267336866, 1.4478893830798967, 1.570796326692386, 2.6779450484263787, -0.3674039087853684, -2.1326085830899757, 2.7449228296430337, -1.5707963267538714, 1.4745526251695888, 2.7795191977243827, 1.5244275926819684, 1.5707963212754228, -3.981668924602958, 0.36207345556490583, -3.1879613890938696, 3.1415926535184586, 1.96783403145703, -2.976672230610278, -2.060203622833494, 1.5707963287255762, 1.0277135905950752, 2.9766722313537137, -4.066488119839899e-09, 1.5707963267903209, 0.1000441735175128, -3.257729283173108, -6.295575621867655e-10, -1.5707963293194807, -2.073510612800667, 0.11613662996529607, -1.5707963266985356, 1.5707963269296419, -2.6388783672970573]),
    (5, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.936893910683001, 1.871831455622161e-08, 3.141592591392152, 1.5707963960678542, 3.1415925774019144, -0.11735621215646262, 3.141592673859249, -2.1394113976540326e-06, -3.1415903419986133, -2.791479456665574, 2.6120048372655607, 1.5707963096362427, -1.5707941247056312, -4.712391388645436, -0.8137608055784339, 0.06594020748485975, 2.2555155295958573, 3.8263118571164636, -1.5707957936378947, -0.5661986805647138, 0.4147308149040186, 1.570796327049326, -3.821641166415872e-08, 2.3561944909274692, 3.2441437257665067, 2.784515549756841, 1.5707956562641578, -3.5846312164310614, 0.443038561075173, 3.0353930185057965, 3.2290247643657475, 3.14159330223582, -4.71238856560492, 1.570796774112821, 0.1061996412574048, -0.14508591862222386, -3.1415926610720684, -1.5707959949354213, 1.5707959844508377, 2.7060146909274128, -2.6880665512405644, -6.283185244722265, 1.2284367264929819, -1.913155929314181, -1.1352183259637156, 2.771796463164189, 3.141592662190583, -0.7853981644060074, 1.599243135582578e-08, 0.8481806271369727, -3.376244876129918, -7.573690193787372e-09, 3.141592523789559, 1.570796327020095, 3.8642083539270997, 0.1509223963730345, -3.1415926077121346, -1.570796403085478, -1.5707963651568324, 7.853981267970707]),
    (5, 4, True): ([[(0, 1)], [(1, 2), (3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(3, 4)]], [1.5707962405969675, 1.5707962905123638, -1.9986545615303268, 1.882695571581244, -2.150853264573379, -1.3630175444522235, -2.677945032234645, 0.8741077143157976, -0.3118992649631603, -2.5615353750516645, 0.8993699254217412, 3.141592661669857, -3.607108004568963, 1.2151378045825205, -0.12295362875472635, 1.4035755263265186, 3.1415927490942335, -0.2192038701614214, 1.4628072402230956, -2.554991419649251, -0.04684644596051661, -2.174502111967305, 2.6953431446289735, 1.570796120146085, -2.295859340028196, -1.9942050409288767, 3.7452984615716796, -2.6021054572634482e-08, 0.5277874645278107, -3.4695003705082152, -0.9333202521755063, 2.013834971859119, 1.5707962329015617, 2.170766630763149, 1.107362284280402, -0.2843893751065307, 0.44303855516324897, 3.1415925356617485, -2.6985540883302965, -3.195983908742667, 0.2843893956898251, -0.4430385404753229, -1.5707960835613908, -2.356194485470851, -1.2561865755063177, 1.8097006292329474, 3.5846311861267632, -1.5707962869997616, 2.776322687234782, 0.7458858235239523, -1.1113611822858258, 1.3080336495753961, -1.5707963186640008, -1.2055262213991538, -3.0191443300641976, -0.9142762245197629, -1.3080336385638183, -2.799233049705837, -1.2284367313976283, -1.1829439056399105, -1.3475879118655933, 1.927622343388024, -2.356194488014053, -3.141592656399593, -0.7631716757874621, -0.2812515271040594, -1.0164348406804926, -3.0157594101766727, -4.712388980018044, 2.33396800250067, 1.8447761792520496, 2.230405148108358, -1.6966296201713749, -1.5707963309089443, 4.712388843133264]),
    (6, 1, False): ([[(0, 1)], [(0, 1)]], [-1.2797755554426096, 0.8369137344221296, 1.1502620026722152, -1.8788000046902575, -0.7914245858992704, 1.5707963332316701]),
    (6, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.186276036096864, -3.1415926501151947, 1.6814398431344826, 1.697436600970363, 1.5707963268663996, -3.252236166781186, -2.6527532137969825, 0.4205343368891023, 2.79093046674452, 1.4461898987749984, 1.57079632171125, -1.6406684798467481, 1.6954027576990653, -1.5707963256518889, 1.570796328167465]),
    (6, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.7853981589339939, -3.549500306381916e-10, -1.5707963266138816, 1.5707963268520218, 1.3975499050013058, 1.5707963277404189, 2.7210583204638326, -0.3458472821383155, -2.1829480611685605, 2.7102644641988918, -1.5707963268088607, 1.4961092518166574, 2.7795191968060693, 1.5244662001349762, 1.5707963257110988, -3.998978796558535, 0.36207345464661134, -3.187922798068281, 3.141592653580437, 1.9505241595014517, -2.976672227870911, -2.0947847693487045, 1.570796346078658, 1.0104037115111997, 2.976672234093094, -1.6588549673737222e-09, 1.5707963267020681, 0.08273429443362254, -3.257729282577532, 1.4514719038134356e-08, -1.570796324836202, -2.0735106124243807, 0.11613663056086608, -1.5707963265947398, 1.5707963268300777, -2.638878366920778]),
    (6, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.860977860665075, 4.211887568202972e-09, 3.1415925994090874, 1.570796374161921, 3.1415925776238254, -0.19327226217438234, 3.1415926593528214, -2.131394467864578e-06, -3.1415903390651003, -2.7699228206087043, 2.5360887867704607, 1.570796349028954, -1.5707941166887, -4.712391385711917, -0.7922041695215678, -0.009975843010240588, 2.3561944886041624, 3.926990817642552, -1.5707956968073693, -0.5446420455053238, 0.4147308157164884, 1.5707963267844822, -3.8721587761431525e-08, 2.3561945037224032, 3.265700360825894, 2.7845155505693104, 1.5707956545162036, -3.61925096606253, 0.477658300054291, 3.035393022424044, 3.2290247651782216, 3.141593300487861, -4.712388457029337, 1.570796847538605, 0.10619964517565275, -0.14508591780975338, -3.141592659337703, -1.5707959270603578, 1.5707959215145864, 2.7060146967600858, -2.68806655042809, -6.283185242987902, 1.26305647846094, -1.878536190479966, -1.1352183201310444, 2.771796463976664, 3.141592663619797, -0.785398167669143, 2.0055353870508316e-08, 0.848180627677561, -3.376244875317446, -6.144477727820846e-09, 3.1415924895261513, 1.5707963263981555, 3.864208354467685, 0.15092239718550374, -3.1415926197304955, -1.5707964300630364, -1.5707963642802332, 7.853981183269579]),
    (6, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [0.42053433258321876, -1.549842151719446e-07, 1.4069652055640711e-07, -4.814425782158794e-07, -1.5707961561581192, 1.570790722550021, -2.1454701325680245, -1.5707955705745127, 1.5707964812134094, 2.067995541065594, 0.23169267349743675, 2.1391579961605434, 1.7249357989312353, 2.186275445323593, -0.11717951096989422, 1.4702009657337565, -2.9527510298225863, 1.4229689420984386, -2.2027260695511885, -1.5707957726301593, -2.408933453134081, 1.1741930365228812, 1.5707954330869682, -2.461089457053677, 2.127905062808956, 0.6362712967333183, 0.5821750039480882, -3.1404325708211096, -2.0484546398535888, -2.251304676921941, -2.0345087876371553, 1.4217632346039295, -0.442161653534664, -1.5719571581091702, -0.4776583127415695, 0.6469591185897509, -1.667821701730063, -1.2635042797529583, -0.14000923504819818, -1.570792161358871, -2.6639343381426914, -0.6469591545480542, -0.8724400198514266, -2.3653261921985655, 0.7853981675227605, -2.3561944649185995, 2.0484546356300783, 1.8552360385534807, -1.6335920843664318, 3.1415920799862747, 3.534291723024262, -5.105088069967489, -3.141592647685746, -1.8552360614075443, 1.0979431865365783, -3.14159205469281, -1.570789511304214, -1.5707952706479762, 3.1415926565033936, 2.058234102260265, -1.8348282752819982, -3.292403744404596, 1.5707976850896372, -1.5707952821403859, 0.4138266023375347, 1.5713217183930421, 2.833499053862503, 0.195996151496813, 2.936540055379419, -2.7488935872297176, 1.1569696989723142, 0.11404910369585425, 0.3080920653761022, -0.19599648934020245, 2.1685480003493236, 2.7108066598343106e-08, 1.570796332022784, -2.172808634567402, 0.32515808135153584, -3.292404013741152, 3.0696863408990336e-06, -1.5707963235490705, 3.1415926592515793, 3.1287609287352875, 3.0097733658169754, 1.7960283654163407, 1.5708002638017422, 1.570796010271263, -1.5707963369046134, -1.5579646289453564, 0.5435464526853294, -1.796028442438568, 3.141596831247581, -3.1415923169523756, -1.5707963423059104, 1.570798642693676]),
    (6, 5, True): ([[(0, 1), (4, 5)], [(1, 2)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(4, 5)], [(3, 4)], [(0, 1), (2, 3), (4, 5)]], [-1.5707967063062631, -1.5707966053324622, 3.5084250869050506, 2.174514996879484, 1.570796416578593, 1.5707929948400583, -3.2321304725218423, 2.356194471144124, 1.2039646207485126, 3.3011564138473246, 0.5350107046829758, -0.7559616733388369, -2.2656566495472283, -3.141592620876377, -2.5261129574939196, 2.3783102821238873, -0.11447635268345013, 3.4770199739511316, 2.116449016504574, -2.526112956192014, 3.1415926713783997, -0.7853981749391284, -1.5707976095503686, -2.861615120789052, -2.1164496684404885, -1.5707966638844573, 2.7488938090961086, -1.963495401656546, -0.45511302034991025, -1.8507737152827088, 1.0636735573579283, 9.774213404797473e-08, 1.5707965937153012, 1.5514493367556417, -1.176648753082684, 3.8577088656033793, 0.5071242981683495, 2.799233010567725, -3.141592507694479, 1.5901432903592503, -2.2952290713254637, 2.4254794148324854, -0.676823189660211, 0.34235962966412387, -2.7488935761256807, 4.712388798444849, -3.619250996113091, -0.3200481872931435, 4.111924830929902, 3.012601329703603, -1.5707965716319867, -1.7768032399906891e-09, -2.6639342962382395, 3.4616407817596326, 2.318439640691369, 3.270583870898994, -1.5707963501098192, 1.570794967995352, -2.356194537811598, -2.33703683514301, 3.671235835350571, 0.34235950394084885, -2.74889361202183, -1.5707929780111125, 1.9761083489767717, 2.435620907081589, -1.7586054687380437, 2.8692998441890154, 0.6218828793787473, 1.5708013131281864, 2.7362804218982, -0.3353872447474936, -0.827287592909352, 2.1820763809269055, -3.3376851305668014, 2.2622084419193995, 0.30773983276575617, 1.8075994239910924, 1.675548169981364, 1.4029888051307544, -0.4257903926571502, 2.7579203924439177, -4.211139133119618e-08, 2.273252018801587, 2.4811397531413544, 3.312772426368298, -3.1415926553194184, -3.141597413636725, 4.712388976721598, -0.7024557017616246, 1.5707965560631183, -2.245069441510587e-07, -3.141592687710977, 1.5707932374034428, -4.712388951043549, -1.5707953552648874]),
    (7, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3028972020112384, 0.8333779884976906, 1.1831996402511757, -1.8543528424609963, -0.7949135493121982, 1.5707963265368166]),
    (7, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.1347389682452946, -3.1415926514608477, 1.6814398416349414, 1.671668064646064, 1.5707963280561572, -3.2522361682807266, -2.6785217501212792, 0.38759668673761727, 2.807399293372445, 1.446189896108908, 1.5707963269096916, -1.6241996532188234, 1.695402755032974, -1.570796327655232, 1.570796326742017]),
    (7, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.8570719450308016, 2.7224391828359735e-09, -1.5707963266472988, 1.5707963268529699, 1.361713010004807, 1.5707963253282293, 2.7539959774212326, -0.3293784467687446, -2.2187849561650554, 2.684477224580026, -1.570796326724253, 1.5125780871862273, 2.779519198114746, 1.5244849215464118, 1.5707963247076349, -4.01186306499019, 0.3620734559552743, -3.1879040578981135, 3.1415926535883965, 1.9376398910697978, -2.976672231734907, -2.120534579407098, 1.5707963288224978, 0.9975194526334087, 2.9766722302290867, 8.618954545886777e-09, 1.5707963268074507, 0.06985003555581411, -3.257729283065587, 1.3163520811314764e-08, -1.5707963293097542, -2.0735106127505665, 0.11613663007281427, -1.5707963265560467, 1.5707963268517058, -2.638878367246968]),
    (7, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.811855528163934, 6.163306385442014e-11, 3.14159257450944, 1.5707963929540798, 3.141592588272582, -0.24239459467553223, 3.1415926552025617, -2.1562941165334087e-06, -3.1415903524728934, -2.753453995612028, 2.4869664510684637, 1.570796290275634, -1.5707941415883497, -4.712391399119711, -0.7757353445248908, -0.059098178712242364, 2.427868274085418, 3.9986646048125714, -1.5707956482183376, -0.5281732181132602, 0.41473081590479977, 1.5707963242941279, -3.649741880095522e-08, 2.3561944856158545, 3.2821691882179596, 2.784515550757625, 1.5707956729654902, -3.6450194986908633, 0.5034268439302398, 3.0353930165705947, 3.2290247653665314, 3.141593318937147, -4.712388442513509, 1.5707968564171182, 0.1061996393222045, -0.14508591762144218, -3.1415926695969363, -1.5707958149091832, 1.5707958656622094, 2.7060146841431343, -2.6880665502397783, -6.283185253247129, 1.2888250003384225, -1.852767648540013, -1.135218332748002, 2.7717964641649786, 3.1415926638256817, -0.7853981583674773, 1.5366865700216192e-08, 0.8481806290140685, -3.37624487512913, -5.938592686656713e-09, 3.1415924413673095, 1.5707963226780821, 3.8642083558042044, 0.15092239737381608, -3.1415925843946475, -1.570796422349087, -1.5707963569840295, 7.853981116210921]),
    (7, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.5776500324580227, 3.141592358382348, -3.1415926359429647, 3.1415926962306853, -1.5708016074384625, 1.57079337465175, -2.4629790476615154, -4.712389021914221, -4.712388995413269, -2.5270113624107724, -1.6034664841135167, 2.813958621000956, -1.2425562300457922, 3.998655116782793, -0.13453469360152306, 3.199519366487829, 1.2158697955088758, 3.0816300186761847, -1.3360714002069587, 1.5708058121143404, 0.8482590789184595, 0.8982881259181561, 1.5708004736565826, -1.5012746529655874, 1.863228749529144, -4.712389051107567, -2.407066087126139, 2.064259771759787, 3.6450195214507763, -0.06951809499040341, 0.7067351753808948, -1.5802353352954774, -2.2641973410745138, -0.4934632559505372, 1.0673695094630073, -0.5437003158632249, -0.4830844791676599, 0.5307047207539994, -1.6119199177820374, 4.712388866887893, 2.074223156788373, 3.685292916942179, -2.281757891985551, -2.0920621638272316, 5.497787171793717, 5.497787104740473, -2.6381657908597305, 2.9008432983581107, -1.610644223476683, -3.141605976736388, -3.4984548205829675, 2.7130567012456592, -6.960607324793232e-08, 0.24075027695944218, -3.3457009163203426, -3.141605975212197, -1.5707918161921472, -1.5707959466254569, -3.141592643371282, 1.5361883234962592, -2.9019069169087413, 0.06614416044605188, 4.71238910035252, -1.5707967177641613, 2.4915012065489424, 1.6028159576049121, -1.6794534324136754, -3.141587867399735, -0.17406386302494634, -0.3568622218244778, -0.9207048928984117, 2.325228257842731, -1.4621405417248492, -3.141588029467776, 2.8871205227797017, -3.395858663399311e-09, -1.5707963325968475, 2.3897471894789963, 1.2421356907490322, 0.06613476150728637, 3.1415932155412136, -4.7123889845219615, -2.9010666534545572e-08, -3.187996632652778, -1.6050850471074223, -1.4377258729314182e-08, 1.570791050615118, 1.5707963139130154, -1.5707963056964873, -1.5243923441400924, -5.9558134170279216, 3.1415926804682517, 3.14159301349042, -3.141592624649799, 4.712389052680858, -1.5707959320249427]),
    (8, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3212505296537456, 0.8307917205497494, 1.2094292020806356, -1.834944378108838, -0.7974701857608345, 1.570796326248331]),
    (8, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.094395104259356, -3.141592649861804, 1.681439841912144, 1.651496132365669, 1.5707963286018456, -3.2522361680035243, -2.698693682401674, 0.36136712550263966, 2.8205140742161325, 1.446189895601535, 1.5707963245891852, -1.6110848723751359, 1.6954027545256016, -1.5707963280095594, 1.5707963268875684]),
    (8, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.9117382921717447, -3.4590898673866783e-09, -1.5707963254926276, 1.5707963277411525, 1.3343798368042403, 1.570796328167911, 2.780225529207832, -0.31626366651715226, -2.2461181293656516, 2.6642948836947005, -1.5707963268852754, 1.5256928674378079, 2.779519197422447, 1.5244953199408244, 1.5707963257852704, -4.021949029203483, 0.36207345526296175, -3.1878936610324193, 3.141592654283685, 1.9275539268565236, -2.976672228653015, -2.140696110717492, 1.5707963242351763, 0.9874334829468264, 2.9766722333110045, -1.3596703063668512e-09, 1.5707963263792186, 0.05976406586926228, -3.2577292833597182, 6.573232268447581e-10, -1.5707963259387991, -2.0735106110430643, 0.11613662977868619, -1.570796326178581, 1.5707963279977222, -2.638878365539461]),
    (8, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.776018631153869, 3.690623860378068e-09, 3.1415925756828003, 1.57079637144368, 3.141592576198967, -0.27823149168559996, 3.1415926588315477, -2.1551207550799243e-06, -3.141590363026949, -2.740339211484763, 2.451129557828329, 1.570796288711113, -1.5707941404149877, -4.71239140967377, -0.7626205603976312, -0.09493507195237647, 2.4825346271262325, 4.05333094696149, -1.5707955163808243, -0.5150584386049629, 0.41473081454951904, 1.5707963343962257, -4.186835705317384e-08, 2.356194515911853, 3.2952839677262653, 2.784515549402349, 1.5707956690846572, -3.6651914238785848, 0.5235987697558312, 3.03539302063172, 3.229024764011251, 3.1415933150563093, -4.71238828654429, 1.5707970011211807, 0.10619964338333165, -0.145085918976726, -3.1415926660175977, -1.5707956821144438, 1.570795772048599, 2.706014691461905, -2.6880665515950604, -6.283185249667801, 1.3089969609356658, -1.832595748826142, -1.1352183254292174, 2.771796462809704, 3.1415926628709707, -0.7853981829456529, 1.993641733286267e-08, 0.8481806236680415, -3.3762448764844124, -6.893303962651722e-09, 3.141592391666859, 1.5707963371139528, 3.864208350458181, 0.1509223960185396, -3.1415925801256535, -1.5707964440972924, -1.5707963579492075, 7.853980971137368]),
    (8, 5, False): ([[(1, 2), (3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(3, 4)], [(2, 3)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (4, 5)]], [2.17210049984681, -1.5707963364933308, -1.5707963279495292, -1.570796326960163, -1.5707963251959494, 2.9891268607565937, -1.1249029131683794, 3.1975562055844735, -0.7853981677463828, 3.665191434395755, 1.0471975313056234, 0.15246580556114217, -0.8673479208749676, -2.4121580387916666, -5.811712208208417e-09, 1.5707963276646373, 3.1415926612725715, 0.2802310934116177, -0.32913148368282347, -2.596475567763119, 2.6857235021441954, 1.5707963322813416, 0.45586913372208465, -3.2411401837974907, -1.2116083596789355, 1.0256792568019226, 2.02666547627613, 1.7956949718277733e-09, -1.5707963312260727, 1.4520121352772053, -0.209906162975556, 3.1415926630251354, -0.6590580376220412, 1.5707963315646591, 1.570796309469623, 1.6895805360266756, 2.279013758504131, 1.7233459178144073, 3.1415926561278518, 1.5707963305774373, -0.4558691341932658, 1.6836114325399996, 1.2094043388736786, -1.4182467333580668, -3.141592657580788, 1.5707963438644246, -0.3196351889655169, 3.3259527783085643, 1.863412139832413, -4.232090008661287, 1.5707963365261095, 0.7853981624576323, 2.4278839139375754, -0.21281311992334284, 2.502151405410028, -0.7853981717750586, 5.180513682399239e-09, 0.7853981647339503, 1.033343924858469, 3.651581212374006, -2.9749135476725024, 0.7853981704224134, -3.1415926655821638, 2.103439575741413e-09, -1.570796325693223, -1.4527085744731505, -1.737475443182789, -0.7853981705003886, -1.5707963384955619, -1.7013223526147541e-09, -1.5707963344238567, -0.11808774591175483, 2.783635536755788, -6.283185317392107, 9.635321641201464e-09, 4.5568216027994415e-09, -2.9609090914433533, -0.09440169254800422, 1.9287534271087525, 1.570796325214528, 0.47904972253073963, 8.221620022005144e-09, -1.7514798905183246, 4.011028577013559, 3.141592645634192, 4.712388968124107, -2.0498460470669797, 3.141592639141935, 7.85398163128895, -4.7123889702423245]),
    (9, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3362794243636635, 0.8288174423076456, 1.2309594130292647, -1.8190493547162818, -0.799424515677198, 1.570796323664979]),
    (9, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.0616790053138483, -3.141592649283687, 1.681439841345574, 1.6351380831473525, 1.5707963296697243, -3.252236168570093, -2.7150517316199916, 0.33983690960082674, 2.83127918196056, 1.4461898933571045, 1.5707963268522276, -1.6003197646307106, 1.6954027522811723, -1.5707963294870269, 1.5707963262823494]),
    (9, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.9553166179696087, 3.61035443818762e-09, -1.5707963267903775, 1.5707963268467324, 1.3125906745652252, 1.570796324693049, 2.801755748788916, -0.3054985580292448, -2.267907291604686, 2.6479305328320812, -1.570796326741588, 1.5364579759256973, 2.7795191988298003, 1.5245016275719345, 1.5707963267503806, -4.030128055632309, 0.3620734566703084, -3.187887351038198, 3.14159265357671, 1.919374900427727, -2.9766722319240935, -2.157047851278967, 1.5707963254324602, 0.979254462128551, 2.976672230039933, -6.629591856691456e-10, 1.5707963268209466, 0.05158504505102767, -3.2577292835114413, -1.5727959484110952e-09, -1.5707963313876194, -2.073510613031172, 0.11613662962697612, -1.5707963266061158, 1.5707963267403529, -2.638878367527567]),
    (9, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.74818337958148, -7.846484335997964e-09, 3.14159257351821, 1.5707963874759805, 3.1415925900599455, -0.30606674325798866, 3.1415926472944333, -2.1572853473664027e-06, -3.1415903656509196, -2.729574107152622, 2.4232943064768806, 1.570796309986381, -1.570794142579581, -4.712391412297742, -0.7518554560654868, -0.12277032330382265, 2.526112944905079, 4.09690927171735, -1.57079541284025, -0.5042933310093146, 0.4147308173037127, 1.5707963269553222, -4.0806164721492375e-08, 2.356194479630151, 3.3060490753219094, 2.784515552156541, 1.5707956681373696, -3.681549473110016, 0.5399568289766049, 3.0353930174280976, 3.2290247667654444, 3.141593314109025, -4.712388207915789, 1.5707970619279716, 0.1061996401797174, -0.1450859162225332, -3.141592665982087, -1.5707956091500899, 1.5707957101394785, 2.706014686816974, -2.688066548840865, -6.283185249632289, 1.3253549772926136, -1.8162376553008115, -1.1352183300741554, 2.7717964655638987, 3.141592660940218, -0.7853981530830609, 2.0263743452160047e-08, 0.8481806268917357, -3.376244873730219, -8.824059727071894e-09, 3.1415923726661092, 1.5707963265579465, 3.8642083536818777, 0.15092239877273333, -3.141592573444156, -1.570796455263786, -1.5707963549874027, 7.853980889490922]),
    (9, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.4118649965778336, 3.141592527633889, -3.141592314257473, 3.1415914198854327, -1.5707975794145332, 1.5708304049124027, -2.5458715640286282, -4.712389280785085, -4.712389391061723, -2.527011238643, -1.5795865942168315, 2.8378385020835935, -1.3254487464118265, 3.8713212865926105, -0.07086254744606545, 3.1995194902563875, 1.2397496854056163, 3.1055098997604986, -1.3360708931155507, 1.5707953549842297, 0.911931225073754, 0.8982882496841029, 1.5708088113001757, -1.5012924298971444, 1.8632292566199655, -4.71238904053475, -2.407066598618577, 2.0642596368911916, 3.6815494507062803, -0.06953587192096365, 0.7067356824713793, -1.580235324723085, -2.2641978525665474, -0.4934633908199167, 1.0308394735936415, -0.5437002481138385, -0.4830839720780924, 0.5307047313269174, -1.6119204292740978, 4.712378573995498, 2.110753182958813, 3.6852929846924307, -2.2817573848962938, -2.0920621532541883, 5.497787156660723, 5.497787133045668, -2.6016358461258933, 2.900842933395718, -1.6106437163860154, -3.1415912671086317, -3.449332508164067, 2.663934352969156, 1.6110306873416106e-07, 0.24074991199529377, -3.3278519862231386, -3.141591259124314, -1.5707850774494052, -1.5708067503388197, -3.1415927967087955, 1.5361887135144643, -2.884057986811396, 0.06799862392313141, 4.7123989738164855, -1.5707865582618203, 2.491501256319552, 1.6028163476242177, -1.6794519093003928, -3.1415930655946256, -0.19862502579365587, -0.30773985085211925, -0.92070484312596, 2.3252286478605204, -1.46213901860937, -3.1415932276623706, 2.8625593600088286, 1.3991002254326521e-07, -1.570796323408394, 2.3897475794993137, 1.2302364147798217, 0.06799960350213356, 3.1415826691271653, -4.712388990486183, -8.968597662164484e-09, -3.187996635535275, -1.6169843230772178, -1.4403238705076012e-07, 1.57078580985941, 1.5707957743091292, -1.5707963262877211, -1.5243923470232699, -5.967712692995975, 3.141592550813532, 3.1415928223235925, -3.14159213204553, 4.7123890133742234, -1.5708083449977233]),
    (10, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3488808670096775, 0.8272607687864508, 1.249045770565431, -1.805720169749727, -0.8009671180962932, 1.5707963256669961]),
    (10, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.034443936180929, -3.141592645232987, 1.6814398425253279, 1.6215205495800442, 1.5707963303722394, -3.252236167390339, -2.7286692651873, 0.32175055462911134, 2.840322358803394, 1.4461898933555006, 1.5707963245934387, -1.591276587787879, 1.6954027522795703, -1.570796329450161, 1.5707963270345593]),
    (10, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [0.9911565868025957, -3.9146245040192376e-09, -1.5707963267909832, 1.5707963267200848, 1.2946706897205136, 1.5707963281607231, 2.8198421044904936, -0.2964553786281865, -2.2858272764493845, 2.634308941497357, -1.5707963267475427, 1.5455011553267735, 2.7795191982433325, 1.5245057754052933, 1.5707963086482921, -4.036936868265358, 0.36207345608382957, -3.187883207162176, 3.1415926536197647, 1.9125660877947135, -2.976672230305594, -2.1706613318739243, 1.5707963151321802, 0.9724457391106534, 2.9766722316584757, -9.810677525911164e-09, 1.5707963269019043, 0.04477632203309971, -3.2577292842820684, -6.156217724481243e-08, -1.5707963825972495, -2.07351061315182, 0.11613662885634245, -1.5707963270577288, 1.5707963271786347, -2.6388783676481977]),
    (10, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.725679150421451, -8.433165907531339e-09, 3.141592571347952, 1.570796384502187, 3.141592570080566, -0.32857097241802574, 3.1415926467077577, -2.1594556071324817e-06, -3.1415903897870883, -2.720530944400553, 2.40079007785746, 1.5707963036329597, -1.5707941447498428, -4.712391436433911, -0.7428122933134167, -0.14527455192324376, 2.5619528981393223, 4.1327492375026225, -1.570795258620601, -0.49525016220060986, 0.41473081499976927, 1.5707963366207034, -4.457418631705929e-08, 2.3561945046458366, 3.315092244130627, 2.7845155498525997, 1.5707956685504603, -3.695167032757759, 0.5535743368837128, 3.0353930209161213, 3.2290247644615047, 3.141593314522116, -4.712388044827314, 1.5707972283685192, 0.10619964366773368, -0.14508591852647676, -3.1415926653008492, -1.5707954502516588, 1.5707956116044257, 2.706014689394058, -2.688066551144811, -6.28318524895105, 1.3389725363834535, -1.802620167471479, -1.1352183274970655, 2.7717964632599497, 3.1415926609823797, -0.7853981567504279, 2.49543834813478e-08, 0.8481806306466274, -3.3762448760341632, -8.78189023303846e-09, 3.141592318547921, 1.5707963417455821, 3.864208357436766, 0.15092239646879113, -3.141592564518023, -1.5707964763928095, -1.5707963627489616, 7.853980727222223]),
    (10, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.3561944878861287, 3.1415923947093494, -3.141592184336502, 3.1415908314305363, -1.5707956578134699, 1.570855929270299, -2.573706817899682, -4.712389367739198, -4.712389561832005, -2.5270112036214067, -1.5705434196123782, 2.846881674581769, -1.3532840002823923, 3.8263083056623106, -0.04835832753869209, 3.19951952527812, 1.2487928600104643, 3.114553072258967, -1.3360715519986894, 1.5707998845272089, 0.9344354449806324, 0.8982882847057544, 1.5708142543213712, -1.5013048288961097, 1.8632285977378102, -4.712389084507961, -2.4070666844168005, 2.064259536254355, 3.6951670022960545, -0.0695482709197509, 0.7067350235883831, -1.5802353686965398, -2.2641979383651205, -0.4934634914559264, 1.017221946339472, -0.5437002031914812, -0.4830846309605834, 0.5307046873532834, -1.6119205150727725, 4.712371911481876, 2.1243707351009586, 3.685293029614024, -2.2817580437786624, -2.0920621972272606, 5.497787158158455, 5.497787129666409, -2.5880183156910457, 2.9008427320922774, -1.6106443752690833, -3.141597693739356, -3.4314125133559825, 2.6460143681778256, 2.4292484769698076e-07, 0.24074971069141587, -3.325869435410485, -3.1415976949244047, -1.5707786293536443, -1.5708133845515035, -3.141592933705808, 1.5361888796922292, -2.8820754359989125, 0.06819865413462754, 4.712405843950528, -1.5707805189409199, 2.4915012933328176, 1.6028165138026813, -1.6794538228545066, -3.141590802657933, -0.20758501739049517, -0.28981988286075494, -0.920704806112774, 2.325228814038373, -1.462140932163623, -3.141590964725394, 2.853599368411741, 2.5172829242651904e-07, -1.5707963232023252, 2.3897477456764116, 1.2289147339409427, 0.06819511863569508, 3.1415757210151196, -4.71238900204379, -3.0850341634745203e-08, -3.187996633479284, -1.6183060039153576, -1.8693170139833828e-07, 1.5707798878712402, 1.5707955471989565, -1.5707963294848848, -1.524392344967457, -5.969034373834367, 3.1415925079140052, 3.1415927299369977, -3.1415919445253695, 4.712389010724299, -1.5708157598196497]),
    (11, 1, False): ([[(0, 1)], [(0, 1)]], [-1.359645604477575, 0.8260018033669561, 1.2645189584098586, -1.7943326514547893, -0.8022157877797601, 1.5707963266832448]),
    (11, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [2.011306988958161, -3.1415926318810947, 1.6814398459649167, 1.6099520768357325, 1.5707963297197503, -3.252236163950751, -2.74023773793161, 0.30627736934417904, 2.848058951171418, 1.4461899016315984, 1.5707963214559186, -1.5835399954198532, 1.6954027605556699, -1.570796326608598, 1.5707963309386355]),
    (11, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.021329081381551, 2.8290014786839912e-09, -1.570796326701034, 1.5707963268597185, 1.2795844409178376, 1.5707963255610202, 2.835315278396422, -0.28871878935973433, -2.3009135252520214, 2.6227376093244152, -1.5707963267203822, 1.5532377445952181, 2.7795191980509846, 1.5245085568885963, 1.570796319904736, -4.042721066966055, 0.3620734558914699, -3.187880430291581, 3.1415926535778778, 1.9067818890939905, -2.9766722287797878, -2.182226958184388, 1.5707963277694967, 0.966661463897588, 2.9766722331842748, -6.974423698891025e-09, 1.5707963268131797, 0.03899204682003281, -3.257729283023431, -1.454295564333775e-08, -1.5707963336267938, -2.073510612697557, 0.11613663011497027, -1.5707963264988842, 1.5707963269056788, -2.6388783671939318]),
    (11, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.706961974618802, -3.894353077411057e-09, 3.141592570107976, 1.5707964261881553, 3.141592579883285, -0.3472881482206599, 3.141592651246574, -2.160695581871293e-06, -3.1415903903700544, -2.712794339308213, 2.3820729011763384, 1.570796300183105, -1.5707941459898127, -4.712391437016875, -0.7350756882210764, -0.16399172860436426, 2.592125413312606, 4.16292173603521, -1.5707951503509257, -0.48751355997914725, 0.41473081650836857, 1.570796330677729, -4.37338650655677e-08, 2.356194487781915, 3.3228288463520856, 2.784515551361203, 1.5707956684433286, -3.7067354825827823, 0.5651428400304933, 3.035393016001225, 3.2290247659701143, 3.1415933144149846, -4.7123879718121575, 1.5707972855154007, 0.10619963875284197, -0.14508591701787857, -3.1415926652462183, -1.5707953713967948, 1.5707955613071873, 2.7060146892837738, -2.6880665496362095, -6.283185248896408, 1.3505409961829453, -1.7910516591388006, -1.1352183276073415, 2.7717964647685522, 3.1415926580521236, -0.7853981688222343, 2.6268186312393338e-08, 0.8481806285098648, -3.376244874525564, -1.1712150765003647e-08, 3.1415923008440814, 1.5707963221931605, 3.8642083553000033, 0.1509223979773943, -3.1415925630840813, -1.5707964810932735, -1.5707963661223339, 7.853980651896173]),
    (11, 5, False): ([[(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)]], [-0.8309155496707723, -3.14159264874557, -1.5707962961909459, -0.13530641728850568, 3.1415926356329806, 1.570796254313455, -0.18291428316265804, -4.220135125911035, -3.1415926888872403, 1.4354906998883485, 0.2568792369404802, -0.3322273734259396, -2.543220312302015, -3.453359766464939, 0.20524631769888457, 2.756319188257261, -3.8236828649142947, -1.2385688591523958, -3.4555610920938706, -0.611744389426383, 1.953213809215219, 1.461749851429764, -1.2248375732985128, -3.3768362088416826, -0.10148922991601593, -0.1684707603822693, -2.737345181306518, -2.362126237863659, 1.9344055907584006, -2.906349104074167, 1.0190449841999305, 5.169862598575636, -0.785398170340539, 3.141592667216413, -3.8431178460690174, -0.9584083307090625, -1.434502922160717, 3.1415924215863753, 1.8455299417374504, 0.903952987525953, -1.652238624207045, 1.6973422594889263, -1.2268013168343643, 2.3873663121744843e-07, 2.318894812611401, 1.5707963237934905, 1.4893537898834939, -0.7287306465782484, 0.8113433075911567, 1.5707962829031021, 2.3188947188010256, -1.5707963896493211, -3.0891009640404264e-08, -0.7956014254523863, 1.8487927043213566, -4.712388949235682, 5.223053540680872, -3.8084360015541283, -1.5707963483902028, -1.9412474521777408, 0.542609627895333, 3.6651912696034286, -1.5707963055545504, 1.036083732061563e-09, -1.5707963318400449, -3.5565396905729716, -3.485902633983327, 0.52359882807049, -2.496131145073723, 1.5707963147683972, 3.1415926597060366, 1.664067793545077, -2.0470923564702495, 0.5235988690017701, 1.758147110476161, -1.5707963154110427, 1.5707963166823404, 3.0483211996893274, 1.767390442007569, 1.5707963163823224, 1.8210384961049617, 2.4881318755367976e-08, 3.9269908843238626, -2.652841905130788, 1.3742022082206091, 3.027035184719727e-08, 4.12981878882625, -1.5952832997666153e-08, 2.35619455956882, 1.0820455894174656]),
    (12, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3689807695775276, 0.8249626574879323, 1.2779535573016527, -1.7844566110680178, -0.803247374204487, 1.57079632786389]),
    (12, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9913306575342296, -3.141592637694843, 1.6814398423111587, 1.5999639099265046, 1.5707963336038868, -3.2522361676045097, -2.7502259048408386, 0.29284276322944336, 2.8547762507800094, 1.4461898872978378, 1.570796320027087, -1.576822695811264, 1.6954027462219092, -1.5707963273065848, 1.5707963274798569]),
    (12, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.0471975482817304, -2.287737980266337e-09, -1.57079632749725, 1.5707963245730612, 1.2666502061733889, 1.5707963258621218, 2.848749878728925, -0.2820014888088393, -2.3138477599964773, 2.6127474227496013, -1.5707963287353992, 1.5599550451461555, 2.7795191969331148, 1.5245105618264185, 1.5707963297285719, -4.047715136598987, 0.36207345477355757, -3.1878784151126136, 3.141592652116615, 1.9017878194610576, -2.9766722303379365, -2.1922130799254633, 1.5707963230135562, 0.9616673732896535, 2.976672231626051, -6.09719643252126e-09, 1.5707963221037264, 0.033997956212165864, -3.257729283038919, -3.96343984307873e-10, -1.570796328951715, -2.073510612722336, 0.11613663009946576, -1.570796325693153, 1.5707963314181423, -2.6388783672187195]),
    (12, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.6910594063830375, -5.477513659996493e-09, 3.1415925621600835, 1.5707964581286302, 3.1415925806378073, -0.36319071645643675, 3.1415926496634135, -2.168643472707877e-06, -3.141590404999487, -2.7060770349859897, 2.366170333061409, 1.570796291775319, -1.5707941539377037, -4.712391451646304, -0.7283583838988495, -0.17989429671929097, 2.617993879104451, 4.188790203595653, -1.570794985724501, -0.48079626099277095, 0.4147308169209157, 1.5707963269355776, -4.4788989185899815e-08, 2.356194487787165, 3.329546145338457, 2.784515551773751, 1.5707956700978873, -3.716723648884822, 0.5751309945286572, 3.0353930130226243, 3.2290247663826666, 3.141593316069539, -4.712387847151195, 1.5707973998850349, 0.10619963577424377, -0.14508591660533052, -3.1415926668049154, -1.5707952331658044, 1.5707954619736897, 2.7060146867837336, -2.6880665492236564, -6.2831852504551104, 1.360529156912404, -1.7810634933033391, -1.1352183301073728, 2.771796465181097, 3.1415926618499728, -0.7853981612787734, 2.7316325796385587e-08, 0.8481806225666289, -3.3762448741130195, -7.914304497039984e-09, 3.141592261897922, 1.5707963353376264, 3.8642083493567707, 0.1509223983899404, -3.1415925438821675, -1.570796493092498, -1.5707963686525677, 7.853980519151064]),
    (12, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (4, 5)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (4, 5)], [(1, 2), (3, 4)]], [0.7297276214303231, -3.1415926964056817, -3.141592632581924, 3.1415926117983703, 1.570796327461837, -0.42053432729754386, 1.7009731833665447, -1.5707962861131808, 1.5707962766462642, 3.1415926071048728, -1.5707963291865261, -1.433333789931961, 2.2400443828197227, -3.7570723317117296, -2.1862760334646048, -4.712388896964051, -4.4195462020865595, 3.0041301347034466, -1.2599525707329402, 3.1415926288471367, 1.5707962938751716, -1.047197511083736, -4.291854641037643, -1.1549806730327914, -1.9513373712140687, 3.1415926447921314, 2.094395204780384, -3.555639915270491e-09, -1.2897614211554367, -3.557408291730901, 7.853981287151208, -1.047197505852707, -2.0943951436485446, -3.1415926493431687, 0.9113047361716651, 1.9634954457812552, 2.7184697893999066, 3.1415926308293924, -1.5707963108230623, -2.0152041803806497e-09, 0.0069053907271297505, -5.204826738562552, 3.819585396394412, -0.2548698833796674, 1.5707963254163886, 3.1415926573644097, 3.0087806612632573, -5.356333430194041, 1.8956843836963504, -1.570796029483958, -1.5707963517066186, -0.7853981962945406, -8.288766909104626e-09, -1.8006537253777737, -1.8956850565441323, -4.712388973414433, -1.5707963337775053, 1.5707963515915586, -1.658846373024138, 0.22985739158528712, -1.0924144692307394, -2.8867227745320707, -3.1415926610168197, 3.1415926385393296, -4.419546201483893, 1.8294570075813898, -2.304048062529405, 3.5211155275067313e-09, -2.356194590400482, 8.462844655062569e-09, 0.8271131774344465, 1.3121356606969596, -2.7411970990424175, 1.5707963776837681, -5.497787036678776, 4.218195840076371e-09, 1.5707963067214656, -1.214362712964122, -1.9711921266310461, -1.5707963051271727, -1.5707960999523884, -1.2897614050855222, -1.65884637654572, -0.3564336319084016]),
    (13, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3771773061262595, 0.8240902780048295, 1.2897614253721392, -1.7757845273852257, -0.8041139037868821, 1.5707963268513274]),
    (13, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9738534009759419, -3.141592645880443, 1.6814398413672589, 1.5912252801620637, 1.5707963321724288, -3.252236168548411, -2.758964534605281, 0.2810349104231624, 2.8606801845771246, 1.4461898869799221, 1.5707963256711486, -1.57091876201415, 1.6954027459039962, -1.570796331870021, 1.5707963254705688]),
    (13, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.0697033044903945, 3.792591564937326e-09, -1.5707963265962894, 1.570796327328663, 1.2553973271529548, 1.570796325844533, 2.8605577819858334, -0.2760975496301068, -2.3251006390169096, 2.6040103408385735, -1.5707963271121428, 1.5658589843248776, 2.7795191967633657, 1.5245090042691591, 1.5707963351478016, -4.052084443859599, 0.36207345460385765, -3.187879967700934, 3.141592653891526, 1.8974185122003535, -2.976672238666549, -2.200953235344855, 1.5707963221747816, 0.957298049221968, 2.9766722232973817, -2.5021470692013266e-09, 1.570796326373183, 0.029628632144397263, -3.2577292818607644, 5.154887481780438e-09, -1.5707963309927835, -2.0735106122319857, 0.11613663127765565, -1.5707963263471192, 1.5707963260917452, -2.6388783667283744]),
    (13, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.6773208530339145, 1.2682982890411549e-08, 3.1415925646608476, 1.5707965118983673, 3.141592598210804, -0.37692926980555963, 3.141592667823922, -2.1661427091417303e-06, -3.1415904113774507, -2.700173102005489, 2.3524317799656966, 1.5707962739141528, -1.5707941514369395, -4.71239145802426, -0.7224544509183468, -0.19363284981500714, 2.6404996402862353, 4.211295966015764, -1.5707948595391765, -0.47489232542529136, 0.4147308131846311, 1.5707963275998482, -4.293070280792139e-08, 2.3561945015245973, 3.3354500809059333, 2.7845155480374575, 1.5707956613927203, -3.7254622680859493, 0.5838696139815004, 3.0353930061630097, 3.2290247626463784, 3.141593307364384, -4.712387750824551, 1.5707974777465377, 0.10619962891463054, -0.14508592034161777, -3.1415926620313672, -1.5707951295440614, 1.5707954080764288, 2.706014674857204, -2.688066552959937, -6.283185245681551, 1.3692677883639373, -1.7723248653460841, -1.1352183420339006, 2.7717964614448083, 3.141592661782931, -0.7853981518441354, 2.5817397056172697e-08, 0.8481806255821893, -3.3762448778493126, -7.981357733262099e-09, 3.1415922427953533, 1.5707963265214877, 3.864208352372326, 0.15092239465365573, -3.1415925563301137, -1.5707965118627445, -1.5707963610978273, 7.853980420793716]),
    (13, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.239760391705169, 3.1415927593448285, -3.141592805419368, 3.14159277100287, -1.5707951475483455, 1.57077976911738, -2.6319238658359128, -4.7123889627969415, -4.712388876878978, -2.5270113950202484, -1.5501855921837522, 2.8672395058805678, -1.411501048218879, 3.7295966051524188, -2.3243603189864482e-08, 3.199519333879617, 1.2691506874403538, 3.1349109035589464, -1.3360709911263475, 1.5707949815069335, 0.9827937492771167, 0.8982880933070186, 1.5707907786653765, -1.501268547121479, 1.8632291586109782, -4.712388969454212, -2.4070655360505593, 2.064260156962968, 3.725462312856251, -0.06951198914472917, 0.7067355844599894, -1.5802352536418323, -2.264196789999132, -0.49346287074711764, 0.9869267201205586, -0.5437003779910207, -0.48308407008786575, 0.5307048024079918, -1.6119193667066334, 4.712391308186535, 2.154665924926349, 3.685292854812955, -2.2817574829059457, -2.0920620821726525, 5.497787108415857, 5.497787149527884, -2.557722994341794, 2.900842779791413, -1.6106438143966484, -3.141590795252503, -3.392139195860771, 2.606740993884004, 2.261360278630306e-07, 0.2407497583894689, -3.3183598219150072, -3.1415907757947332, -1.5707855192247462, -1.570793410135262, -3.141592794196774, 1.5361888307721436, -2.8745658225033353, 0.06926541292992588, 4.7123868476062825, -1.5708000738245094, 2.491501318158994, 1.60281646488206, -1.6794522795619042, -3.141593227697131, -0.2272216805577288, -0.25054650986297944, -0.9207047812847805, 2.32522876511957, -1.4621393888718903, -3.1415933897646924, 2.8339627052452165, 2.094972572753169e-08, -1.5707963157749616, 2.389747696760189, 1.2239082959058731, 0.06926672499959817, 3.1415956136883563, -4.712388973098454, 4.813492633287407e-09, -3.187996631605237, -1.623312441948872, 1.9035578602163647e-08, 1.57078417983729, 1.5707962003947404, -1.5707963179039361, -1.5243923430933326, -5.974040811868812, 3.1415927138815363, 3.1415934455584265, -3.141592651391824, 4.712389001295918, -1.5707921161240779]),
    (14, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3844494436241623, 0.8233474891201374, 1.3002465657025088, -1.768089972335896, -0.8048520733355694, 1.5707963266214846]),
    (14, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9583930138198578, -3.1415926445978886, 1.6814398425548254, 1.5834950867006599, 1.5707963300638148, -3.2522361673608464, -2.7666947280666867, 0.2705497653394672, 2.8659227553038273, 1.4461898927569714, 1.5707963269861747, -1.5656761912874475, 1.6954027516810468, -1.5707963299206738, 1.5707963269871645]),
    (14, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.0895209518028461, -2.006126692183694e-09, -1.5707963266770442, 1.570796326625493, 1.2454885072159396, 1.5707963278283348, 2.8710428959482432, -0.2708549850093781, -2.3350094589539663, 2.59628132330302, -1.5707963266763418, 1.5711015489457338, 2.7795191971768345, 1.524507833899254, 1.570796325803807, -4.055949549059083, 0.36207345501723964, -3.1878811442399906, 3.1415926535034697, 1.8935534070008158, -2.976672230991362, -2.208684631782823, 1.5707963274028325, 0.9534329600953768, 2.976672230972561, -2.7678982455346344e-09, 1.5707963269258907, 0.025763543017837526, -3.257729283090427, 2.1150160461803413e-09, -1.570796326297687, -2.0735106127712952, 0.11613663004791681, -1.5707963268610958, 1.570796327203698, -2.638878367267597]),
    (14, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.66529087251781, -3.943714507450022e-09, 3.14159256672785, 1.5707964689960385, 3.1415925917764427, -0.38895925032166845, 3.1415926511972208, -2.1640757007360535e-06, -3.1415903989653318, -2.694930533121339, 2.3404017992110124, 1.5707962971491447, -1.5707941493699318, -4.712391445612145, -0.717211882034194, -0.20566283056968918, 2.6603172798111694, 4.231113606517928, -1.5707949799679415, -0.46964975764083444, 0.41473081670322404, 1.570796326717281, -3.8697312150268826e-08, 2.3561944900529013, 3.340692648690398, 2.7845155515560487, 1.5707956686949613, -3.7331924732711235, 0.5915998205865481, 3.035393011914538, 3.2290247661649696, 3.141593314666618, -4.712387835556736, 1.5707973958307138, 0.1061996346661606, -0.14508591682302244, -3.141592670657187, -1.5707952215304366, 1.5707954723106563, 2.706014685709386, -2.6880665494413423, -6.283185254307362, 1.376997983465929, -1.764594669923955, -1.1352183311817152, 2.771796464963402, 3.141592661602161, -0.7853981631164108, 2.5993310453960164e-08, 0.8481806258289745, -3.3762448743307174, -8.162135578127076e-09, 3.1415922798617175, 1.5707963284131068, 3.864208352619115, 0.1509223981722466, -3.1415925616887788, -1.5707964871557438, -1.5707963684466115, 7.853980506536657]),
    (14, 5, False): ([[(0, 1), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(1, 2)], [(4, 5)]], [0.9302740077210688, 4.703700437865649e-09, -0.5907423771192686, -1.7839224002444989, 2.0228260556167323, 3.14159264287724, -2.1131118899490238, 4.712388977936777, 2.1615386749859042, 0.21312603125973364, -1.570796376038179, -0.38639412981006227, 1.1828378816808467, 3.7055352892334414, -4.1484463321537595, 0.7516326708135612, 1.9583930781105625, -2.7551985158886403, -0.12309696474647114, -4.712388972558349, -4.4351715485291345e-09, 1.1601662051175606e-07, -1.570796273000144, -2.567214575353038, 2.8082676640256694, 1.0236079366312232, -3.9716949464312377, -0.6614359515747497, -2.555002425746361, 2.4319396891561116, 0.45642200266619165, 0.547188361557319, -1.0260358590388783, 1.7348984322840886, -1.1206977792248605, 0.3445586545477685, -0.8436718267674369, -1.5886155966322343e-09, 1.3939950731446378, -4.712388895368881, -2.020894786232354, -2.7307984024132934, 3.9435847833208695, 1.133686175796428e-07, 0.32508121521296274, -1.5707962645975222, -1.8067667436453987, 1.0863975307962577, -2.721935022650626, -1.0985624875911133e-07, 0.24063769708029006, 0.8423183647351621, -2.9056222442774553, -2.86398927430525, -1.3876066138366603, 0.9688557058838178, 2.356194502226287, -5.343098055434735e-09, -1.5707963399554208, -0.6838323679376742, -0.6974276400498773, -0.8863374532916649, 3.1433502651125638, 1.5707963276571801, -5.5529495398998805e-09, -1.3326436582992733, -2.6560367180836586, -0.08251818094917478, -2.013267478039115, 1.5707963398103895, -1.5707963066340038, -0.23815266883642383, 1.1790940139469597, 1.9396984751697899, 0.2562998140726224, -3.1415926794348796, 1.4355214412548993, 0.2538754219297182, 1.4072641649127768, 1.201894114540167, 0.873783160060808, 3.14159257793732, 2.9755951552923836, -3.491565667576432, -1.8479920640559189, 3.1415925974059538, 1.0706050869188612, 0.18277215505698174, 3.442865029847501, 0.09609761018634198, 0.48313422492030805, -3.1415925989168323, -1.291327412942891, -0.1827720929053387, 1.868786643614875e-08, 4.712388896674768]),
    (15, 1, False): ([[(0, 1)], [(0, 1)]], [-1.3909589032690937, 0.8227074029892689, 1.3096389159210724, -1.761202002534714, -0.8054884477937535, 1.5707963268353513]),
    (15, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9445885019389597, -3.141592645002856, 1.6814398435215172, 1.5765928310206765, 1.5707963289152784, -3.252236166394156, -2.773596983746668, 0.261157411072487, 2.87061893144989, 1.4461898954678418, 1.5707963284342483, -1.5609800151413846, 1.6954027543919181, -1.570796328099931, 1.5707963276240389]),
    (15, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.1071487179537243, 1.5455948895814684e-09, -1.5707963272745071, 1.5707963265213027, 1.2366746241008735, 1.5707963263700762, 2.8804352418751145, -0.26615880922548885, -2.343823342069099, 2.5893799890709848, -1.570796326755805, 1.5757977247296207, 2.7795191981697567, 1.5245069136041534, 1.5707963271443501, -4.05940067873059, 0.36207345601028196, -3.1878820672646295, 3.1415926533349072, 1.8901022773293021, -2.9766722309288514, -2.215587814934759, 1.5707963269247323, 0.949981833134305, 2.9766722310350016, 7.456143978844525e-11, 1.5707963268082143, 0.022312416056573723, -3.2577292834312033, 2.9598925862437583e-12, -1.5707963265306093, -2.073510613625593, 0.11613662970712765, -1.5707963270253167, 1.5707963262040725, -2.6388783681218735]),
    (15, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.654639103098011, -3.347747885411105e-10, 3.1415925621946146, 1.5707964507974426, 3.1415925996816814, -0.399611019741456, 3.141592654806148, -2.1686089310652687e-06, -3.141590391448078, -2.6902343576503527, 2.329750029750611, 1.5707962896954417, -1.5707941539031602, -4.712391438094886, -0.7125157065631991, -0.21631460003009037, 2.677945044929563, 4.2487413715482765, -1.570795101965015, -0.46495358276213056, 0.41473081576572857, 1.570796324257339, -3.334685523887552e-08, 2.3561944884162127, 3.345388823569105, 2.7845155506185506, 1.5707956667458336, -3.7400947338489927, 0.5985020738327836, 3.0353930123043886, 3.2290247652274693, 3.1415933127174878, -4.712387931737874, 1.5707973030874165, 0.10619963505601135, -0.1450859177605154, -3.1415926745663616, -1.5707952961965463, 1.5707955425059559, 2.706014682138693, -2.6880665503788426, -6.283185258216529, 1.383900236429805, -1.7576924125441082, -1.1352183347524234, 2.7717964640258987, 3.141592663326701, -0.7853981589601734, 2.1668792874351845e-08, 0.8481806273910713, -3.3762448752682137, -6.437598765796574e-09, 3.141592313887838, 1.570796321310567, 3.8642083541812067, 0.15092239723475506, -3.141592559768154, -1.570796466026187, -1.5707963609996447, 7.853980591663908]),
    (15, 5, False): ([[(4, 5)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(3, 4)], [(4, 5)], [(3, 4)], [(4, 5)], [(3, 4)]], [-0.8177065888844087, -2.7015970146788635, 3.1415926311955986, 1.8025695740928373e-08, 1.5707963216967116, -0.3737921744646206, 1.7730232059361184, 5.843189695951172, -0.1652028425191549, -0.398682925885038, -0.6624315399806526, 2.596038531609348, 2.2528935992356742, -4.712389080696102, -1.4055935191365445, 0.3986829063562616, -0.9083647915761278, -0.17732909567572053, 1.8440156722087453, -0.5426391534837175, -2.1134354482577136, -1.5707965423874286, -1.3096389363671874, -0.8479131470466658, 0.643336066221612, 1.5707962913657814, 3.1415926585681553, 0.23182381710478095, -1.8582235129038895, 2.6433943232480197, -2.5357238172284444, -2.870273122820647, 3.3734164503006916, 3.141592652547295, -3.141592628525677, 1.2791422619303905, -0.6283644650047022, 1.2994767585606044, -4.712388902403707, -3.0794055341899853e-09, 3.1415926324157737, 0.9514591358502199, 0.4659877290136949, 3.141592393173447, 1.5707963186644842, 3.141592848963851, 2.4803734297992857, -3.303199390913076, -1.0974737851019365, -3.141592521621844, -3.5156488237420764, -4.4805651722240425, 1.4492992201255292e-08, -2.68541322786826, -3.7661551607254293, -3.1415926469448516, -0.9649163373882843, -5.673179505968342e-09, -0.7853982025595831, -3.5977720543081944, -0.20103105680733058, 0.00566525892362839, 3.458478381919039, -0.7853980841129313, 1.8234765594374132, -3.7009219906980686, 0.44601767585196855, 3.1472595847571725, -1.8876816985093, -1.5707962824646968, 3.14159267239086, -1.0114668859642686, -1.160233675038211, 3.1302606357353047, -3.201647316350281e-07, 3.1415927116663735, -1.8319537160934243, 0.5704084644253993, -1.5911375092817188, 3.1415925664553312, 2.1126305180918283, 4.965069350032525, 0.8187562870752721, -2.8932449417118513, 0.24134652715752006, 3.1415944247997127, -1.145610437996983, 1.3181159251605212, 1.5707963128319231, 0.2850191275037668, -0.2413465065767634, -1.6673346326674416e-06, 2.1745725749395737, 1.8234765617596147, 3.14159279361948, 1.2857772320763805]),
    (16, 1, False): ([[(0, 1)], [(0, 1)]], [-1.396830358471015, 0.8221500918172666, 1.3181160716581939, -1.7549888182555455, -0.8060427320206347, 1.5707963268288248]),
    (16, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9321634509001153, -3.1415926461652663, 1.6814398434627535, 1.5703803060840011, 1.5707963283240325, -3.252236166452919, -2.7798095086833428, 0.2526802553120026, 2.8748575088452277, 1.446189896443132, 1.570796327764351, -1.5567414377460476, 1.6954027553672055, -1.5707963275172632, 1.5707963276568375]),
    (16, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.1229639300019953, 1.6643459389793535e-09, -1.5707963267602623, 1.5707963270019494, 1.2287670155715356, 1.5707963262610816, 2.888912394141669, -0.26192023893730654, -2.35173095059849, 2.5831681419835917, -1.570796326915663, 1.5800362950179045, 2.779519197733498, 1.5245061335700483, 1.5707963384893602, -4.0625068932182815, 0.3620734555740693, -3.1878828504170795, 3.141592653713425, 1.886996062841625, -2.976672230297107, -2.221801023374271, 1.570796340697235, 0.9468755217725572, 2.976672231666839, 1.677174361219939e-09, 1.5707963266484604, 0.019206104694649815, -3.257729283003, 4.5309786749489524e-08, -1.5707962890177842, -2.073510612781338, 0.11613663013523695, -1.5707963266501128, 1.5707963264994007, -2.638878367277598]),
    (16, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.645118933308322, -1.477318811944053e-09, 3.141592567850351, 1.5707964180554952, 3.141592600267703, -0.4091311895311347, 3.141592653663595, -2.1629531981176094e-06, -3.1415903845126554, -2.685995788211049, 2.3202298594634825, 1.5707963006272674, -1.570794148247429, -4.712391431159456, -0.7082771371238934, -0.2258347703172227, 2.6937602547360853, 4.264556571150523, -1.5707951857654168, -0.4607150028987943, 0.41473081733816386, 1.5707963156556892, -2.97430249611427e-08, 2.356194480612328, 3.3496274034324376, 2.7845155521909764, 1.5707956683524036, -3.7463072653266565, 0.6047146309836249, 3.0353930159738796, 3.229024766799899, 3.1415933143240515, -4.712387973005604, 1.570797259396153, 0.10619963872550103, -0.1450859161880789, -3.141592673504324, -1.5707953462610331, 1.570795587076018, 2.70601468189037, -2.688066548806405, -6.283185257154494, 1.3901127856007132, -1.7514799190388264, -1.1352183350007463, 2.771796465598324, 3.141592652616904, -0.7853982204624363, 2.020315216436058e-08, 0.8481806373459261, -3.376244873695777, -1.7147408064638986e-08, 3.1415923388305855, 1.5707963035523556, 3.8642083641360636, 0.15092239880718625, -3.1415925763605, -1.5707964594768875, -1.5707963592641507, 7.8539806373896]),
    (16, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(1, 2)], [(0, 1), (2, 3), (4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(3, 4)], [(0, 1), (4, 5)]], [1.5707963326882795, 4.712389047095583, -3.1415926452127483, 6.2831853040388985, 1.5707963563564342, -4.783811280204081e-09, 3.5410713498063195, 3.72282411494036, 3.3225742897034602e-09, 4.712388973286948, 1.570796350484416, 7.926473456288015, 0.2829319381530003, 2.9994116002848967, 6.283185308958077, 5.983530568273209, 3.141592521276496, 4.387216918091621, 4.2704728697538865, 2.2365761539337052, 6.283185299519427, -1.7638035390971722, 4.760323472256777, -2.2617195627903883, 1.784299579685881, 1.967161416031111, 1.5707962520778966, 0.4926619421765004, -2.5848125805105697, -0.6271928122363309, 0.12087396706902688, -1.1043679715590051, -2.989151808546806, 1.269876456859845e-08, -2.5368781142092605, -2.369258182661241, -0.5615255646076576, -2.6317147000336902, 1.4183554361192523, 5.8353529057307005, 0.6047145381168522, -0.7723344367093948, -0.8812162636700113, -4.117898901405563, 1.5707963636094449, 0.5614819834295693, -8.45869622585598, 1.1421105688565083, -0.2489285077501784, 8.639379773612129, -1.5707963059384635, -0.5614819516201521, 1.570796318693671, 3.2561191059865884, -1.8521221884109451, -7.068583494626183, 6.283185333530639, 2.5801106881387836, -1.5707962761777956, 2.214336836334175, -6.903165300742154, 3.926990843395746, 17.27875958700634, 7.292499662692995, 7.853981693757997, 3.3062635297736596, -3.811083124350612, -12.566370628485393, 6.283185304980066, 3.141592608319103, -1.5707963376074365, 1.0767442810052712, 4.0943289311463, 10.21017612471036, 7.85398163137902, -4.712388985741156, -4.172779661368957e-09, -2.1809028321306854, 3.759652656652197, 1.4336106026124183e-08, 1.4809381927529568e-08, -4.712389006514364, 1.5707963272550407, 3.7516991442834615, 4.712388941807937, -3.141592661743596, 3.1415926296421754, 2.480595913180179e-08, -1.5707963287764986, 3.141592681320604]),
    (17, 1, False): ([[(0, 1)], [(0, 1)]], [-1.4021616902902747, 0.821660470626402, 1.3258176636612564, -1.7493469367106214, -0.8065298522222192, 1.570796326813298]),
    (17, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9209021045154797, -3.1415926489371904, 1.6814398422614074, 1.5647496325361754, 1.5707963282076478, -3.2522361676542624, -2.7854401822311674, 0.24497866224428375, 2.8787083054975815, 1.4461898952475947, 1.5707963280335104, -1.5528906410936945, 1.695402754171667, -1.5707963282626252, 1.5707963269764054]),
    (17, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.137258417342416, -6.853522956645567e-10, -1.5707963267730254, 1.5707963266672782, 1.2216197744822295, 1.5707963271590268, 2.896613988894931, -0.25806943520675096, -2.3588781916878214, 2.5775381099058405, -1.5707963267162597, 1.5838870987486124, 2.7795191975892792, 1.5245055945577886, 1.5707963275509673, -4.065322278115665, 0.36207345542962704, -3.187883386898151, 3.1415926535073004, 1.8841806779443488, -2.976672230731083, -2.227432333191875, 1.5707963270603693, 0.9440602338608128, 2.9766722312328207, 3.2091526288604783e-10, 1.5707963268763834, 0.016390816782893666, -3.257729283226966, -4.0620283617439016e-10, -1.5707963264972058, -2.0735106129219436, 0.1161366299112451, -1.5707963268440848, 1.570796326979268, -2.6388783674183194]),
    (17, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.636541872123857, -1.113609857849455e-09, 3.1415925642741063, 1.5707964029988917, 3.1415926092250426, -0.41770825071557743, 3.1415926540272876, -2.1665294514681163e-06, -3.1415903816093738, -2.6821449817711867, 2.3116527986980944, 1.5707962952733754, -1.57079415182368, -4.712391428256173, -0.7044263306840302, -0.2344118310826128, 2.7080547436294924, 4.2788510673174835, -1.570795270045712, -0.45686420536559696, 0.41473081621904995, 1.570796322024211, -2.6477307882876094e-08, 2.3561944727895776, 3.353478200965639, 2.7845155510718658, 1.5707956672142944, -3.751937919551934, 0.6103452821549744, 3.035393014139828, 3.229024765680778, 3.1415933131859273, -4.712388049764459, 1.5707971834987766, 0.10619963689143713, -0.14508591730718973, -3.141592678075722, -1.5707954062008846, 1.5707956404623278, 2.706014681035412, -2.688066549925515, -6.28318526172588, 1.395743426895111, -1.745849211771274, -1.1352183358557009, 2.7717964644792104, 3.141592659829417, -0.785398156149669, 1.66765851059918e-08, 0.848180624762178, -3.376244874814885, -9.934902148709628e-09, 3.1415923649150153, 1.5707963294942968, 3.864208351552305, 0.15092239768807728, -3.141592568200231, -1.5707964436435844, -1.5707963501067888, 7.853980703789778]),
    (17, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.143999638388394, 3.141592410220003, -3.1415925743859945, 3.1415927589564143, -1.5707887962887586, 1.5707980911902344, -2.679804243340385, -4.712389232923743, -4.712389334422656, -2.527011372286997, -1.5321574664439317, 2.8852676235339607, -1.4593814257230868, 3.6480345543559127, 0.040778956282478636, 3.1995193566134805, 1.2871788131795088, 3.152939021212232, -1.3360712891677025, 1.5707990691785167, 1.0235727288055578, 0.898288116040663, 1.5707894257444237, -1.5012778759330998, 1.863228860568266, -4.712389066505099, -2.4070664225624134, 2.0642598527400136, 3.7519379910338886, -0.06952131795640772, 0.706735286419023, -1.58023535069276, -2.264197676511219, -0.49346317497005865, 0.9604511120042176, -0.5437002843044406, -0.48308436813009875, 0.5307047053570554, -1.6119202532188104, 4.712388262616848, 2.1811415446187814, 3.685292948499011, -2.2817577809469936, -2.0920621792237672, 5.497787170637552, 5.497787126262995, -2.5312473145763574, 2.9008422322518754, -1.6106441124379491, -3.141596520593061, -3.358361583216086, 2.572963438954181, 9.222543283235522e-08, 0.24074921085148576, -3.323615429609295, -3.1415965141707747, -1.5707917235286744, -1.5707968136195392, -3.1415927674399184, 1.5361892333888305, -2.8798214301971092, 0.06879183016453375, 4.712391101065712, -1.5707964438383737, 2.491501259892125, 1.6028168674992667, -1.6794530229132572, -3.1415912133424753, -0.24411047861752458, -0.21676897163765727, -0.9207048395518445, 2.325229167736953, -1.4621401322234966, -3.141591375410024, 2.8170739071847595, 3.388522825854958e-08, -1.5707963303669203, 2.3897480993757036, 1.2274120557028123, 0.06878908963775511, 3.1415907734797974, -4.712388986983394, -2.8231821314582524e-08, -3.1879966391023613, -1.6198086821530382, -7.40898662106572e-08, 1.5707916134332813, 1.5707963414433381, -1.570796336543584, -1.5243923505909571, -5.970537052073158, 3.1415926207556044, 3.141592595135328, -3.141592736446672, 4.7123889760593425, -1.570796411176397]),
    (18, 1, False): ([[(0, 1)], [(0, 1)]], [-1.4070309611634295, 0.8212269113847128, 1.3328552033657557, -1.7441938193090516, -0.8069613214331393, 1.5707963253554789]),
    (18, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9106332375224038, -3.1415926499337123, 1.6814398420416712, 1.559615198181439, 1.5707963279500055, -3.252236167873998, -2.790574616585905, 0.23794112468438214, 2.882227075299116, 1.4461898954337506, 1.5707963292861733, -1.5493718712921591, 1.6954027543578263, -1.5707963281149033, 1.5707963268803136]),
    (18, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.150261991323049, 2.9702124526020387e-10, -1.5707963265996079, 1.570796327031829, 1.2151179876861893, 1.5707963265128873, 2.9036515275975097, -0.2545506687983135, -2.365379978483986, 2.5724041337103083, -1.570796326881318, 1.5874058651570953, 2.779519198005869, 1.52450509300303, 1.5707963288757387, -4.06788947316738, 0.36207345584616657, -3.1878838895085333, 3.1415926537773555, 1.881613482892856, -2.9766722311579916, -2.232567226679011, 1.5707963318666318, 0.941492995343581, 2.976672230805947, -2.492982806397426e-09, 1.5707963267320582, 0.013823578265710863, -3.2577292834076963, 1.9490969672714922e-08, -1.5707963071892574, -2.073510612639531, 0.11613662973054499, -1.5707963266161318, 1.5707963269097234, -2.6388783671358396]),
    (18, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.62876089082273, 3.207143727180451e-10, 3.141592563581479, 1.5707963930512265, 3.1415926132452694, -0.4254892320169296, 3.1415926554617153, -2.1672220499150302e-06, -3.141590372524593, -2.678626214663134, 2.303871817248053, 1.570796296726074, -1.5707941525163147, -4.712391419171327, -0.7009075635759452, -0.2421928125325318, 2.72105831832851, 4.291854644996096, -1.5707953418704574, -0.45334543866000415, 0.414730815935805, 1.5707963269479945, -2.370638594891855e-08, 2.356194490316906, 3.3569969676711406, 2.7845155507884134, 1.570795665612129, -3.757072362414367, 0.6154797081994632, 3.0353930153592836, 3.2290247653976105, 3.141593311583848, -4.71238810134235, 1.5707971352574008, 0.10619963811097126, -0.14508591759042166, -3.1415926783367416, -1.570795454516797, 1.5707956783320547, 2.706014679237488, -2.688066550208643, -6.283185261986906, 1.4008778718715453, -1.7407147815004738, -1.135218337653684, 2.771796464195714, 3.141592661751455, -0.7853981631824641, 1.4939738616182514e-08, 0.848180626709212, -3.376244875098048, -8.012930238985273e-09, 3.141592386093584, 1.570796326856873, 3.8642083534992877, 0.15092239740483854, -3.1415925677925705, -1.5707964271594188, -1.5707963464884251, 7.853980754630019]),
    (18, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.1259175090880995, 3.1415927660743885, -3.1415928144969154, 3.1415923381729813, -1.5707903990877186, 1.5707881937914514, -2.6888453089212776, -4.712388899126223, -4.7123888247286825, -2.5270118255635077, -1.528638703398775, 2.88878639665368, -1.468422491304016, 3.632475691325994, 0.0485599346329109, 3.199518903337444, 1.2906975762248016, 3.156457794331338, -1.3360712417146272, 1.570795964114473, 1.0313537071561634, 0.8982876627646335, 1.5707891217981789, -1.5012729176453454, 1.863228908021797, -4.7123889714604426, -2.407064900417929, 2.0642603983921557, 3.7570723607054157, -0.06951635966853904, 0.7067353338725699, -1.5802352556488906, -2.264196154366971, -0.49346262931794366, 0.955316605584988, -0.543700382150284, -0.48308432067667895, 0.5307048004009658, -1.6119187310745455, 4.712386737834227, 2.186276036899597, 3.685292850653054, -2.2817577334937336, -2.0920620841796898, 5.497787147888042, 5.497787138774169, -2.5261129475221407, 2.9008424039483076, -1.6106440649845013, -3.141592132099869, -3.3518598296625024, 2.566461666258846, 2.3763253865867652e-07, 0.24074938254755515, -3.323836777942886, -3.1415921385189267, -1.5707699416141656, -1.570796229659382, -3.141593006598407, 1.53618907907868, -2.880042778530744, 0.06877210252509998, 4.712389259583628, -1.5707982071541544, 2.491501367840538, 1.6028167131892337, -1.6794530434657111, -3.1415927520040103, -0.24736136651795773, -0.21026716604704326, -0.9207047316034975, 2.3252290134267137, -1.462140152775915, -3.141592914071555, 2.8138230192840257, 1.9477664212958315e-07, -1.5707963106445366, 2.3897479450653134, 1.2275595986603491, 0.06877245580414194, 3.1415933412511112, -4.712388968126655, -1.157594023235437e-08, -3.1879966323872218, -1.6196611391953892, 3.4313954214484555e-08, 1.5707681223083487, 1.57079640954849, -1.570796315520519, -1.5243923438756317, -5.9703895091152015, 3.1415927291590386, 3.1415940801955187, -3.1415928445660124, 4.712389021086119, -1.570794672084388]),
    (19, 1, False): ([[(0, 1)], [(0, 1)]], [-1.411501297412312, 0.8208403040523914, 1.3393189638948062, -1.7394627188689487, -0.807346163925788, 1.5707963259311697]),
    (19, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.9012189755992324, -3.141592650369, 1.6814398420806824, 1.5549080676576679, 1.5707963276939607, -3.252236167834986, -2.7952817471096765, 0.23147736386476733, 2.8854589553945846, 1.4461898959561432, 1.5707963284807889, -1.5461399911966898, 1.6954027548802209, -1.5707963277694992, 1.5707963269325345]),
    (19, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.162158472215834, -1.407527607884688e-09, -1.570796326220397, 1.570796327272357, 1.2091697476053875, 1.570796326403748, 2.910115289834435, -0.25131878270635666, -2.371328218564624, 2.567697453067902, -1.5707963269268959, 1.590637751248939, 2.77951919896617, 1.5245047450146954, 1.5707963203750903, -4.070243091575727, 0.36207345680662256, -3.1878842374603993, 3.141592654030621, 1.8792598644844911, -2.9766722326001296, -2.2372748043401183, 1.5707963200893813, 0.9391394811666794, 2.9766722293638166, -4.8277851166014905e-09, 1.570796326792092, 0.011470064088889778, -3.257729284408622, -2.776778524083472e-08, -1.5707963475237265, -2.073510612310364, 0.11613662872960843, -1.5707963264637952, 1.570796327538948, -2.6388783668066007]),
    (19, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.62165935945674, 9.766421305982161e-11, 3.1415925672546705, 1.5707963823984248, 3.141592617022536, -0.4325907633829147, 3.1415926552386595, -2.1635488609597886e-06, -3.1415903662888907, -2.6753943221717096, 2.296770286170475, 1.5707963005476053, -1.570794148843123, -4.71239141293561, -0.6976756710845273, -0.24929434361011102, 2.7329547907902962, 4.303751118549044, -1.5707954016906793, -0.4501135539571842, 0.41473081624610453, 1.570796321170412, -2.088826864658271e-08, 2.3561944845031855, 3.3602288523739685, 2.7845155510987176, 1.5707956651397972, -3.7617795189697767, 0.6201868269054217, 3.0353930155455404, 3.229024765707903, 3.1415933111115315, -4.712388146226601, 1.5707970884410838, 0.10619963829722959, -0.14508591728012454, -3.141592678985754, -1.5707954990192856, 1.570795714499143, 2.706014679796496, -2.6880665498983545, -6.283185262635914, 1.4055849994331295, -1.7360076502534938, -1.135218337094687, 2.771796464506007, 3.1415926671642125, -0.7853981672961201, 1.3682106553394079e-08, 0.8481806115717427, -3.3762448747877603, -2.6001726977247663e-09, 3.1415924060789795, 1.5707963513281193, 3.8642083383618204, 0.15092239771513946, -3.1415925771431383, -1.5707964222725763, -1.5707963445022453, 7.853980799780794]),
    (19, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.1094597832515576, 3.141592437815424, -3.1415926792773177, 3.141592298723076, -1.5707936253919565, 1.5707814270338227, -2.6970741700829284, -4.712389190324258, -4.712389319545677, -2.5270118978291802, -1.5254068279453676, 2.8920182651316813, -1.4766513524656255, 3.6182705966223017, 0.05566146843969931, 3.1995188310721043, 1.293929451678048, 3.159689662809036, -1.3360711910100977, 1.570798008871361, 1.0384552409626902, 0.8982875904990347, 1.5707908289592991, -1.5012693503157803, 1.8632289587259816, -4.712389059210522, -2.40706401859793, 2.0642606577273583, 3.7617796184099634, -0.0695127923388756, 0.7067353845773434, -1.5802353433990168, -2.2641952725470205, -0.49346236998320825, 0.9506096147141925, -0.5437004182582298, -0.48308426997225773, 0.5307047126509548, -1.611917849254453, 4.7123869754258925, 2.1909830437044553, 3.6852928145447446, -2.281757682789115, -2.0920621719297903, 5.497787064747109, 5.497787221228274, -2.5214056998006362, 2.90084268076123, -1.6106440142799432, -3.141595036464852, -3.3459116166111724, 2.5605133628653727, 3.2438277568665277e-07, 0.2407496593605894, -3.324023704021629, -3.141595027038177, -1.5707575475792912, -1.5707957170492066, -3.141593072142554, 1.5361888764214453, -2.880229704609482, 0.0687577306262717, 4.71239032092117, -1.5707993969459757, 2.4915014420083237, 1.6028165105321623, -1.6794527305326643, -3.1415917426722504, -0.25033545647791927, -0.2043188939847434, -0.9207046574361036, 2.325228810769833, -1.4621398398429377, -3.1415919047398084, 2.8108489293242607, 2.3693223397457294e-07, -1.5707963288881837, 2.3897477424086415, 1.2276842427756085, 0.06875606647549923, 3.1415927288529426, -4.712388970321664, 6.370662451119819e-09, -3.1879966346069852, -1.6195364950796818, -3.9121330302093886e-08, 1.5707550225428748, 1.5707963679669295, -1.5707963367961002, -1.5243923460954358, -5.970264864999724, 3.141592655723905, 3.141594620100111, -3.1415927815588836, 4.712389041311043, -1.5707935739943866]),
    (20, 1, False): ([[(0, 1)], [(0, 1)]], [-1.4156243707886076, 0.8204934126167481, 1.3452829214104252, -1.73509899227247, -0.8076915516906306, 1.5707963265629739]),
    (20, 2, False): ([[(0, 1)], [(1, 2)], [(0, 1)], [(1, 2)]], [1.8925468826055205, -3.1415926511958943, 1.681439841816826, 1.5505720208778462, 1.5707963275728969, -3.25223616809884, -2.7996177938894977, 0.22551340577989756, 2.888440934773306, 1.4461898958648296, 1.5707963292681235, -1.543158011817969, 1.695402754788907, -1.5707963278041848, 1.5707963268054226]),
    (20, 3, False): ([[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]], [1.1730969117431556, -5.05618698019837e-10, -1.5707963267538547, 1.5707963265766975, 1.2037005272290833, 1.5707963270356782, 2.916079247572796, -0.24833680673367978, -2.3767974389412507, 2.5633617115817158, -1.570796326648774, 1.5936197272220027, 2.7795191976860725, 1.524504379123497, 1.5707963271457426, -4.072411082793509, 0.36207345552610776, -3.187884601280673, 3.1415926534383196, 1.8770918732669184, -2.9766722309909652, -2.241611157309419, 1.5707963271512002, 0.9369714270865429, 2.976672230972439, -3.519403485423575e-11, 1.5707963269462875, 0.009302010009136944, -3.2577292832765017, 1.4072769005783467e-09, -1.570796325551871, -2.0735106129269063, 0.11613662986150404, -1.5707963268438485, 1.5707963271583931, -2.6388783674229863]),
    (20, 4, False): ([[(3, 4)], [(0, 1)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)], [(2, 3)], [(1, 2), (3, 4)]], [5.615143356051757, -1.9107838427853744e-09, 3.1415925765534287, 1.570796370462534, 3.141592618206295, -0.4391067667879246, 3.1415926532302114, -2.154250106480041e-06, -3.141590361274036, -2.672412354804256, 2.290254282896003, 1.5707963101085356, -1.5707941395443632, -4.712391407920757, -0.6946937037170811, -0.2558103468845722, 2.743893238649604, 4.314689564998943, -1.5707954581390686, -0.4471315791029756, 0.4147308164497065, 1.5707963271828613, -1.921270740189741e-08, 2.3561944902649032, 3.363210827228179, 2.78451555130231, 1.5707956673122134, -3.7661155407193836, 0.6245228848063047, 3.0353930170522165, 3.229024765911506, 3.1415933132839444, -4.712388178996348, 1.5707970595034528, 0.10619963980390433, -0.14508591707652344, -3.141592680267768, -1.5707955429633507, 1.5707957493769815, 2.7060146805512457, -2.6880665496947627, -6.2831852639179155, 1.4099210493723893, -1.731671603852521, -1.1352183363399382, 2.7717964647095967, 3.1415926617897907, -0.785398163048827, 1.3500474565034411e-08, 0.8481806260562521, -3.3762448745841644, -7.974580570068023e-09, 3.1415924235780692, 1.5707963282525206, 3.8642083528463265, 0.1509223979187401, -3.1415926022040233, -1.5707964215477588, -1.5707963452417653, 7.853980839001154]),
    (20, 5, False): ([[(0, 1), (2, 3), (4, 5)], [(1, 2)], [(0, 1), (4, 5)], [(1, 2), (3, 4)], [(4, 5)], [(3, 4)], [(2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(0, 1), (2, 3)], [(3, 4)], [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4)], [(2, 3), (4, 5)]], [2.0943951094654607, 3.1415925532643842, -3.1415927548457048, 3.141592169891693, -1.5707951533478952, 1.5707775707924474, -2.704606508247201, -4.712389063011616, -4.712389025720749, -2.5270122039296568, -1.522424838878677, 2.8950002528630843, -1.4841836906301817, 3.605239194590624, 0.0621774774842041, 3.199518524971668, 1.296911440744896, 3.1626716505405903, -1.3360712805961485, 1.57079739974409, 1.0449712500074602, 0.8982872843983549, 1.5707918452051257, -1.5012673196863027, 1.8632288691401448, -4.712389007951291, -2.407063638660775, 2.064260830966303, 3.7661155558258166, -0.06951076170951609, 0.7067352949914016, -1.5802352921394138, -2.2641948926099675, -0.4934621967445474, 0.946273447797361, -0.5437004636604581, -0.4830843595584481, 0.5307047639105209, -1.6119174693173264, 4.712386179928758, 2.1953191988622973, 3.6852927691422, -2.28175777237506, -2.092062120670162, 5.4977870512636935, 5.497787234677558, -2.517069741129078, 2.90084282513934, -1.610644103865743, -3.141594162525163, -3.340442401618678, 2.555044129957265, 2.7397273562735086e-07, 0.24074980373876703, -3.324181670499133, -3.14159415275473, -1.5707507319716514, -1.5707952709567052, -3.141593156110166, 1.536188774866976, -2.8803876710867997, 0.0687445070370974, 4.712388812771856, -1.5708000012947811, 2.4915014480111943, 1.6028164089774306, -1.67945306365034, -3.1415920506021218, -0.25307006162703277, -0.1988496677248504, -0.9207046514333892, 2.325228709215539, -1.4621401729606118, -3.141592212669618, 2.808114324174802, 3.0294899710136937e-07, -1.5707963249783183, 2.3897476408540843, 1.227789537257534, 0.06874346180540658, 3.14159434490389, -4.712388987055188, 2.7272508672623742e-08, -3.1879966391795413, -1.619431200597835, 5.613310771880372e-09, 1.5707476930260773, 1.5707965558109644, -1.5707963359608785, -1.5243923506680774, -5.970159570518078, 3.1415927004588333, 3.141595119377687, -3.1415929557096884, 4.7123890322108295, -1.5707928669140505]),
}
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


# Minimum free single-qubit slots in an idle window before we echo it (XY4 needs 4).
# In simulation, echoing every short window cost up to ~1% of the ceiling share (pulse error
# with nothing to refocus in sub-0.3 us idles); long windows are where hardware dephasing
# and ZZ crosstalk accumulate, so only those are protected.
_DD_MIN_SLOTS = 4


def _add_dd(circuit: cirq.Circuit) -> cirq.Circuit:
    """Echo long idle windows of qubits that may be in superposition with X-Y-X-Y.

    A qubit is protected from its first non-virtual single-qubit gate (before that it sits
    in |0>, which dephasing cannot touch) to the end of the circuit, including the trailing
    window before measurement. Pulses go only into single-qubit moments, and only windows
    with >= _DD_MIN_SLOTS free slots are echoed. XYXY = -I, so the noiseless state is
    unchanged up to a global phase.
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
            seq = [_X, _Y, _X, _Y] if len(slots) >= _DD_MIN_SLOTS else []
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
    scs: bool = True, tiny: bool = True, alap: bool = False, dd: bool = True,
    bryan: bool = True,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """`create_dicke_state` with switches for the experiments (defaults = submission)."""
    if bryan and _bryan_wins(int(num_qubits), int(num_excitations)):
        try:
            return _bryan_create_dicke_state(
                int(num_qubits), int(num_excitations), cz_pauli_error_rate,
                sq_pauli_error_rate, zero_errors, one_errors,
            )
        except Exception:  # any failure there: fall back to our constructions below
            pass
    circuit, qubits = _build_core(
        num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
        zero_errors, one_errors, w_tree=w_tree, synth=synth, scs=scs, tiny=tiny,
    )
    if alap:
        circuit = _alap(circuit)
    return (_add_dd(circuit) if dd else circuit), qubits


def _alap(circuit: cirq.Circuit) -> cirq.Circuit:
    """Schedule every operation as late as possible, so qubits stay in |0> (immune to
    T1 and dephasing) until they are needed: pack the time-reversed circuit earliest-first,
    then reverse it back. Order on each qubit is unchanged."""
    rev = cirq.Circuit(op for m in reversed(circuit) for op in m)
    c = cirq.Circuit(reversed(list(rev)))
    if any(any(op.gate == cirq.CZ for op in m) and any(len(op.qubits) == 1 for op in m) for m in c):
        c = cirq.stratified_circuit(c, categories=[lambda op: op.gate == cirq.CZ])
    return c


def _bryan_wins(N: int, n: int) -> bool:
    """Where the vendored constructions measured better (experiments/results/vs_bryan.csv,
    25 cases x 3 calibrations, idle noise): k = 2 for N >= 6 (1-7% higher median score; ~3
    fewer CZ) and k >= 3 for N >= 9 (21-39%; the split B-E halves the CZ count). Ours win the
    W states, k = 2 up to N = 5 and k >= 3 up to N = 8 (whole-state synthesis, up to +28%)."""
    if N < 1 or not 0 <= n <= N:
        return False
    k = min(n, N - n)
    return (k == 2 and N >= 6) or (k >= 3 and N >= 9)


def _build_core(
    num_qubits, num_excitations, cz_pauli_error_rate, sq_pauli_error_rate,
    zero_errors, one_errors, *, w_tree: bool, synth: bool, scs: bool, tiny: bool,
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

    # Measured (experiments/results/tiny.csv): for W2 this raises E[score] 60.2 -> 65.6 while
    # the median dips 4.37 -> 4.16; for W3 it gains nothing, so only N = 2 uses it.
    if k == 1 and w_tree and tiny and N == 2:
        picked = _best_tiny_w(g, N, n, flip, cz_pauli_error_rate, sq_pauli_error_rate,
                              zero_errors, one_errors)
        if picked is not None:
            return picked

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


# ---- bonus track ------------------------------------------------------------------------------


def _uniform_calibration() -> dict:
    """The bonus provides no error rates: every Willow105 qubit and coupler, equal rates."""
    import cirq_google

    qubits = sorted(cirq_google.Willow105.metadata.qubit_set)
    pairs = [tuple(sorted(p)) for p in cirq_google.Willow105.metadata.qubit_pairs]
    return dict(
        cz_pauli_error_rate={p: (2.5e-3, 0.0) for p in pairs},
        sq_pauli_error_rate={q: (4e-4, 0.0) for q in qubits},
        zero_errors={q: (5e-3, 0.0) for q in qubits},
        one_errors={q: (1e-2, 0.0) for q in qubits},
    )


def _pack(circuit: cirq.Circuit) -> cirq.Circuit:
    """Fewest moments: merge single-qubit runs, then re-insert every operation as early as
    possible (single-qubit gates may share a moment with CZs; depth is all that is scored)."""
    c = cirq.merge_single_qubit_gates_to_phxz(circuit)
    c = cirq.drop_empty_moments(cirq.drop_negligible_operations(c))
    return cirq.Circuit(c.all_operations())


def create_dicke_state_adaptive(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state, allowing for adaptive circuits.

    This entry is not adaptive yet: it returns the shallower of our exact unitary
    construction (uniform calibration, packed for depth) and the vendored bonus solution.
    Every measurement branch is trivially exact because there are no measurements.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.

    Returns:
        A Cirq circuit to produce Dicke state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested (N, n) combination is not implemented.
    """
    cal = _uniform_calibration()
    ours, qs = build(num_qubits, num_excitations, **cal, dd=False, bryan=False)
    best = (_pack(ours), qs)
    try:  # the vendored bonus solution is shallower for most cases: keep whichever is shallower
        theirs = _bryan_create_dicke_state_adaptive(int(num_qubits), int(num_excitations))
        if len(theirs[0]) < len(best[0]):
            best = theirs
    except NotImplementedError:
        pass
    return best


# =============================================================================================
# Vendored: Bryan's solution (origin/bryan @ b94cb62, dicke_solution.py), verbatim except that
# its imports moved to the top of this file and four names were renamed to avoid collisions:
# _X -> _BRYAN_X, _Y -> _BRYAN_Y, create_dicke_state -> _bryan_create_dicke_state,
# create_dicke_state_adaptive -> _bryan_create_dicke_state_adaptive.
# build() dispatches to it where it measured better (experiments/results/vs_bryan.csv).
# =============================================================================================

# %% [markdown]
# # DuQuantum solution: exact, hardware-aware Dicke-state preparation on Willow
#
# This section contains our complete solution. It follows the research plan in
# `DuQuantum_Research_and_Implementation_Plan.md`: several *exact* constructions are generated,
# placed on the calibrated device graph, compiled to the native gate set
# (`CZ` + `PhasedXZGate`) with dynamical decoupling, and the candidate with the highest
# *predicted* hardware success probability is returned.
#
# **Notation.** $N$ = number of qubits, $n$ = number of excitations (the notebook's names),
# $k=\min(n, N-n)$ = the excitation count we actually prepare (see the complement trick below).
# $R_y(\theta)=e^{-i\theta Y/2}$, so $R_y(\theta)|0\rangle=\cos\frac\theta2|0\rangle+\sin\frac\theta2|1\rangle$.
#
# **Pipeline.**
#
# | Stage | What it does | Why |
# | --- | --- | --- |
# | 1. Exact building blocks | Splits, Givens rotations and "split-and-cyclic-shift" (SCS) blocks written directly in CZs | Every CZ costs ~0.25% error, so each block uses the fewest CZs we could prove or find |
# | 2. Logical constructions | W-state tree ($k=1$), Bärtschi–Eidenbenz (BE) path and a two-arm "split" construction ($k\ge2$) | Different shapes trade CZ count against depth |
# | 3. Placement | Search trees / paths / ladders on the calibrated coupler graph | Uses the best qubits, couplers and readout |
# | 4. Compilation | Merge 1-qubit gates into `PhasedXZ`, ALAP layering into alternating 1q/CZ moments, XY dynamical decoupling | Native gate set, minimal idle exposure |
# | 5. Selection | Predict $\log F$ (gates + readout + idling) for every candidate and keep the best | Calibration-aware choice |
#
# **Complement trick.** $|D^{(N)}_n\rangle = X^{\otimes N}|D^{(N)}_{N-n}\rangle$ because flipping every bit maps
# weight-$n$ strings bijectively onto weight-$(N-n)$ strings. We therefore always prepare
# $k=\min(n,N-n)\le N/2$ excitations and, if $n>N/2$, append $X$ on every qubit. The extra $X$'s merge into
# the final single-qubit layer, so the trick costs nothing.

# %%
# (imports are at the top of main.py)

# %% [markdown]
# ## 1. Exact building blocks
#
# We build circuits from a tiny internal instruction list: `('U', q, M)` applies the $2\times2$ unitary `M`
# to qubit `q`, and `('CZ', a, b)` applies a CZ. Single-qubit matrices are merged later into one
# `PhasedXZGate` between consecutive CZs, so they are free in CZ count; only CZs matter.
#
# **(a) CNOT from one CZ.** $\mathrm{CNOT}_{c\to t}=H_t\,\mathrm{CZ}_{ct}\,H_t$ because $HZH=X$.
#
# **(b) Controlled rotation onto a known $|0\rangle$: one CZ.** For any $\varphi$,
# $R_y(\varphi)\,Z\,R_y(-\varphi)=R_y(2\varphi)Z$ (since $ZR_y(\varphi)Z=R_y(-\varphi)$), and $Z|0\rangle=|0\rangle$. Hence
# $$R_y(\tfrac\theta2)_t\,\mathrm{CZ}_{ct}\,R_y(-\tfrac\theta2)_t:\quad |0\rangle_c|0\rangle_t\mapsto|0\rangle|0\rangle,\qquad |1\rangle_c|0\rangle_t\mapsto|1\rangle R_y(\theta)|0\rangle .$$
# A general controlled-$R_y$ needs two CZs; one suffices whenever the target is guaranteed to be
# $|0\rangle$ when the control is $|1\rangle$. We use this "known-target" trick everywhere.
#
# **(c) Excitation transfer (W-state split): two CZs.** $|1\rangle_s|0\rangle_d\mapsto\sqrt{1-p}\,|10\rangle+\sqrt p\,|01\rangle$,
# $|00\rangle\mapsto|00\rangle$: a known-target rotation of $d$ followed by $\mathrm{CNOT}_{d\to s}$ (which clears $s$ when $d$
# took the excitation). Two CZs are necessary: with one CZ, $(A_1\otimes B_1)\,\mathrm{CZ}\,(A_0\otimes B_0)$ applied to $|x\rangle|0\rangle$
# gives $\sum_y (A_0)_{yx}A_1|y\rangle\otimes B_1Z^y|b\rangle$; keeping $|00\rangle$ a product state forces $A_0$ to be diagonal, and then
# the $|10\rangle$ input also stays a product state, so the entangled branch cannot be produced.
#
# **(d) Givens rotation: two CZs.** $G(\theta)=\exp\!\big[\tfrac\theta2(|10\rangle\langle01|-|01\rangle\langle10|)\big]=\exp\!\big[i\tfrac\theta4(XY-YX)\big]$
# fixes $|00\rangle,|11\rangle$ and rotates $|01\rangle\mapsto\cos\frac\theta2|01\rangle+\sin\frac\theta2|10\rangle$. Conjugating by $S$ on qubit $b$ maps
# $XY-YX\to-(XX+YY)$, and $R_x(\frac\pi2)^{\otimes2}$ maps $YY\to ZZ$. Finally
# $\mathrm{CNOT}_{a\to b}\,[R_x(\alpha)\otimes R_z(\beta)]\,\mathrm{CNOT}_{a\to b}=e^{-i\alpha XX/2}e^{-i\beta ZZ/2}$, so
# $$G(\theta)=V^\dagger W^\dagger\,\mathrm{CNOT}_{ab}\,[R_x(\tfrac\theta2)\otimes R_z(\tfrac\theta2)]\,\mathrm{CNOT}_{ab}\,W V,\quad V=S_b,\ W=R_x(\tfrac\pi2)^{\otimes2}.$$
#
# **(e) Moving-hole gate $G_3$: five CZs** (an exhaustive numerical search found no 3- or 4-CZ circuit on a line). On qubits
# $(c,a,b)$ with $a$ in the middle and input domain $\{000,001,011,101,111\}$ it maps $001\mapsto\cos\frac\theta2|001\rangle+\sin\frac\theta2|010\rangle$
# (partial move of a "hole" from $a$ to $b$), $101\mapsto|110\rangle$ (full move when $c=1$, i.e. the hole is "in transit"), and fixes
# the rest. Construction: $\mathrm{CNOT}_{a\to b}\,M\,\mathrm{CNOT}_{a\to b}$ with
# $M=R_y(\alpha)Z^{c}R_y(\beta)Z^{b}R_y(-\beta)Z^{c}R_y(-\alpha)$ acting on $a$, $\alpha=\frac{\theta+\pi}4$, $\beta=\frac{\theta-\pi}4$. After the first
# CNOT, $b=1\Rightarrow a=0$, and pushing the $Z$'s through the rotations gives total $R_y$ angles $0$ ($c=b=0$), $0$ ($c=1,b=0$),
# $\theta$ ($c=0,b=1$) and $\pi$ ($c=b=1$), exactly as required.
#
# **(f) Merged top block: four CZs instead of 2 + 5 = 7.** Found by an exhaustive numerical search over CZ orderings
# (only the ordering $ab,bc,ab,bc$ works) and then derived analytically. On a line $a-b-c$ with domain
# $\{000,001,011\}$:
# 1. known-target rotation $b\to a$ with angle $\theta_2$ (only the $011$ input has $b=1$, and there $a=0$);
# 2. a Givens rotation $G(\theta_1)$ on $(b,c)$ whose middle $R_x(\theta_1/2)$ on $b$ is replaced by
#    $R_x(\frac\pi2)\,\mathrm{CZ}_{ab}\,R_x(\frac{\theta_1-\pi}2)$. For $a=0$ nothing changes (the $R_x$ angles add to $\theta_1/2$), so
#    $001\mapsto\cos\frac{\theta_1}2|001\rangle+\sin\frac{\theta_1}2|010\rangle$ and $011$ is fixed. For $a=1$ the middle becomes $R_x(\pi-\frac{\theta_1}2)Z$;
#    moving the $Z$ out of the Givens frame shows that the $|11\rangle_{bc}$ input is rotated by an extra $\pi-\theta_1$ in the
#    $\{|01\rangle,|10\rangle\}$ plane and lands exactly on $|10\rangle_{bc}$. So $|111\rangle\mapsto|110\rangle$ up to a phase $-i$;
# 3. an $S$ gate on $a$ removes that phase ($a=1$ only in that branch).
#
# Every block is checked numerically against its defining isometry in the verification section.

# %%
_I2 = np.eye(2, dtype=complex)  # identity
_BRYAN_X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
_BRYAN_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli Y
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
_S = np.diag([1, 1j]).astype(complex)  # phase gate S = sqrt(Z)


def _RY(theta):
    """R_y(theta) = exp(-i theta Y / 2): real rotation, |0> -> cos(theta/2)|0> + sin(theta/2)|1>."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)  # half-angle cosine and sine
    return np.array([[c, -s], [s, c]], dtype=complex)  # standard real rotation matrix


def _RX(theta):
    """R_x(theta) = exp(-i theta X / 2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)  # half-angle cosine and sine
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)  # rotation about the X axis


def _RZ(theta):
    """R_z(theta) = exp(-i theta Z / 2)."""
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])  # diagonal phase rotation


def _u(q, m):
    """Instruction: apply the 2x2 unitary m to qubit q."""
    return ('U', q, m)  # tuples are cheap; matrices are merged later


def _cz(a, b):
    """Instruction: CZ between qubits a and b (symmetric)."""
    return ('CZ', a, b)  # the only two-qubit gate we ever emit


def _cnot(control, target):
    """CNOT = H_t CZ H_t (one CZ, derivation (a))."""
    return [_u(target, _H), _cz(control, target), _u(target, _H)]  # HZH = X on the target


def _cry_on_zero(control, target, theta):
    """Controlled-R_y(theta), valid when target = |0> whenever control = |1> (one CZ, derivation (b))."""
    return [_u(target, _RY(-theta / 2)), _cz(control, target), _u(target, _RY(theta / 2))]  # R_y(t/2) CZ R_y(-t/2)


def _split_angle(p):
    """Angle theta with sin^2(theta/2) = p, i.e. R_y(theta)|0> has |1>-probability p."""
    return 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))  # clamping guards tiny float overshoots


def _transfer(src, dst, p):
    """|1>_src|0>_dst -> sqrt(1-p)|10> + sqrt(p)|01>, |00> -> |00> (two CZs, derivation (c))."""
    return _cry_on_zero(src, dst, _split_angle(p)) + _cnot(dst, src)  # rotate dst, then clear src if dst took it


def _givens(a, b, theta, middle_a=None):
    """Givens rotation G(theta) on (a, b) with two CZs (derivation (d)).

    `middle_a` optionally replaces the middle R_x(theta/2) on qubit a (used by the merged top block).
    """
    if middle_a is None:  # plain Givens rotation
        middle_a = [_u(a, _RX(theta / 2))]  # the XX-part angle
    ops = [_u(b, _S), _u(a, _RX(np.pi / 2)), _u(b, _RX(np.pi / 2))]  # W V: map XY-YX onto -(XX+ZZ)
    ops += _cnot(a, b)  # CNOT conjugation turns X_a into X_a X_b and Z_b into Z_a Z_b ...
    ops += middle_a + [_u(b, _RZ(theta / 2))]  # ... so these rotations become exp(-i t XX/2) exp(-i t ZZ/2)
    ops += _cnot(a, b)  # close the CNOT conjugation
    ops += [_u(a, _RX(-np.pi / 2)), _u(b, _RX(-np.pi / 2)), _u(b, _S.conj().T)]  # undo W and V
    return ops


def _g3(c, a, b, theta):
    """Moving-hole gate on (c, a, b): partial move a->b if c=0, full move if c=1 (five CZs, derivation (e))."""
    al = (theta + np.pi) / 4  # alpha in the derivation
    be = (theta - np.pi) / 4  # beta in the derivation
    ops = _cnot(a, b)  # afterwards b = 1 implies a = 0, enabling known-target rotations of a
    ops += [_u(a, _RY(-al)), _cz(c, a), _u(a, _RY(-be)), _cz(b, a), _u(a, _RY(be)), _cz(c, a), _u(a, _RY(al))]  # M
    ops += _cnot(a, b)  # map |a=1,b=1> back to |a=1,b=0>, completing the excitation move
    return ops


def _top_block(a, b, c, p_top, p_next):
    """Merged first two steps of an SCS sweep on a line a-b-c, four CZs (derivation (f)).

    Domain {000, 001, 011} (qubit order a, b, c). The excitation on c moves to b with probability p_next
    when the input is 001, and to a with probability p_top when the input is 011.
    """
    t2 = _split_angle(p_top)  # rotation for the 011 branch (landing site a)
    t1 = _split_angle(p_next)  # rotation for the 001 branch (landing site b)
    ops = _cry_on_zero(b, a, t2)  # step 1: only 011 has b = 1, and there a = 0
    middle = [_u(b, _RX((t1 - np.pi) / 2)), _cz(a, b), _u(b, _RX(np.pi / 2))]  # step 2: Z_b inserted iff a = 1
    ops += _givens(b, c, t1, middle_a=middle)  # Givens on (b, c) with the modified middle layer
    ops += [_u(a, _S)]  # step 3: cancel the -i phase acquired by the a = 1 branch
    return ops


# %% [markdown]
# ## 2. Logical constructions
#
# ### 2.1 W states ($k=1$): excitation tree, $2N-3$ CZs, depth $O(\log N)$
#
# Pick a spanning tree of $N$ qubits with root $r$, and write $s(v)$ for the size of the subtree under $v$. Starting from
# $|1\rangle_r$, every node $v$ that currently "owns" the excitation for $m_v$ qubits (itself plus its not-yet-served child subtrees)
# transfers it to child $c$ with probability $s(c)/m_v$. The probability that the excitation reaches a given qubit is the product
# of conditional probabilities along its root path, which telescopes to $1/N$; all amplitudes stay real and positive, so the
# result is exactly $|W_N\rangle$.
#
# The *first* transfer from the root acts on the product state $|1\rangle_r|0\rangle_c$, so it needs only one CZ:
# prepare $r$ in $\sqrt p|0\rangle+\sqrt{1-p}|1\rangle$, then apply $\mathrm{CNOT}_{r\to c}$ and $X_c$. Total: $1+2(N-2)=2N-3$ CZs. An exhaustive numerical
# search confirmed that $W_3$ needs 3 and $W_4$ needs 5 CZs, so this count is optimal at those sizes.
#
# Children are served in order of decreasing subtree height, so the longest chain starts first; the depth is roughly
# $2\times$(tree height + branching), i.e. logarithmic in $N$ for a bushy tree on the 2D grid.
#
# ### 2.2 General $k$: Bärtschi–Eidenbenz (BE) on a path
#
# BE prepares $|D^{(N)}_k\rangle=U_N|0^{N-k}1^k\rangle$ with $U_N=\prod_{m=N}^{2}\mathrm{SCS}_m$. Before step $m$ the first $m$ positions hold
# $|0^{m-r}1^r\rangle$ (ones right-justified, $r$ = remaining excitations). Since
# $$|D^{(m)}_r\rangle=\sqrt{\tfrac rm}\,|D^{(m-1)}_{r-1}\rangle|1\rangle+\sqrt{\tfrac{m-r}m}\,|D^{(m-1)}_r\rangle|0\rangle,$$
# step $m$ must map $|0^{m-r}1^r\rangle\mapsto\sqrt{\tfrac rm}|0^{m-r}1^r\rangle+\sqrt{\tfrac{m-r}m}|0^{m-r-1}1^r0\rangle$: either position $m$ keeps its 1, or the
# "hole" at the boundary $m-r$ travels to $m$ (a cyclic shift). Only the window $m-k,\dots,m$ is touched.
#
# We implement $\mathrm{SCS}_m$ as a *local sweep* along the window, so it needs only nearest-neighbour CZs: the hole moves
# one site at a time, partially at its original boundary and fully while "in transit". The first two sweep steps form the
# 4-CZ merged top block; the remaining steps are $G_3$ gates. Cost per step: 2 CZs ($k=1$), 4 CZs ($k=2$), $4+5(k-2)$ in general
# (textbook BE uses one non-local three-qubit gate per extra branch, each several CNOTs, **plus** SWAP routing on a line).
#
# *Caveat handled in code:* when $m\le k$ the window is the whole prefix and the all-ones input $|1^m\rangle$ can occur; the
# merged block is not valid there (numerically it would need at least 6 CZs), so those steps use the plain sweep, which fixes $|1\dots1\rangle$.
#
# ### 2.3 General $k$: two-arm "split" construction (our main $k\ge2$ workhorse)
#
# Split the qubits into a left arm $L$ ($a$ qubits) and a right arm $R$ ($b=N-a$ qubits). Then
# $$|D^{(N)}_k\rangle=\sum_j\sqrt{P(j)}\;|D^{(a)}_{k-j}\rangle_L|D^{(b)}_j\rangle_R,\qquad P(j)=\frac{\binom bj\binom a{k-j}}{\binom Nk}\ \text{(hypergeometric)}.$$
# Because the total weight $k$ is *fixed* at the start, the weight distribution is cheap to prepare in unary:
# 1. **Thermometer chain** on the first $k$ qubits $r_1..r_k$ of $R$: $r_1\leftarrow R_y$ with $P(j\ge1)$, then known-target rotations
#    $r_{u-1}\to r_u$ with conditional probabilities $P(j\ge u)/P(j\ge u-1)$. Result: $\sum_j\sqrt{P(j)}|1^j0^{k-j}\rangle$ ($k-1$ CZs, exact).
# 2. **Rung copy.** The left arm's first $k$ qubits are laid out as a parallel row in *reverse* order, so that $l_{k+1-u}$ neighbours
#    $r_u$ (a "ladder"). The left arm needs the thermometer $l_v=[k-j\ge v]=\lnot r_{k+1-v}$, so one parallel layer of CNOTs plus $X$ gates
#    writes it ($k$ CZs, depth 1).
# 3. Each arm now holds a right-justified unary input, so we run the BE unitary on both arms **in parallel**.
#
# Compared with one long BE path, the two arms run concurrently (about half the depth) and the construction also *saves CZs*,
# because the expensive full-window sweeps happen on half as many positions. Examples: $(N,k)=(6,3)$: 23 vs 36 CZs; $(8,4)$: 49 vs 77.

# %%
def _scs_ops(window, m, all_ones_possible):
    """SCS_m on window = [p_k, ..., p_1, p_0] (p_0 = position m) as a nearest-neighbour sweep."""
    k = len(window) - 1  # window size minus one = largest number of excitations handled
    if k == 1:  # a single partial move = Givens rotation (it also fixes |11>)
        return _givens(window[0], window[1], _split_angle((m - 1) / m))  # move the hole with probability (m-1)/m
    ops = []  # accumulated instructions
    i = k  # current hole index: branch r = i has its hole at p_i
    if not all_ones_possible:  # the merged top block needs p_k = 0 on every input
        ops += _top_block(window[0], window[1], window[2], (m - k) / m, (m - k + 1) / m)  # branches r = k and k-1
        i = k - 2  # the block already performed the top two sweep steps
    while i >= 1:  # remaining sweep steps, top to bottom
        theta = _split_angle((m - i) / m)  # branch r = i moves its hole with probability (m-i)/m
        if i == k:  # very first step of an unmerged sweep: no "in transit" control yet
            ops += _givens(window[0], window[1], theta)  # partial move p_k <-> p_{k-1}
        else:  # generic step: the control is the site above the hole position
            ops += _g3(window[k - i - 1], window[k - i], window[k - i + 1], theta)  # (c, a, b) = (p_{i+1}, p_i, p_{i-1})
        i -= 1  # move one site towards the head p_0
    return ops


def dicke_unitary_ops(path, k):
    """BE unitary on `path`: maps |0^{N-r} 1^r> (ones at the END of the list) to |D^(N)_r> for every r <= k."""
    N = len(path)  # number of positions
    k = min(k, N)  # cannot hold more excitations than qubits
    ops = []  # accumulated instructions
    for m in range(N, 1, -1):  # decide positions N, N-1, ..., 2 (position 1 is then determined)
        kk = min(k, m - 1)  # the window reaches back k sites, but not past position 1
        ops += _scs_ops(list(path[m - 1 - kk: m]), m, all_ones_possible=(k >= m))  # all-ones input reachable iff k >= m
    return ops


def dicke_path_ops(path, k):
    """|D^(N)_k> on a path: X on the last k sites, then the BE unitary."""
    return [_u(q, _BRYAN_X) for q in path[len(path) - k:]] + dicke_unitary_ops(path, k)  # |0^{N-k}1^k>, then U_N


def split_ops(left, right, k):
    """Two-arm construction (Section 2.3).

    left = [l_1, ..., l_a], right = [r_1, ..., r_b]; l_1 and r_1 are the arm heads.
    Ladder requirement: l_{k+1-u} must neighbour r_u for every rung that is actually used.
    """
    a, b = len(left), len(right)  # arm lengths
    N = a + b  # total number of qubits
    P = [math.comb(b, j) * math.comb(a, k - j) / math.comb(N, k) if (0 <= k - j <= a and j <= b) else 0.0
         for j in range(k + 1)]  # hypergeometric law of j = number of excitations in the right arm
    tail = [sum(P[u:]) for u in range(k + 2)]  # tail[u] = P(j >= u)
    ops = []  # accumulated instructions
    for u in range(1, min(k, b) + 1):  # thermometer chain r_1 ... r_k
        if tail[u] <= 1e-15:  # j >= u impossible: r_u stays |0>, and so do all later r's
            break  # nothing more to prepare
        if u == 1:  # first bit from scratch
            ops.append(_u(right[0], _RY(_split_angle(tail[1]))))  # P(r_1 = 1) = P(j >= 1)
        else:  # later bits conditioned on the previous one (known |0> target)
            ops += _cry_on_zero(right[u - 2], right[u - 1], _split_angle(tail[u] / tail[u - 1]))  # P(j>=u | j>=u-1)
    for u in range(1, k + 1):  # rung copies l_{k+1-u} = NOT r_u
        v = k + 1 - u  # left index paired with r_u
        if v > a:  # left arm too short: this left bit does not exist (then r_u is always 1)
            continue  # nothing to write
        if u > b or tail[u] <= 1e-15:  # r_u is always 0, so l_v is always 1
            ops.append(_u(left[v - 1], _BRYAN_X))  # deterministic bit, no CZ needed
        elif tail[u] < 1 - 1e-15:  # r_u genuinely random: copy its complement across the rung
            ops += _cnot(right[u - 1], left[v - 1]) + [_u(left[v - 1], _BRYAN_X)]  # l_v = 1 XOR r_u
        # otherwise r_u is always 1 and l_v correctly stays |0>
    ops += dicke_unitary_ops(list(reversed(right)), min(k, b))  # BE on the right arm (head r_1 = last element)
    ops += dicke_unitary_ops(list(reversed(left)), min(k, a))  # BE on the left arm; runs in parallel
    return ops


def _subtree_sizes(root, children):
    """Subtree sizes s(v) for a rooted tree given as {node: [children]}."""
    order = [root]  # BFS order
    for v in order:  # the list grows while we iterate
        order.extend(children.get(v, []))  # append children
    size = {}  # node -> subtree size
    for v in reversed(order):  # leaves first
        size[v] = 1 + sum(size[c] for c in children.get(v, []))  # itself plus its children's subtrees
    return size


def w_tree_ops(root, children):
    """|W_N> on a rooted tree with 2N-3 CZs (Section 2.1)."""
    size = _subtree_sizes(root, children)  # s(v)
    kids = children.get(root, [])  # root's children in serving order
    if not kids:  # single qubit: |W_1> = |1>
        return [_u(root, _BRYAN_X)]  # just flip it
    first = kids[0]  # first child, served with the 1-CZ product-state split
    p = size[first] / size[root]  # probability that the excitation goes to the first child's subtree
    ops = [_u(root, _RY(_split_angle(1 - p)))]  # root = sqrt(p)|0> + sqrt(1-p)|1>
    ops += _cnot(root, first) + [_u(first, _BRYAN_X)]  # -> sqrt(p)|01> + sqrt(1-p)|10>
    stack = [root]  # depth-first traversal keeps every qubit's instruction order valid
    while stack:  # visit every node once
        v = stack.pop()  # current owner of an excitation
        remaining = size[v] - (size[first] if v == root else 0)  # mass still owned by v
        for c in children.get(v, []):  # serve the children in order
            if v == root and c == first:  # already served by the 1-CZ split
                continue  # skip it
            ops += _transfer(v, c, size[c] / remaining)  # hand child c its share
            remaining -= size[c]  # v now owns less
        stack.extend(reversed(children.get(v, [])))  # children then process their own subtrees
    return ops


# %% [markdown]
# ## 3. Compilation: native gates, layering, dynamical decoupling
#
# 1. **Merge.** All single-qubit matrices on a qubit between two of its CZs are multiplied into one unitary and emitted
#    as a single `PhasedXZGate` (every single-qubit unitary is a `PhasedXZ` up to global phase). This minimises the
#    single-qubit gate count and error.
# 2. **Layer.** CZs form a DAG (two CZs depend on each other if they share a qubit). We compute the ASAP depth $D$ and then
#    place every CZ **as late as possible** (ALAP) within $D$ layers. Moments alternate *1q layer / CZ layer*, which matches
#    how Google calibrates parallel CZ moments. ALAP keeps each qubit in $|0\rangle$ for as long as possible, and $|0\rangle$ is
#    immune to both $T_1$ decay and dephasing, so idle error only accumulates after a qubit's first gate.
# 3. **Dynamical decoupling (DD).** Once a qubit is active, every 1q slot in which it would otherwise idle receives an $X$ or $Y$
#    pulse (alternating, XY4-style). The last slot of each idle window receives the inverse of the accumulated Pauli product,
#    merged with the real gate that belongs there. Paulis multiply to a Pauli and the window contains no CZ on that qubit, so
#    the logical unitary is **exactly** unchanged, while quasi-static $Z$ errors (detuning, spectator phases from neighbouring CZs)
#    are echoed out. The challenge text flags DD as important; our emulator (later section) quantifies the trade-off, since each
#    pulse costs about one single-qubit error.

# %%
def _is_identity(u, atol=1e-9):
    """True if u is the identity up to a global phase."""
    phase = u[0, 0] if abs(u[0, 0]) > 0.5 else 1.0  # an identity-like matrix has |u00| ~ 1
    return np.allclose(u / phase, _I2, atol=atol)  # compare after removing the phase


def _phxz(q, u):
    """Native gate for an arbitrary single-qubit unitary."""
    return cirq.PhasedXZGate.from_matrix(u).on(q)  # exact up to a global phase


def compile_layers(ops, dd=True):
    """Instruction list -> cirq.Circuit with alternating PhasedXZ / CZ moments (ALAP, optional DD)."""
    pending = {}  # qubit -> product of 1q matrices since its last CZ
    last = {}  # qubit -> index of its most recent CZ node
    nodes = []  # CZ nodes: qubits, pre-gates, DAG edges
    for kind, a, b in ops:  # walk the instructions in program order
        if kind == 'U':  # single-qubit matrix b on qubit a
            pending[a] = b @ pending.get(a, _I2)  # left-multiply: later gates act after earlier ones
            continue  # nothing else to do
        nodes.append({'qubits': (a, b),  # a CZ node
                      'pre': {a: pending.pop(a, None), b: pending.pop(b, None)},  # 1q gates right before it
                      'preds': {last[q] for q in (a, b) if q in last},  # CZs that must come first
                      'succs': set()})  # filled below
        last[a] = last[b] = len(nodes) - 1  # this CZ is now the latest on both qubits
    trailing = pending  # 1q gates after a qubit's last CZ (or the only gates of CZ-free qubits)
    for i, nd in enumerate(nodes):  # build successor lists
        for p in nd['preds']:  # every predecessor ...
            nodes[p]['succs'].add(i)  # ... points forward to i
    asap = [0] * len(nodes)  # earliest CZ layer (1-based)
    for i, nd in enumerate(nodes):  # program order is a topological order
        asap[i] = 1 + max((asap[p] for p in nd['preds']), default=0)  # one after the latest predecessor
    depth = max(asap, default=0)  # number of CZ layers
    layer = [0] * len(nodes)  # ALAP layer
    for i in range(len(nodes) - 1, -1, -1):  # reverse topological order
        layer[i] = min((layer[s] for s in nodes[i]['succs']), default=depth + 1) - 1  # one before the earliest successor
    n_mom = 2 * depth + 1  # moments: 1q, CZ, 1q, CZ, ..., 1q
    slots = [dict() for _ in range(n_mom)]  # 1q moments: qubit -> matrix
    cz_moments = [[] for _ in range(n_mom)]  # CZ moments
    q_layers = {}  # qubit -> list of (CZ layer, pre-gate)
    for i, nd in enumerate(nodes):  # place the CZs
        cz_moments[2 * layer[i] - 1].append(cirq.CZ(*nd['qubits']))  # CZ layer t lives in moment 2t-1
        for q in nd['qubits']:  # remember each qubit's timeline
            q_layers.setdefault(q, []).append((layer[i], nd['pre'][q]))  # its pre-gate goes right before this CZ
    for q in set(q_layers) | set(trailing):  # every qubit that appears
        seq = sorted(q_layers.get(q, []), key=lambda t: t[0])  # its CZs in time order
        if not seq:  # no CZ at all: put its only gate in the final 1q moment (stay |0> as long as possible)
            u = trailing.get(q)  # its merged gate
            if u is not None and not _is_identity(u):  # skip identities
                slots[n_mom - 1][q] = u  # last moment
            continue  # done with this qubit
        t0, u0 = seq[0]  # first CZ of this qubit
        if u0 is not None:  # gates before the first CZ (the qubit is |0> before them, so no DD there)
            slots[2 * t0 - 2][q] = u0  # 1q moment right before CZ layer t0
        windows = [(seq[j][0], seq[j + 1][0], seq[j + 1][1], 'mid') for j in range(len(seq) - 1)]  # between CZs
        windows.append((seq[-1][0], depth + 1, trailing.get(q), 'tail'))  # after the last CZ, until the end
        for t_a, t_b, u, kind in windows:  # each idle window of an active qubit
            moms = [2 * t for t in range(t_a, t_b)]  # its 1q moments
            u = _I2 if u is None else u  # net single-qubit operation required in this window
            if not dd or len(moms) < 2:  # no idle CZ layer inside (or DD disabled)
                if not _is_identity(u):  # only emit real gates
                    slots[moms[-1] if kind == 'mid' else moms[0]][q] = u  # just before next CZ / just after last CZ
                continue  # next window
            pulses = [(_BRYAN_X if i % 2 == 0 else _BRYAN_Y) for i in range(len(moms) - 1)]  # X, Y, X, Y, ...
            acc = _I2  # running product of the pulses
            for p in pulses:  # accumulate
                acc = p @ acc  # later pulses act after earlier ones
            pulses.append(acc.conj().T)  # the final slot undoes the product: window total = identity
            if kind == 'mid':  # the real gate must act right before the next CZ
                pulses[-1] = u @ pulses[-1]  # merge it into the final slot
            else:  # tail window: apply the real gate first, then decouple
                pulses[0] = pulses[0] @ u  # merge it into the first slot
            for mom, p in zip(moms, pulses):  # emit
                if not _is_identity(p):  # identity slots stay empty
                    slots[mom][q] = p  # one PhasedXZ per qubit per moment
    moments = []  # assemble the circuit
    for i in range(n_mom):  # in time order
        ops_i = [_phxz(q, u) for q, u in slots[i].items()] if i % 2 == 0 else cz_moments[i]  # 1q or CZ moment
        if ops_i:  # drop empty moments
            moments.append(cirq.Moment(ops_i))  # one homogeneous moment
    return cirq.Circuit(moments)  # final native circuit


# %% [markdown]
# ## 4. Calibration model, success-probability estimate, placement search
#
# **Pessimistic error rates.** Each calibration entry is a measured mean $\hat e$ with statistical uncertainty $\sigma$. Choosing the
# minimum of many noisy estimates systematically selects under-estimates (the "winner's curse"), so we rank with
# $e=\hat e+\beta\sigma$, $\beta=1$ (a one-standard-error penalty), following the plan's uncertainty-aware placement.
#
# **Predicted log-fidelity** of a compiled candidate (higher is better):
# $$\log F=\sum_{\mathrm{CZ}}\log(1-e_{\mathrm{CZ}})+\sum_{\mathrm{1q}}\log(1-e_{\mathrm{1q}})+\sum_{q}\log\!\big(1-[(1-p)e^{0\to1}_q+p\,e^{1\to0}_q]\big)-\sum_q\frac{t^{\mathrm{active}}_q}{T_{\mathrm{idle}}}.$$
# The readout term uses the exact excitation probability $p=n/N$ of a Dicke qubit (asymmetric readout matters: $e^{1\to0}\approx2e^{0\to1}$).
# $t^{\mathrm{active}}_q$ is the time from the qubit's first gate to measurement (25 ns per 1q moment, 42 ns per CZ moment), and
# $T_{\mathrm{idle}}=60\,\mu$s is a nominal combined $T_1$/dephasing time for Willow-class qubits with DD (it only has to rank
# candidates, not predict absolute numbers).
#
# **Search spaces** (exhaustive over starting points, pruned by cheap proxies, then compiled and scored exactly):
# * *Trees* ($k=1$): for every root and three depth penalties, a Prim-like greedy growth adds the cheapest frontier qubit
#   (2 CZs on its edge + readout + single-qubit cost + depth penalty).
# * *Paths* (BE): beam search over simple paths, with position-dependent weights equal to the actual number of CZs / 1q gates
#   the construction places on each edge / qubit.
# * *Ladders* (split): every straight $2\times k$ ladder on the grid, extended into two disjoint arms by beam search.

# %%
T_1Q = 25e-9  # duration of a single-qubit moment (s)
T_CZ = 42e-9  # duration of a CZ moment (s)
T_IDLE = 60e-6  # nominal effective coherence time used to price idling (s)


class Calibration:
    """Pessimistic error rates and the usable coupling graph built from the four calibration dictionaries."""

    def __init__(self, cz, sq, e0, e1, beta=1.0):
        def pess(v):  # mean + beta * sigma, clipped to a sane range
            mean, std = v  # calibration tuple
            return float(min(max(mean + beta * std, 1e-6), 0.5))  # keeps the logarithms finite

        self.sq = {q: pess(v) for q, v in sq.items()}  # usable qubits = those with single-qubit data
        med0 = float(np.median([v[0] for v in e0.values()])) if e0 else 0.01  # fallback 0->1 readout error
        med1 = float(np.median([v[0] for v in e1.values()])) if e1 else 0.02  # fallback 1->0 readout error
        self.e0 = {q: pess(e0[q]) if q in e0 else med0 for q in self.sq}  # 0 -> 1 readout error
        self.e1 = {q: pess(e1[q]) if q in e1 else med1 for q in self.sq}  # 1 -> 0 readout error
        self.cz = {}  # frozenset pair -> CZ error
        for pair, v in cz.items():  # only couplers whose two qubits are usable
            a, b = tuple(pair)  # unpack (tuple or frozenset)
            if a in self.sq and b in self.sq and a != b:  # both ends usable
                key = frozenset((a, b))  # orientation-free key
                self.cz[key] = min(self.cz.get(key, 1.0), pess(v))  # keep the better value if listed twice
        self.adj = {q: set() for q in self.sq}  # adjacency of the usable graph
        for key in self.cz:  # fill it
            a, b = tuple(key)  # endpoints
            self.adj[a].add(b)  # undirected ...
            self.adj[b].add(a)  # ... edge

    def edge_cost(self, a, b):
        """-log(1 - CZ error) for coupler (a, b)."""
        return -math.log(1 - self.cz[frozenset((a, b))])  # additive cost

    def ro_cost(self, q, p):
        """-log(1 - readout error) for a qubit that is |1> with probability p."""
        return -math.log(1 - ((1 - p) * self.e0[q] + p * self.e1[q]))  # mixture of both error types

    def sq_cost(self, q):
        """-log(1 - single-qubit error)."""
        return -math.log(1 - self.sq[q])  # additive cost


def estimate_log_fidelity(circuit, cal, data_qubits, p_exc):
    """Predicted log success probability of a compiled circuit (see the formula above)."""
    total = 0.0  # running log-fidelity
    first = {}  # qubit -> index of its first moment
    durations = []  # moment durations
    for i, moment in enumerate(circuit):  # scan the moments
        is_cz = any(len(op.qubits) == 2 for op in moment)  # moments are homogeneous: CZ or 1q
        durations.append(T_CZ if is_cz else T_1Q)  # its duration
        for op in moment:  # gate errors
            for q in op.qubits:  # remember when each qubit becomes active
                first.setdefault(q, i)  # first occurrence only
            if len(op.qubits) == 2:  # CZ
                total += math.log(1 - cal.cz[frozenset(op.qubits)])  # its error
            else:  # PhasedXZ
                total += math.log(1 - cal.sq[op.qubits[0]])  # its error
    cum = np.cumsum([0.0] + durations)  # cumulative time at each moment boundary
    for q, i in first.items():  # idling from the first gate until measurement
        total -= (cum[-1] - cum[i]) / T_IDLE  # linearised decay
    for q in data_qubits:  # readout
        total -= cal.ro_cost(q, p_exc)  # its error
    return total


def _tree_children(parent):
    """{child: parent} -> {node: [children sorted by decreasing subtree height]}."""
    children = {}  # node -> children
    for v, u in parent.items():  # invert the parent map
        children.setdefault(u, []).append(v)  # u is v's parent
    height = {}  # memoised subtree heights

    def h(v):  # height of v's subtree
        if v not in height:  # compute once
            height[v] = 1 + max((h(c) for c in children.get(v, [])), default=0)  # 1 + tallest child
        return height[v]  # cached value

    for v in list(children):  # order every child list
        children[v].sort(key=lambda c: (-h(c), str(c)))  # tallest first (critical path first), deterministic ties
    return children


def grow_tree(cal, root, N, p_exc, depth_weight):
    """Prim-like greedy N-node tree from `root`; returns {child: parent} or None."""
    in_tree = {root}  # nodes already in the tree
    parent = {}  # child -> parent
    depth = {root: 0}  # tree depth of each node
    heap = []  # frontier edges keyed by cost

    def push(u):  # add u's frontier edges
        for v in cal.adj[u]:  # neighbours
            if v not in in_tree:  # only new nodes
                c = (2 * cal.edge_cost(u, v) + cal.ro_cost(v, p_exc) + 4 * cal.sq_cost(v)
                     + depth_weight * (depth[u] + 1))  # 2 CZs + readout + ~4 1q gates + depth penalty
                heapq.heappush(heap, (c, str(v), v, u))  # str(v) breaks ties deterministically

    push(root)  # start from the root
    while len(in_tree) < N and heap:  # grow until N nodes
        _, _, v, u = heapq.heappop(heap)  # cheapest frontier edge
        if v in in_tree:  # stale entry
            continue  # skip
        in_tree.add(v)  # accept v
        parent[v] = u  # reached via u
        depth[v] = depth[u] + 1  # one level deeper
        push(v)  # extend the frontier
    return parent if len(in_tree) == N else None  # None if the component is too small


def beam_paths(cal, N, edge_w, node_w, p_exc, beam=200):
    """Beam search for low-cost simple paths of N qubits with position-dependent weights."""
    beams = [(node_w[0] * cal.sq_cost(q) + cal.ro_cost(q, p_exc), (q,)) for q in sorted(cal.sq, key=str)]  # all starts
    for pos in range(1, N):  # append one position at a time
        cand = {}  # dedupe partial paths by (qubit set, end)
        for cost, path in beams:  # expand every beam
            u = path[-1]  # current end
            for v in cal.adj[u]:  # possible next qubit
                if v in path:  # simple paths only
                    continue  # skip
                c = cost + edge_w[pos - 1] * cal.edge_cost(u, v) + node_w[pos] * cal.sq_cost(v) + cal.ro_cost(v, p_exc)  # new cost
                key = (frozenset(path) | {v}, v)  # equivalent partial paths
                if key not in cand or cand[key][0] > c:  # keep the best representative
                    cand[key] = (c, path + (v,))  # extended path
        beams = heapq.nsmallest(beam, cand.values(), key=lambda t: t[0])  # prune
        if not beams:  # dead end everywhere
            return []  # no path of this length
    return beams  # sorted best-first


def extend_path(cal, prefix, length, forbidden, p_exc, w_edge, w_node, beam=30):
    """Beam-search continuations of `prefix` to `length` qubits avoiding `forbidden`."""
    beams = [(0.0, tuple(prefix))]  # start from the prefix
    while len(beams[0][1]) < length:  # until long enough
        cand = {}  # dedupe
        for cost, path in beams:  # expand
            u = path[-1]  # current end
            for v in cal.adj[u]:  # neighbours
                if v in path or v in forbidden:  # keep the arms disjoint and simple
                    continue  # skip
                c = cost + w_edge * cal.edge_cost(u, v) + w_node * cal.sq_cost(v) + cal.ro_cost(v, p_exc)  # new cost
                key = (frozenset(path) | {v}, v)  # equivalence class
                if key not in cand or cand[key][0] > c:  # keep the best
                    cand[key] = (c, path + (v,))  # extended path
        beams = heapq.nsmallest(beam, cand.values(), key=lambda t: t[0])  # prune
        if not beams:  # cannot extend
            return []  # failure
    return beams  # best-first continuations


def ladder_candidates(cal, k, p_exc, w_edge, w_node):
    """All straight 2 x k ladders: right row r_1..r_k and a left row with l_{k+1-u} next to r_u."""
    out = []  # (cost, left_head, right_head)
    for q in sorted(cal.sq, key=str):  # r_1
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):  # direction of the right row
            for s in ((d[1], d[0]), (-d[1], -d[0])):  # side on which the left row lies
                right = [cirq.GridQubit(q.row + u * d[0], q.col + u * d[1]) for u in range(k)]  # r_1..r_k
                left = [cirq.GridQubit(r.row + s[0], r.col + s[1]) for r in reversed(right)]  # l_1..l_k (reversed)
                if any(x not in cal.sq for x in right + left):  # all qubits must be usable
                    continue  # skip
                edges = (list(zip(right, right[1:])) + list(zip(left, left[1:]))
                         + [(right[u], left[k - 1 - u]) for u in range(k)])  # rows + rungs
                if any(frozenset(e) not in cal.cz for e in edges):  # all couplers must be usable
                    continue  # skip
                cost = (sum(w_edge * cal.edge_cost(*e) for e in edges)
                        + sum(w_node * cal.sq_cost(x) + cal.ro_cost(x, p_exc) for x in right + left))  # proxy cost
                out.append((cost, left, right))  # keep
    out.sort(key=lambda t: t[0])  # cheapest first
    return out


def path_profile(N, k):
    """CZ count per edge and gate count per position of the BE construction (weights for the path search)."""
    qs = cirq.LineQubit.range(N)  # abstract line
    circ = compile_layers(dicke_path_ops(qs, k), dd=False)  # compile once
    pos = {q: i for i, q in enumerate(qs)}  # qubit -> position index
    edge_w = [0] * max(N - 1, 1)  # CZs per edge
    node_w = [0] * N  # 1q gates per position
    for op in circ.all_operations():  # tally
        if len(op.qubits) == 2:  # CZ (always nearest-neighbour)
            edge_w[min(pos[q] for q in op.qubits)] += 1  # its edge index
        else:  # PhasedXZ
            node_w[pos[op.qubits[0]]] += 1  # its position
    return edge_w, node_w


# %% [markdown]
# ## 5. `_bryan_create_dicke_state`
#
# Candidate families per $k=\min(n,N-n)$:
#
# | $k$ | Families tried |
# | --- | --- |
# | 0 | No entangling gates: the $N$ qubits with the best relevant readout error, optionally flipped |
# | 1 | Excitation trees (all roots × 3 depth penalties) |
# | $\ge2$ | BE paths (top beams) **and** two-arm split ladders (top ladders × arm extensions) |
#
# Every candidate is compiled and scored with the predicted log-fidelity; the best one with at most 1750 moments is returned.
# A wall-clock budget (30 s, leaving a 2x margin under the 60 s limit for slower machines) stops the search early. Impossible requests
# ($n<0$, $n>N$, more qubits than usable ones) raise `NotImplementedError`.

# %%
def _x_all(qubits):
    """Instructions flipping every qubit (complement trick)."""
    return [_u(q, _BRYAN_X) for q in qubits]  # merged into the final 1q layer by the compiler


def generate_candidates(N, n, cal, *, dd=True, deadline=None, max_trees=40, max_paths=10, max_ladders=12):
    """Yield exact candidates (circuit, qubit_list, kind) for |D^(N)_n> on the calibrated device."""
    deadline = deadline if deadline is not None else time.monotonic() + 30.0  # default budget
    k = min(n, N - n)  # prepare the lighter of the two complementary states
    flip = n > N - n  # append X everywhere afterwards if we prepared the complement
    p_exc = n / N  # excitation probability of each Dicke qubit (for readout pricing)
    if k == 0:  # computational basis state: no entanglement needed
        qs = sorted(cal.sq, key=lambda q: ((cal.e1[q] if flip else cal.e0[q]) + cal.sq[q], str(q)))[:N]  # best readout
        gate = cirq.PhasedXZGate(x_exponent=1 if flip else 0, z_exponent=0, axis_phase_exponent=0)  # X or identity
        yield cirq.Circuit(cirq.Moment(gate.on(q) for q in qs)), qs, 'trivial'  # explicit gate keeps qubits visible
        return  # only candidate
    if k == 1:  # W-state trees
        seen = set()  # avoid compiling duplicate trees
        proxies = []  # (proxy cost, root name, root, parent map)
        for w in (0.0, 0.01, 0.03):  # depth penalties: from pure quality to compact trees
            for r in sorted(cal.sq, key=str):  # every root
                parent = grow_tree(cal, r, N, p_exc, w)  # greedy tree
                if parent is None:  # component too small
                    continue  # skip
                key = (r, frozenset(parent.items()))  # identity of the tree
                if key in seen:  # duplicate
                    continue  # skip
                seen.add(key)  # remember it
                cost = (sum(2 * cal.edge_cost(v, u) for v, u in parent.items())
                        + sum(cal.ro_cost(q, p_exc) for q in [r, *parent]))  # cheap proxy: CZs + readout
                proxies.append((cost, str(r), r, parent))  # keep
        proxies.sort(key=lambda t: (t[0], t[1]))  # best proxies first
        for _, _, r, parent in proxies[:max_trees]:  # compile the most promising trees
            qs = [r, *parent]  # its qubits
            ops = w_tree_ops(r, _tree_children(parent)) + (_x_all(qs) if flip else [])  # exact construction
            yield compile_layers(ops, dd=dd), qs, 'tree'  # candidate
            if time.monotonic() > deadline:  # out of time
                return  # stop
        return  # trees beat paths for k = 1 (same CZ count, logarithmic instead of linear depth)
    edge_w, node_w = path_profile(N, k)  # per-position weights of the BE path
    for _, path in beam_paths(cal, N, edge_w, node_w, p_exc)[:max_paths]:  # best paths
        ops = dicke_path_ops(list(path), k) + (_x_all(path) if flip else [])  # exact construction
        yield compile_layers(ops, dd=dd), list(path), 'path'  # candidate
        if time.monotonic() > deadline:  # out of time
            return  # stop
    a, b = N // 2, N - N // 2  # balanced arms (b >= a >= k because k <= N/2)
    ew, nw = path_profile(b, k)  # typical arm weights
    w_edge = sum(ew) / max(len(ew), 1)  # average CZs per arm edge
    w_node = sum(nw) / max(len(nw), 1)  # average 1q gates per arm qubit
    for _, left0, right0 in ladder_candidates(cal, k, p_exc, w_edge, w_node)[:max_ladders]:  # best ladders
        used = set(left0) | set(right0)  # ladder qubits
        for _, right in extend_path(cal, right0, b, used, p_exc, w_edge, w_node)[:2]:  # right-arm options
            for _, left in extend_path(cal, left0, a, used | set(right), p_exc, w_edge, w_node)[:1]:  # best left arm
                qs = list(left) + list(right)  # all qubits
                ops = split_ops(list(left), list(right), k) + (_x_all(qs) if flip else [])  # exact construction
                yield compile_layers(ops, dd=dd), qs, 'split'  # candidate
        if time.monotonic() > deadline:  # out of time
            return  # stop


def _bryan_create_dicke_state(
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
    start = time.monotonic()  # the challenge counts > 60 s as NotImplementedError
    N, n = int(num_qubits), int(num_excitations)  # accept numpy integers too
    if N < 1 or n < 0 or n > N:  # no such Dicke state
        raise NotImplementedError(f'invalid (N, n) = ({N}, {n})')  # explicit refusal
    cal = Calibration(cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)  # pessimistic model
    if N > len(cal.sq):  # not enough working qubits
        raise NotImplementedError('not enough usable qubits')  # cannot embed
    best = None  # (predicted log-fidelity, circuit, qubits)
    for circ, qs, _ in generate_candidates(N, n, cal, deadline=start + 30.0):  # exact candidates
        if len(circ) > 1750:  # would not be run on the hardware
            continue  # discard
        score = estimate_log_fidelity(circ, cal, qs, n / N)  # predicted log success probability
        if best is None or score > best[0]:  # keep the best
            best = (score, circ, qs)  # new incumbent
    if best is None:  # nothing embeddable / everything too deep
        raise NotImplementedError('no embedding within the 1750-moment limit')  # explicit refusal
    return best[1], set(best[2])  # circuit and the qubits that hold the state


# %% [markdown]
# ## 6. Bonus: shallowest circuits (adaptive circuits allowed, no noise)
#
# The bonus scores depth only (`len(circuit)`), on the ideal Willow105 connectivity. We considered the adaptive designs
# in the plan (measurement-based fan-out and the "binary index → one-hot → $X$-basis measurement → $Z$ correction" W-state
# scheme). For $N\le20$ they lose to our unitary constructions: a measurement moment plus a classically controlled
# correction moment already cost 2 moments, the constant-depth fan-out gadget needs several more, and the binary→one-hot
# decoder needs multi-controlled gates routed across the grid, whereas the excitation tree reaches $N$ qubits in
# $\approx4\lceil\log_2N\rceil+1$ moments. Moreover mid-circuit measurements must cover whole readout diagonals
# ($\text{row}+\text{col}=$ const), which forces ancillas onto separate diagonals and lengthens routing. We therefore submit
# **non-adaptive** circuits (explicitly allowed) and spend the effort on depth:
#
# * **W states:** a *broadcast tree*. In round $t$ every informed qubit hands the excitation to one uninformed neighbour, so the
#   informed set can double every two CZ layers; we build the tree by greedy bipartite matching round by round from every
#   possible root and keep the shallowest compiled circuit.
# * **$k\ge2$:** the two-arm split construction (both arms in parallel) and the BE path, compiled on the ideal device, keeping
#   the shallower.
# * **Packing:** for the bonus, single-qubit gates may share moments with CZs on other qubits (no hardware calibration constraint
#   applies), so we re-insert every operation at the earliest possible moment and keep whichever version is shorter.

# %%
def _ideal_calibration():
    """Uniform 'calibration' covering every Willow105 qubit and coupler (the bonus is noise-free)."""
    device = cirq_google.Willow105  # the target device
    qubits = sorted(device.metadata.qubit_set)  # all 105 qubits
    cz = {tuple(sorted(p)): (1e-3, 0.0) for p in device.metadata.qubit_pairs}  # every coupler, equal cost
    flat = {q: (1e-3, 0.0) for q in qubits}  # equal single-qubit and readout costs
    return Calibration(cz, flat, flat, flat, beta=0.0)  # deterministic, uniform model


def broadcast_tree(cal, root, N):
    """Greedy round-by-round matching tree: every informed node recruits at most one new neighbour per round."""
    informed = [root]  # informed nodes in order of recruitment
    parent = {}  # child -> parent
    center = (np.mean([q.row for q in cal.sq]), np.mean([q.col for q in cal.sq]))  # prefer central growth
    while len(informed) < N:  # until N qubits hold the excitation
        taken = set()  # qubits recruited this round
        new = []  # (child, parent) pairs of this round
        for u in list(informed):  # each informed node may recruit once
            options = [v for v in cal.adj[u] if v not in parent and v != root and v not in taken]  # free neighbours
            if not options or len(informed) + len(new) >= N:  # nothing to do / enough nodes
                continue  # next node
            v = min(options, key=lambda q: (len([w for w in cal.adj[q] if w not in parent and w != root]) == 0,
                                            (q.row - center[0]) ** 2 + (q.col - center[1]) ** 2, str(q)))  # central, non-dead-end
            taken.add(v)  # reserve it
            new.append((v, u))  # recruit
        if not new:  # stuck (cannot happen on a connected device with N <= 105)
            return None  # give up on this root
        for v, u in new:  # commit the round
            parent[v] = u  # tree edge
            informed.append(v)  # v can recruit from the next round on
    return parent  # {child: parent}


def pack_earliest(circuit):
    """Re-insert every operation at the earliest moment allowed by its qubits (mixed 1q/CZ moments)."""
    return cirq.Circuit(circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST)  # cirq does the packing


def _bryan_create_dicke_state_adaptive(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state, allowing for adaptive circuits.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.

    Returns:
        A Cirq circuit to produce Dicke state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested (N, n) combination is not implemented.
    """
    N, n = int(num_qubits), int(num_excitations)  # accept numpy integers
    if N < 1 or n < 0 or n > N or N > 105:  # impossible request
        raise NotImplementedError(f'invalid (N, n) = ({N}, {n})')  # explicit refusal
    cal = _ideal_calibration()  # all qubits, uniform costs
    k = min(n, N - n)  # complement trick, as in the hardware solution
    flip = n > N - n  # whether to flip everything at the end
    cands = []  # (circuit, qubits)
    if k == 1:  # W states: broadcast trees from every root
        for r in sorted(cal.sq, key=str):  # every possible root
            parent = broadcast_tree(cal, r, N)  # shallow tree
            if parent is None:  # failed
                continue  # skip
            qs = [r, *parent]  # its qubits
            ops = w_tree_ops(r, _tree_children(parent)) + (_x_all(qs) if flip else [])  # exact construction
            cands.append((compile_layers(ops, dd=False), qs))  # no DD in a noise-free setting
    else:  # k = 0 or k >= 2: reuse the hardware candidate generator on the ideal device
        for circ, qs, _ in generate_candidates(N, n, cal, dd=False, max_trees=5, max_paths=2, max_ladders=3):  # few suffice
            cands.append((circ, qs))  # keep
    if not cands:  # nothing found
        raise NotImplementedError('no embedding found')  # explicit refusal
    best = None  # (depth, circuit, qubits)
    for circ, qs in cands:  # pick the shallowest, trying both layouts
        for c in (circ, pack_earliest(circ)):  # layered and packed versions
            if best is None or len(c) < best[0]:  # strictly shallower
                best = (len(c), c, qs)  # new incumbent
    return best[1], set(best[2])  # circuit and data qubits
