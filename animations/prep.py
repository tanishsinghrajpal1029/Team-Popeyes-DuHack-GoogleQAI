"""Precompute the data every animation shows, from the real code and result CSVs.

  .venv/bin/python animations/prep.py              # every data file
  .venv/bin/python animations/prep.py merge bonus  # only these

Writes animations/data/{score,wtree,synth,layout,revised,transfer,coherent,merge,bonus}.json. Scenes read only these files, so
rendering never imports cirq. Fixed seeds throughout; calibration seed 0 is the one
every experiment reports on. The asserts pin the numbers quoted in docs/main.tex and
docs/writeup/main.tex.
"""

import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import main  # noqa: E402
import synth  # noqa: E402
from calib import make_calibration  # noqa: E402
from scoring import expected_score  # noqa: E402

OUT = os.path.join(HERE, "data")
SEED = 0


def _dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", os.path.join("animations", "data", name))


def _rc(q):
    return [q.row, q.col]


# ---- Idea 1: score ------------------------------------------------------------------------


def score():
    Ks = [2, 3, 4, 6, 10, 20]
    f, x = 0.9, 1e-3
    deltas = np.linspace(0, 0.03, 31)
    table = {str(K): [expected_score(K, f, 1e4 * f * d * d, x) for d in deltas] for K in Ks}
    assert abs(table["2"][0] - 51.4) < 0.1 and abs(table["2"][10] - 33.3) < 0.1
    assert abs(table["2"][-1] - 1.3) < 0.05 and abs(table["3"][0] - 8.15) < 0.01

    # density of z = chi^2 / K (f = 1) and the chance of a lucky draw z < 0.05 (score > 20)
    zs = np.linspace(0.002, 3, 400)
    dens = {str(K): (stats.chi2(K - 1).pdf(zs * K) * K).tolist() for K in (2, 20)}
    lucky = {str(K): float(stats.chi2(K - 1).cdf(0.05 * K)) for K in (2, 20)}
    _dump("score.json", dict(Ks=Ks, f=f, x=x, deltas=deltas.tolist(), expected=table,
                             z=zs.tolist(), density=dens, lucky=lucky))


# ---- Idea 2: W tree -----------------------------------------------------------------------


def wtree(cal):
    N = 10
    g = main._usable_graph(cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"])
    tree_cost = {q: main._qubit_cost(q, cal["sq_pauli_error_rate"], cal["zero_errors"],
                                     cal["one_errors"], 1 / N, n_sq=4)
                 for q in cal["sq_pauli_error_rate"]}
    root, rounds = main._best_w_tree(g, N, tree_cost, cal["zero_errors"], cal["one_errors"])

    # the chain baseline: the deployed code with the tree switched off
    chain_circ, chain_qs = main.build(N, 1, **cal, w_tree=False, dd=False)
    edges = {frozenset(op.qubits) for op in chain_circ.all_operations() if op.gate == main.cirq.CZ}
    sub = main.nx.Graph([tuple(e) for e in edges])
    ends = [q for q in sub.nodes if sub.degree(q) == 1]
    chain = main.nx.shortest_path(sub, ends[0], ends[1])
    assert len(chain) == N, "chain circuit is not a simple path"

    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "experiments/results/w_tree.csv")))
            if r["N"] == str(N)]
    num = {r["label"]: r for r in rows if r["seed"] == str(SEED)}
    tree_row, chain_row = num["w-tree"], num["chain w_tree=False"]
    assert int(tree_row["moments"]) < int(chain_row["moments"])
    # the docs quote the X term as a mean over calibration seeds 0-2
    x_mean = {lab: float(np.mean([float(r["x_inf_Mall"]) for r in rows if r["label"] == lab]))
              for lab in ("w-tree", "chain w_tree=False")}
    assert round(x_mean["w-tree"], 4) == 0.0042 and round(x_mean["chain w_tree=False"], 4) == 0.0075

    # subtree sizes give each split its share
    children = {}
    for r in rounds:
        for p, c in r:
            children.setdefault(p, []).append(c)

    def size(v):
        return 1 + sum(size(c) for c in children.get(v, []))

    _dump("wtree.json", dict(
        N=N, root=_rc(root),
        rounds=[[[_rc(p), _rc(c), size(c)] for p, c in r] for r in rounds],
        chain=[_rc(q) for q in chain],
        tree_moments=int(tree_row["moments"]), chain_moments=int(chain_row["moments"]),
        tree_x=x_mean["w-tree"], chain_x=x_mean["chain w_tree=False"],
        tree_cz=int(tree_row["cz"]), chain_cz=int(chain_row["cz"]),
    ))


# ---- Idea 3: synthesis --------------------------------------------------------------------


def synthesis():
    """Replays experiments/synth.py:synthesize for D^4_2 on a ladder, logging every step."""
    N, k, shape = 4, 2, "ladder"
    rng = np.random.default_rng(SEED)
    coords = synth.shape_coords(shape, N)
    edges = synth.shape_edges(coords)
    target = synth.dicke_vector(N, k)

    depth_log, found, history = [], None, []
    for L in range(1, 40):
        layers = synth.brickwork(edges, L)
        circ = synth.Circuit(N, layers)
        best = (math.inf, None, [])
        for _ in range(6):
            hist = []
            res = minimize(circ.loss_and_grad, rng.uniform(-math.pi, math.pi, circ.num_params),
                           args=(target,), jac=True, method="L-BFGS-B",
                           callback=lambda xk: hist.append(circ.loss_and_grad(xk, target)[0]),
                           options=dict(maxiter=4000, ftol=1e-18, gtol=1e-14, maxcor=30))
            if res.fun < best[0]:
                best = (res.fun, res.x, hist)
            if res.fun < synth.TOL:
                break
        depth_log.append(dict(L=L, cz=sum(map(len, layers)), infidelity=max(float(best[0]), 1e-16)))
        if best[0] < synth.TOL:
            found, history = (layers, best[1]), best[2]
            break
    layers, x = found
    start = [list(map(list, l)) for l in layers]

    steps = []  # each attempted deletion: (layer index, edge, accepted, layers after)
    improved = True
    while improved:
        improved = False
        order = [(li, e) for li, l in enumerate(layers) for e in l]
        rng.shuffle(order)
        for li, e in order:
            trial = synth.drop_cz(layers, li, e)
            f, xt = synth.solve(N, k, trial, rng, restarts=2, x0=x)
            ok = f < synth.TOL
            steps.append(dict(layer=int(li), edge=list(map(int, e)), accepted=bool(ok),
                              infidelity=max(float(f), 1e-16)))
            if ok:
                layers, x = synth.compact(trial, xt, N)
                steps[-1]["layers"] = [list(map(list, l)) for l in layers]
                improved = True
                break
    final_cz = sum(map(len, layers))
    deployed = [sum(map(len, main._SYNTH[c]["cz_layers"])) for c in ((7, 3), (8, 3), (8, 4))]
    assert deployed == [18, 30, 26], deployed
    assert final_cz == sum(map(len, main._SYNTH[(4, 2)]["cz_layers"])) == 5, final_cz

    _dump("synth.json", dict(
        N=N, k=k, coords=coords, edges=edges, depth_log=depth_log,
        loss=[max(float(v), 1e-16) for v in history], start_layers=start, steps=steps,
        final_layers=[list(map(list, l)) for l in layers],
        # docs/main.tex, section 6 table (D73, D83, D84 are the teammates' Bridges-2 circuits)
        table=dict(cases=["D^4_2", "D^5_2", "D^6_3", "D^7_3", "D^8_3", "D^8_4"],
                   search=[sum(map(len, main._SYNTH[c]["cz_layers"]))
                           for c in ((4, 2), (5, 2), (6, 3), (7, 3), (8, 3), (8, 4))],
                   textbook=[21, 35, 80, 98, 123, 162]),
    ))


# ---- Idea 4: placement --------------------------------------------------------------------


def layout(cal):
    cz, sq, e0, e1 = (cal[k] for k in ("cz_pauli_error_rate", "sq_pauli_error_rate",
                                       "zero_errors", "one_errors"))
    dev = main.cirq_google.Willow105.metadata
    g = main._usable_graph(cz, sq)
    cz_err = {frozenset(p): main._robust(v) for p, v in cz.items()}

    # beam search for a 6-qubit path, recording the best few partial paths after each step
    N, n = 6, 3
    node_cost = {q: main._qubit_cost(q, sq, e0, e1, n / N) for q in sq}
    # the beam is prefix-deterministic, so the length-L search's survivors are its frontier
    # after L - 1 steps
    snaps = [[[_rc(q) for q in p] for _, p in main._best_path(g, L, node_cost, keep=12)]
             for L in range(1, N + 1)]
    frontier = main._best_path(g, N, node_cost, keep=256)
    best_path = frontier[0][1]
    path_cost = frontier[0][0]
    median_cost = float(np.median([c for c, _ in frontier]))

    # readout matching: the best- and worst-matched couplers, read as a W2 pair
    def mismatch(a, b):
        return abs((e1[a][0] - e0[a][0]) - (e1[b][0] - e0[b][0]))

    def read_probs(a, b):  # P(read 10 | 10), P(read 01 | 01)
        return [(1 - e1[a][0]) * (1 - e0[b][0]), (1 - e0[a][0]) * (1 - e1[b][0])]

    pairs = sorted(g.edges, key=lambda p: mismatch(*p))
    matched, leaky = pairs[0], pairs[-1]

    # W2 pair scan: expected score of every coupler, as main._best_tiny_w ranks them
    compiled, lq, steps = main._tiny_program(2, [[(0, 1)]], False)
    sq_err = {q: main._robust(v) for q, v in sq.items()}
    scan = []
    for a, b in g.edges:
        es = [main._predict_tiny(steps, phys, 1, cz_err, sq_err, e0, e1, exact=False)
              for phys in ([a, b], [b, a])]
        scan.append([_rc(a), _rc(b), max(es)])
    picked_circ, picked_qs = main.build(2, 1, **cal, dd=False)
    generic_circ, generic_qs = main.build(2, 1, **cal, tiny=False, dd=False)

    def exact_score(qs):
        a, b = sorted(qs)
        return max(main._predict_tiny(steps, phys, 1, cz_err, sq_err, e0, e1, exact=True)
                   for phys in ([a, b], [b, a]))

    picked_E, generic_E = exact_score(picked_qs), exact_score(generic_qs)

    _dump("layout.json", dict(
        qubits=[_rc(q) for q in sorted(dev.qubit_set)],
        usable=[_rc(q) for q in sorted(g.nodes)],
        couplers=[[_rc(a), _rc(b), cz_err.get(frozenset((a, b)))]
                  for a, b in sorted(tuple(sorted(p)) for p in dev.qubit_pairs)],
        readout={f"{q.row},{q.col}": [e0[q][0], e1[q][0]] for q in sorted(g.nodes)},
        beam_snapshots=snaps, best_path=[_rc(q) for q in best_path],
        best_path_cost=path_cost, median_beam_cost=median_cost,
        matched=dict(pair=[_rc(q) for q in matched], read=read_probs(*matched),
                     e1=[e1[q][0] for q in matched]),
        leaky=dict(pair=[_rc(q) for q in leaky], read=read_probs(*leaky),
                   e1=[e1[q][0] for q in leaky]),
        w2_scan=scan, w2_picked=[_rc(q) for q in sorted(picked_qs)], w2_picked_E=picked_E,
        w2_generic=[_rc(q) for q in sorted(generic_qs)], w2_generic_E=generic_E,
    ))
    print(f"  W2: generic {generic_E:.1f} -> picked {picked_E:.1f} "
          f"(+{100 * (picked_E / generic_E - 1):.0f}%)")


# ---- docs/main.tex: the organizers' revised ranking ------------------------------------------


def revised():
    """E[S] (one draw) against 1/E[z + x] (the organizers' Oct 25 preliminary ranking)."""
    Ks = [2, 3, 4, 6, 10, 20]
    f, x = 0.9, 1e-3
    mean_S = [expected_score(K, f, 0.0, x) for K in Ks]
    inv_mean = [1 / ((K - 1) / (K * f) + x) for K in Ks]  # E[chi^2] = K - 1 with lambda = 0
    assert abs(mean_S[0] - 51.4) < 0.1

    # E[z + x] = (K - 1)/(K f) + lambda/(K f) + x, with chi^2/(K - 1) = 1 + lambda/(K - 1);
    # deployed rows of the docs/main.tex Results table (means over seeds 0-2, wf_*.csv)
    def budget(K, f, ratio, x):
        lam = (ratio - 1) * (K - 1)
        return dict(K=K, f=f, x=x, lam=lam, errors=(K - 1) / (K * f), imbalance=lam / (K * f))

    budgets = {"D^4_2": budget(6, 0.968, 1.019, 0.0058), "D^{10}_5": budget(252, 0.692, 1.046, 0.8695)}
    for name, b, want in (("D^4_2", budgets["D^4_2"], 1.13), ("D^{10}_5", budgets["D^{10}_5"], 0.42)):
        assert abs(1 / (b["errors"] + b["imbalance"] + b["x"]) - want) < 0.006, name

    # docs/main.tex, "The organizers' pairs under 1/E[z+x]" (gate noise / idle noise)
    pairs = dict(cases=["D^{10}_3", "D^{10}_5", "D^{15}_4"],
                 before=[[0.765, 0.639], [0.569, 0.421], [0.491, 0.318]],
                 deployed=[[0.776, 0.639], [0.603, 0.421], [0.546, 0.351]],
                 combined=[[0.722, 0.582], [0.582, 0.413], [0.447, 0.310]])
    _dump("revised.json", dict(Ks=Ks, f=f, x=x, mean_S=mean_S, inv_mean=inv_mean,
                               budgets=budgets, pairs=pairs))


# ---- docs/main.tex: one CZ cannot do a transfer -------------------------------------------


def transfer():
    """CZ on (cos a|0> + sin a|1>)(cos b|0> + sin b|1>): concurrence |sin 2a sin 2b|, sampled
    on a grid to check the closed form the scene draws."""
    import cirq
    for a in np.linspace(0, math.pi, 7):
        for b in np.linspace(0, math.pi, 7):
            psi = np.kron([math.cos(a), math.sin(a)], [math.cos(b), math.sin(b)])
            out = cirq.unitary(cirq.CZ) @ psi
            conc = 2 * abs(out[0] * out[3] - out[1] * out[2])
            assert abs(conc - abs(math.sin(2 * a) * math.sin(2 * b))) < 1e-12
    _dump("transfer.json", dict(w_optimal={"3": 3, "4": 5}))  # exhaustive search, docs/main.tex


# ---- docs/main.tex: coherent errors on the deployed D^4_2 ---------------------------------


def coherent():
    """Population error of the deployed D^4_2 under a relative R_y over-rotation eps and under a
    stray Z rotation eps on every qubit after each CZ layer: O(eps) against O(eps^2)."""
    N, k = 4, 2
    s = main._SYNTH[(N, k)]
    circ = synth.Circuit(N, s["cz_layers"])
    th = np.array(s["params"], float).reshape(-1, N)
    bits = (np.arange(2**N)[:, None] >> (N - 1 - np.arange(N))) & 1
    zsum = (1 - 2 * bits).sum(axis=1)  # eigenvalue of Z_1 + ... + Z_N

    def probs(over=0.0, z=0.0):
        psi = np.zeros(2**N, complex)
        psi[0] = 1
        for layer in range(len(th)):
            for q in range(N):
                psi = circ._ry(psi, q, th[layer, q] * (1 + over))
            if layer < len(circ.signs):
                psi = psi * circ.signs[layer] * np.exp(-0.5j * z * zsum)
        return np.abs(psi) ** 2

    ideal = probs()
    assert np.abs(ideal - synth.dicke_vector(N, k) ** 2).max() < 1e-8
    eps = np.logspace(-3, -1, 9)
    tv = {name: [0.5 * float(np.abs(probs(**{name: e}) - ideal).sum()) for e in eps]
          for name in ("over", "z")}
    # docs/main.tex: eps 1e-2 -> 1e-3 cuts the error 10x (over-rotation) and 100x (stray Z)
    assert 9 < tv["over"][4] / tv["over"][0] < 11 and 90 < tv["z"][4] / tv["z"][0] < 110
    _dump("coherent.json", dict(eps=eps.tolist(), tv=tv, d52_lam=[50, 9]))


# ---- docs/main.tex: the single-qubit merge pass -------------------------------------------


def merge(cal):
    """Bryan's D^15_4 before the merge pass, after it, and after sparse XY4: gate counts, error
    budgets and the per-moment timeline of the qubits with the most single-qubit gates."""
    import cirq
    sq_e = {q: v[0] for q, v in cal["sq_pauli_error_rate"].items()}
    cz_e = {frozenset(p): v[0] for p, v in cal["cz_pauli_error_rate"].items()}
    raw, _ = main._bryan_create_dicke_state(15, 4, **cal)
    merged = main._merge_single_qubit_runs(raw)
    dd = main._add_dd(merged)

    def stats_of(c):
        ops = list(c.all_operations())
        one = [op for op in ops if len(op.qubits) == 1]
        two = [op for op in ops if len(op.qubits) == 2]
        return dict(sq=len(one), cz=len(two), moments=len(c),
                    sq_err=sum(sq_e[op.qubits[0]] for op in one),
                    cz_err=sum(cz_e[frozenset(op.qubits)] for op in two))

    def timeline(c, q):
        row = ""
        for m in c:
            op = m.operation_at(q)
            row += "." if op is None else ("c" if len(op.qubits) == 2 else "s")
        return row

    st = {name: stats_of(c) for name, c in (("raw", raw), ("merged", merged), ("dd", dd))}
    assert (st["raw"]["sq"], st["merged"]["sq"], st["dd"]["sq"]) == (777, 276, 424), st
    assert round(st["raw"]["sq_err"], 2) == 0.33 and round(st["raw"]["cz_err"], 2) == 0.27
    assert st["raw"]["moments"] == st["merged"]["moments"] == st["dd"]["moments"]
    busy = sorted(raw.all_qubits(), key=lambda q: -sum(
        1 for op in raw.all_operations() if op.qubits == (q,)))[:4]
    busy.sort()
    _dump("merge.json", dict(
        stats=st, qubits=[_rc(q) for q in busy],
        timelines={name: [timeline(c, q) for q in busy]
                   for name, c in (("raw", raw), ("merged", merged), ("dd", dd))},
        # docs/main.tex, section 10: other cases (single-qubit gates before -> after)
        others={"D^{10}_5": [371, 230], "D^9_4": [286, 170]},
    ))


# ---- docs/main.tex: the bonus track --------------------------------------------------------


def bonus():
    """Numbers quoted in docs/main.tex, section 12 (from bonus_variants.csv and
    notebook_bonus.csv); the scene only draws them."""
    m0 = 7
    cols = [int("0" * (m0 - r) + "1" * r, 2) for r in range(3)]  # inputs of U_7 with r <= 2
    _dump("bonus.json", dict(
        first=37.15, leader=41.10, teammates=41.47, now=44.47,
        d157=dict(moments=160, arm_alone=147),
        blocks=[dict(name=r"U_7,\ r\le2", sweep=37, block=27), dict(name=r"U_6,\ r\le4", sweep=97, block=43)],
        cases=[["D^{12}_4", 88, 51], ["D^9_3", 54, 31], ["D^{15}_2", 46, 35],
               ["D^{15}_7", 160, 144], ["W_9", 14, 12]],
        m0=m0, fixed_columns=cols, remaining=dict(cases=12, points=8.0),
    ))


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["score", "wtree", "synthesis", "layout",
                            "revised", "transfer", "coherent", "merge", "bonus"]
    cal = make_calibration(SEED)
    for job in jobs:
        fn = globals()[job]
        fn(cal) if job in ("wtree", "layout", "merge") else fn()
