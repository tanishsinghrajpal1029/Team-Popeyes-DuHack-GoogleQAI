"""Precompute the data every animation shows, from the real code and result CSVs.

  .venv/bin/python animations/prep.py

Writes animations/data/{score,wtree,synth,layout}.json. Scenes read only these files, so
rendering never imports cirq. Fixed seeds throughout; calibration seed 0 is the one
every experiment reports on. The asserts pin the numbers quoted in docs/writeup/main.tex.
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
            if r["N"] == str(N) and r["seed"] == str(SEED)]
    num = {r["label"]: r for r in rows}
    tree_row, chain_row = num["w-tree"], num["chain w_tree=False"]
    assert int(tree_row["moments"]) < int(chain_row["moments"])

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
        tree_x=float(tree_row["x_inf_Mall"]), chain_x=float(chain_row["x_inf_Mall"]),
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
    assert final_cz == sum(map(len, main._SYNTH[(4, 2)]["cz_layers"])) == 5, final_cz

    _dump("synth.json", dict(
        N=N, k=k, coords=coords, edges=edges, depth_log=depth_log,
        loss=[max(float(v), 1e-16) for v in history], start_layers=start, steps=steps,
        final_layers=[list(map(list, l)) for l in layers],
        # docs/writeup/main.tex, Idea 3 table
        table=dict(cases=["D^4_2", "D^5_2", "D^6_3", "D^7_3", "D^8_3", "D^8_4"],
                   search=[5, 10, 15, 25, 31, 27], textbook=[21, 35, 80, 98, 123, 162]),
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


if __name__ == "__main__":
    cal = make_calibration(SEED)
    score()
    wtree(cal)
    synthesis()
    layout(cal)
