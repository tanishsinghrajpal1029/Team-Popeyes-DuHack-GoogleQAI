"""Exact numerical synthesis of small Dicke-state circuits from Ry + CZ.

Dicke states are real with positive amplitudes, and {Ry, CZ} is universal for real
states, so we only need one angle per qubit per layer. A circuit is
    Ry layer 0, CZ layer 0, Ry layer 1, CZ layer 1, ..., Ry layer L
with CZ layers drawn from a fixed edge set (a layout "shape": a path or a 2 x m ladder).
We minimize 1 - <D|psi>^2 with L-BFGS using an exact gradient (backpropagation through
real orthogonal gates), then greedily delete CZs while the circuit stays exact
(infidelity < TOL), warm-starting from the previous solution.

Results are JSON files in experiments/results/synth/, consumed by `export_table`, which
writes the table that main.py embeds.

Example:
  python experiments/synth.py --cases 4,2 5,2 6,3 --shapes path ladder --workers 8
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "synth")
TOL = 1e-13


# ---- shapes ------------------------------------------------------------------------------


def shape_coords(shape: str, N: int) -> list[tuple[int, int]]:
    """Grid coordinates of the N logical qubits (used later to embed the shape on Willow)."""
    if shape == "path":
        return [(0, i) for i in range(N)]
    if shape == "ladder":  # 2 rows, filled column by column: (0,0) (1,0) (0,1) (1,1) ...
        return [(i % 2, i // 2) for i in range(N)]
    raise ValueError(shape)


def shape_edges(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    idx = {c: i for i, c in enumerate(coords)}
    return [
        (idx[a], idx[b])
        for a, b in itertools.combinations(coords, 2)
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    ]


def brickwork(edges: list[tuple[int, int]], num_layers: int) -> list[list[tuple[int, int]]]:
    """Greedy edge colouring into matchings, cycled for `num_layers` CZ layers."""
    colours: list[list[tuple[int, int]]] = []
    for e in edges:
        for col in colours:
            if all(not set(e) & set(f) for f in col):
                col.append(e)
                break
        else:
            colours.append([e])
    return [list(colours[i % len(colours)]) for i in range(num_layers)]


# ---- real statevector circuit with exact gradient ----------------------------------------


def dicke_vector(N: int, k: int) -> np.ndarray:
    d = np.zeros(2**N)
    for c in itertools.combinations(range(N), k):
        d[sum(1 << (N - 1 - q) for q in c)] = 1.0  # cirq big-endian (symmetric anyway)
    return d / np.linalg.norm(d)


class Circuit:
    def __init__(self, N: int, cz_layers: list[list[tuple[int, int]]]):
        self.N = N
        self.cz_layers = [list(map(tuple, l)) for l in cz_layers]
        bits = (np.arange(2**N)[:, None] >> (N - 1 - np.arange(N))) & 1
        self.signs = [
            np.prod([1 - 2 * (bits[:, i] & bits[:, j]) for i, j in layer], axis=0)
            if layer else np.ones(2**N)
            for layer in self.cz_layers
        ]
        self.num_params = N * (len(self.cz_layers) + 1)

    def _ry(self, psi, q, theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        v = psi.reshape(2**q, 2, -1)
        out = np.empty_like(v)
        out[:, 0] = c * v[:, 0] - s * v[:, 1]
        out[:, 1] = s * v[:, 0] + c * v[:, 1]
        return out.reshape(-1)

    def state(self, params):
        psi = np.zeros(2**self.N)
        psi[0] = 1.0
        th = params.reshape(-1, self.N)
        for l in range(len(th)):
            for q in range(self.N):
                psi = self._ry(psi, q, th[l, q])
            if l < len(self.signs):
                psi = psi * self.signs[l]
        return psi

    def loss_and_grad(self, params, target):
        th = params.reshape(-1, self.N)
        psi = np.zeros(2**self.N)
        psi[0] = 1.0
        before = []  # state before each Ry gate
        for l in range(len(th)):
            for q in range(self.N):
                before.append(psi)
                psi = self._ry(psi, q, th[l, q])
            if l < len(self.signs):
                psi = psi * self.signs[l]
        a = float(target @ psi)
        grad = np.zeros_like(th)
        phi = target.copy()
        gi = len(before)
        for l in range(len(th) - 1, -1, -1):
            if l < len(self.signs):
                phi = phi * self.signs[l]
            for q in range(self.N - 1, -1, -1):
                gi -= 1
                # phi is the backward vector after this gate; d/dtheta Ry = Ry(theta + pi) / 2
                grad[l, q] = 0.5 * float(phi @ self._ry(before[gi], q, th[l, q] + math.pi))
                phi = self._ry(phi, q, -th[l, q])
        return 1 - a * a, (-2 * a * grad).reshape(-1)


def optimize(circ: Circuit, target, x0, maxiter=4000):
    res = minimize(
        circ.loss_and_grad, x0, args=(target,), jac=True, method="L-BFGS-B",
        options=dict(maxiter=maxiter, ftol=1e-18, gtol=1e-14, maxcor=30),
    )
    return res.fun, res.x


def solve(N, k, cz_layers, rng, restarts, x0=None):
    circ = Circuit(N, cz_layers)
    target = dicke_vector(N, k)
    best = (math.inf, None)
    tries = ([x0] if x0 is not None else []) + [
        rng.uniform(-math.pi, math.pi, circ.num_params) for _ in range(restarts)
    ]
    for x in tries:
        f, xs = optimize(circ, target, x)
        if f < best[0]:
            best = (f, xs)
        if f < TOL:
            break
    return best


# ---- structure search ---------------------------------------------------------------------


def drop_cz(cz_layers, li, e):
    new = [list(l) for l in cz_layers]
    new[li].remove(e)
    return new


def compact(cz_layers, params, N):
    """Remove empty CZ layers by merging the Ry layers on both sides (Ry's add)."""
    th = params.reshape(-1, N).copy()
    layers, rows = [], [th[0]]
    for l, layer in enumerate(cz_layers):
        if layer:
            layers.append(layer)
            rows.append(th[l + 1])
        else:
            rows[-1] = rows[-1] + th[l + 1]
    return layers, np.array(rows).reshape(-1)


def synthesize(N, k, shape, seed=0, max_layers=40, restarts=6, log=print):
    rng = np.random.default_rng(seed)
    coords = shape_coords(shape, N)
    edges = shape_edges(coords)
    t0 = time.time()
    # 1. smallest brickwork depth that reaches an exact solution
    found = None
    for L in range(1, max_layers + 1):
        layers = brickwork(edges, L)
        f, x = solve(N, k, layers, rng, restarts)
        log(f"  [{N},{k},{shape}] L={L} cz={sum(map(len, layers))} infid={f:.2e}")
        if f < TOL:
            found = (layers, x)
            break
    if found is None:
        return None
    layers, x = found
    # 2. greedy CZ pruning (warm start + a few random restarts)
    improved = True
    while improved:
        improved = False
        order = [(li, e) for li, l in enumerate(layers) for e in l]
        rng.shuffle(order)
        for li, e in order:
            trial = drop_cz(layers, li, e)
            f, xt = solve(N, k, trial, rng, restarts=2, x0=x)
            if f < TOL:
                layers, x = compact(trial, xt, N)
                log(f"  [{N},{k},{shape}] pruned -> cz={sum(map(len, layers))}")
                improved = True
                break
    # 3. polish
    f, x = optimize(Circuit(N, layers), dicke_vector(N, k), x, maxiter=20000)
    return dict(
        N=N, k=k, shape=shape, coords=coords, cz_layers=layers,
        params=[float(v) for v in x], infidelity=float(f),
        cz=sum(map(len, layers)), cz_layers_count=len(layers), seed=seed,
        seconds=round(time.time() - t0, 1),
    )


def _job(args):
    N, k, shape, seed = args
    lines = []
    res = synthesize(N, k, shape, seed=seed, log=lambda s: lines.append(s))
    os.makedirs(OUT, exist_ok=True)
    if res is not None:
        with open(os.path.join(OUT, f"D{N}_{k}_{shape}_s{seed}.json"), "w") as fh:
            json.dump(res, fh)
    summary = f"D^{N}_{k} {shape} seed={seed}: " + (
        f"cz={res['cz']} layers={res['cz_layers_count']} infid={res['infidelity']:.1e} ({res['seconds']}s)"
        if res else "FAILED"
    )
    return summary, lines


BEGIN = "# BEGIN SYNTH TABLE (generated by experiments/synth.py --export; do not edit)\n"
END = "# END SYNTH TABLE\n"


def export_table(main_path=os.path.join(os.path.dirname(HERE), "main.py")):
    """Write the best synthesized circuit per (N, k) into main.py between the markers.

    Best = fewest CZs, then fewest CZ layers. Every entry is re-verified first.
    """
    best = {}
    for name in sorted(os.listdir(OUT)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(OUT, name)) as fh:
            r = json.load(fh)
        circ = Circuit(r["N"], r["cz_layers"])
        infid = 1 - float(dicke_vector(r["N"], r["k"]) @ circ.state(np.array(r["params"]))) ** 2
        if infid > TOL:
            print(f"skip {name}: infidelity {infid:.1e}")
            continue
        key = (r["N"], r["k"])
        if key not in best or (r["cz"], r["cz_layers_count"]) < (best[key]["cz"], best[key]["cz_layers_count"]):
            best[key] = r
    lines = [BEGIN, "_SYNTH = {\n"]
    for (N, k), r in sorted(best.items()):
        lines.append(f"    ({N}, {k}): dict(  # {r['cz']} CZ, {r['shape']}, seed {r['seed']}\n")
        lines.append(f"        coords={[tuple(c) for c in r['coords']]!r},\n")
        lines.append(f"        cz_layers={[[tuple(e) for e in l] for l in r['cz_layers']]!r},\n")
        lines.append(f"        params={[float(p) for p in r['params']]!r},\n")
        lines.append("    ),\n")
    lines += ["}\n", END]
    with open(main_path) as fh:
        src = fh.read()
    head, rest = src.split(BEGIN)
    _, tail = rest.split(END)
    with open(main_path, "w") as fh:
        fh.write(head + "".join(lines) + tail)
    for (N, k), r in sorted(best.items()):
        print(f"D^{N}_{k}: {r['cz']} CZ ({r['shape']}, seed {r['seed']})")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="write the best results into main.py and exit")
    ap.add_argument("--cases", nargs="+", help="N,k pairs, e.g. 4,2 6,3")
    ap.add_argument("--shapes", nargs="+", default=["path", "ladder"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.export:
        export_table()
        return
    jobs = [
        (int(c.split(",")[0]), int(c.split(",")[1]), s, seed)
        for c in args.cases for s in args.shapes for seed in args.seeds
    ]
    with Pool(args.workers) as pool:
        for summary, lines in pool.imap_unordered(_job, jobs):
            if args.verbose:
                print("\n".join(lines))
            print(summary, flush=True)


if __name__ == "__main__":
    main_cli()
