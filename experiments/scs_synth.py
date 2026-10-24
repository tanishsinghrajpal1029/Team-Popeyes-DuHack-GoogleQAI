"""Exact Ry + CZ circuits for the Baertschi-Eidenbenz Split & Cyclic Shift blocks.

SCS_{m,k} acts on the k+1 consecutive qubits m-k .. m of a line (1-indexed), and its
input is always a sorted string |0^{w-l} 1^l> on that window (w = k+1). It must map
    l = 0, l = m             ->  unchanged
    1 <= l <= k, l < m       ->  sqrt(l/m) |0^{w-l} 1^l> + sqrt((m-l)/m) |0^{w-l-1} 1^l 0>
coherently, so it is an isometry on a (k+1)- or (k+2)-dimensional subspace. Synthesizing
it on a path of w qubits makes the whole B-E circuit nearest-neighbour: no routing.

The CZ pattern depends only on k (and on whether the window is the whole register, which
adds the all-ones input), so we prune it once per (k, kind) and then re-fit only the
angles for every m, warm-starting from the neighbouring m.

  python experiments/scs_synth.py --kmin 1 --kmax 4 --mmax 24
writes experiments/results/scs/scs_k<kmin>-<kmax>.json.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from synth import brickwork, compact, drop_cz  # noqa: E402

OUT = os.path.join(HERE, "results", "scs")
TOL = 1e-13


def basis(w: int, bits: str) -> np.ndarray:
    v = np.zeros(2**w)
    v[int(bits, 2)] = 1.0  # qubit 0 = leftmost = most significant (cirq order)
    return v


def scs_io(m: int, k: int, all_ones: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Input and target output vectors (columns) of SCS_{m,k} on its w = k+1 window.

    all_ones adds the input |1^w>. It occurs only for the second-group blocks
    SCS_{m,m-1} with m <= K (the window is the whole register and can be all ones),
    not for SCS_{K+1,K}, where at most K ones exist.
    """
    w = k + 1
    assert not all_ones or m == w
    ins, outs = [], []
    for l in range(0, (w if all_ones else k) + 1):
        src = basis(w, "0" * (w - l) + "1" * l)
        ins.append(src)
        if l == 0 or l == m:
            outs.append(src)
        else:
            moved = basis(w, "0" * (w - l - 1) + "1" * l + "0")
            outs.append(math.sqrt(l / m) * src + math.sqrt((m - l) / m) * moved)
    return np.array(ins).T, np.array(outs).T


class BatchCircuit:
    """Ry layers + CZ layers on a w-qubit path, applied to a batch of input states."""

    def __init__(self, w, cz_layers):
        self.w = w
        self.cz_layers = [list(map(tuple, l)) for l in cz_layers]
        bits = (np.arange(2**w)[:, None] >> (w - 1 - np.arange(w))) & 1
        self.signs = [
            (np.prod([1 - 2 * (bits[:, i] & bits[:, j]) for i, j in layer], axis=0)
             if layer else np.ones(2**w))[:, None]
            for layer in self.cz_layers
        ]
        self.num_params = w * (len(self.cz_layers) + 1)

    def _ry(self, psi, q, theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        v = psi.reshape(2**q, 2, -1)
        out = np.empty_like(v)
        out[:, 0] = c * v[:, 0] - s * v[:, 1]
        out[:, 1] = s * v[:, 0] + c * v[:, 1]
        return out.reshape(psi.shape)

    def apply(self, params, psi):
        th = params.reshape(-1, self.w)
        for l in range(len(th)):
            for q in range(self.w):
                psi = self._ry(psi, q, th[l, q])
            if l < len(self.signs):
                psi = psi * self.signs[l]
        return psi

    def loss_and_grad(self, params, ins, outs):
        """1 - (mean_b <out_b|U|in_b>)^2: zero iff U maps every input exactly (common sign)."""
        B = ins.shape[1]
        th = params.reshape(-1, self.w)
        psi = ins.copy()
        before = []
        for l in range(len(th)):
            for q in range(self.w):
                before.append(psi)
                psi = self._ry(psi, q, th[l, q])
            if l < len(self.signs):
                psi = psi * self.signs[l]
        a = float(np.sum(outs * psi)) / B
        grad = np.zeros_like(th)
        phi = outs.copy()
        gi = len(before)
        for l in range(len(th) - 1, -1, -1):
            if l < len(self.signs):
                phi = phi * self.signs[l]
            for q in range(self.w - 1, -1, -1):
                gi -= 1
                grad[l, q] = 0.5 * float(np.sum(phi * self._ry(before[gi], q, th[l, q] + math.pi))) / B
                phi = self._ry(phi, q, -th[l, q])
        return 1 - a * a, (-2 * a * grad).reshape(-1)


def fit(circ, ins, outs, x0, maxiter=6000):
    res = minimize(circ.loss_and_grad, x0, args=(ins, outs), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-18, gtol=1e-14, maxcor=30))
    return res.fun, res.x


def solve(w, layers, ins, outs, rng, restarts, x0=None):
    circ = BatchCircuit(w, layers)
    best = (math.inf, None)
    starts = ([x0] if x0 is not None else []) + [rng.uniform(-math.pi, math.pi, circ.num_params) for _ in range(restarts)]
    for x in starts:
        f, xs = fit(circ, ins, outs, x)
        if f < best[0]:
            best = (f, xs)
        if f < TOL:
            break
    return best


def structure(m, k, all_ones=False, seed=0, max_layers=30, restarts=8, log=print):
    """Minimal-ish CZ pattern for SCS_{m,k}: shallowest exact brickwork, then pruning."""
    rng = np.random.default_rng(seed)
    w = k + 1
    ins, outs = scs_io(m, k, all_ones)
    edges = [(i, i + 1) for i in range(w - 1)]
    found = None
    for L in range(1, max_layers + 1):
        layers = brickwork(edges, L)
        f, x = solve(w, layers, ins, outs, rng, restarts)
        log(f"  SCS({m},{k}) L={L} cz={sum(map(len, layers))} infid={f:.1e}")
        if f < TOL:
            found = (layers, x)
            break
    if found is None:
        return None
    layers, x = found
    improved = True
    while improved:
        improved = False
        order = [(li, e) for li, l in enumerate(layers) for e in l]
        rng.shuffle(order)
        for li, e in order:
            trial = drop_cz(layers, li, e)
            f, xt = solve(w, trial, ins, outs, rng, restarts=3, x0=x)
            if f < TOL:
                layers, x = compact(trial, xt, w)
                log(f"  SCS({m},{k}) pruned -> cz={sum(map(len, layers))}")
                improved = True
                break
    return layers, x


def build_table(kmin, kmax, mmax, seed=0, log=print):
    """For each k: prune a pattern at a representative m, then fit angles for all m."""
    table = {}
    rng = np.random.default_rng(seed + 1)
    for k in range(kmin, kmax + 1):
        w = k + 1
        # keys: "m,k" (inputs l <= k) for m >= k+1, and "m,k,1" (adds |1^w>) for m = k+1
        for all_ones, ms in ((True, [w]), (False, list(range(w, mmax + 1)))):
            t0 = time.time()
            suffix = ",1" if all_ones else ""
            rep = ms[len(ms) // 2]
            st = structure(rep, k, all_ones, seed=seed, log=log)
            if st is None:
                log(f"k={k} all_ones={all_ones}: no exact pattern found")
                continue
            layers, x_rep = st
            cz = sum(map(len, layers))
            # fit every m, walking outward from the representative to warm-start
            order = sorted(ms, key=lambda m: abs(m - rep))
            fitted = {rep: x_rep}
            for m in order:
                if m in fitted:
                    continue
                near = min(fitted, key=lambda mm: abs(mm - m))
                ins, outs = scs_io(m, k, all_ones)
                f, x = solve(w, layers, ins, outs, rng, restarts=10, x0=fitted[near])
                if f >= TOL:  # pattern too tight for this m: give it its own structure
                    st_m = structure(m, k, all_ones, seed=seed + m, log=log)
                    if st_m is None:
                        log(f"  SCS({m},{k}) FAILED")
                        continue
                    table[f"{m},{k}{suffix}"] = dict(cz_layers=st_m[0], params=[float(v) for v in st_m[1]])
                    continue
                fitted[m] = x
            for m, x in fitted.items():
                table.setdefault(f"{m},{k}{suffix}", dict(cz_layers=layers, params=[float(v) for v in x]))
            log(f"k={k} all_ones={all_ones}: {cz} CZ per block, m in [{min(ms)}, {max(ms)}] ({time.time() - t0:.0f}s)")
    return table


BEGIN = "# BEGIN SCS TABLE (generated by experiments/scs_synth.py --export; do not edit)\n"
END = "# END SCS TABLE\n"


def export_table(main_path=os.path.join(os.path.dirname(HERE), "main.py")):
    """Merge every scs_k*.json, re-verify each block, and write _SCS into main.py."""
    blocks = {}
    for name in sorted(os.listdir(OUT)):
        if name.startswith("scs_k") and name.endswith(".json"):
            with open(os.path.join(OUT, name)) as fh:
                blocks.update(json.load(fh))
    lines = [BEGIN, "_SCS = {  # (m, k, all_ones): (cz_layers, ry_params)\n"]
    for key in sorted(blocks, key=lambda s: tuple(map(int, s.split(",")))):
        parts = list(map(int, key.split(",")))
        m, k, all_ones = parts[0], parts[1], len(parts) == 3
        e = blocks[key]
        ins, outs = scs_io(m, k, all_ones)
        got = BatchCircuit(k + 1, e["cz_layers"]).apply(np.array(e["params"]), ins)
        # one common sign for every input is a global phase of the whole state: allowed
        err = min(np.abs(got - outs).max(), np.abs(got + outs).max())
        if err > 1e-6:
            print(f"skip SCS{key}: max amplitude error {err:.1e}")
            continue
        layers = [[tuple(x) for x in l] for l in e["cz_layers"]]
        lines.append(f"    ({m}, {k}, {all_ones}): ({layers!r}, {[float(p) for p in e['params']]!r}),\n")
    lines += ["}\n", END]
    with open(main_path) as fh:
        src = fh.read()
    head, rest = src.split(BEGIN)
    _, tail = rest.split(END)
    with open(main_path, "w") as fh:
        fh.write(head + "".join(lines) + tail)
    print(f"exported {len(lines) - 4} blocks")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="write all block tables into main.py and exit")
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--mmax", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.export:
        export_table()
        return
    log = print if args.verbose else (lambda s: print(s) if not s.startswith("  ") else None)
    table = build_table(args.kmin, args.kmax, args.mmax, args.seed, log=log)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"scs_k{args.kmin}-{args.kmax}.json"), "w") as fh:
        json.dump(table, fh)
    print(f"wrote {len(table)} blocks")


if __name__ == "__main__":
    main_cli()
