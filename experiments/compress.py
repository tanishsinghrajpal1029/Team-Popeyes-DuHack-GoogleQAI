"""Warm-start compression of Bryan's exact Dicke circuits. NEGATIVE RESULT: greedy CZ
deletion from D9,3 never recovered exactness (0 of 8 tries); see ARCHITECTURE.md, section 6.

synth.py finds circuits from random starts, which stops scaling around N = 9. Here we start
from a circuit that is already exact (the vendored split Baertschi-Eidenbenz construction,
compiled without DD, placed on the uniform calibration so the shape is compact) and try to
delete CZs one at a time, re-optimizing every single-qubit gate after each deletion.

Ansatz: the reference circuit's CZ layers, with a general single-qubit gate
U = Rz(a) Ry(b) Rz(c) on every qubit between consecutive CZ layers (Bryan's gates include
S and Rx, so the Ry-only ansatz of synth.py does not cover them). The Rz parts and the CZ
signs of one layer are all diagonal, so they are applied as one phase vector, and their
gradients come from one matrix-vector product per layer. The loss is 1 - |<D|psi>|^2 with
an exact backpropagated gradient; a deletion is kept if L-BFGS gets back below TOL.

Results: experiments/results/compress/D{N}_{k}.json (rewritten after every accepted
deletion, so a stopped run keeps its best exact circuit).

  python experiments/compress.py --cases 10,5 9,4 --workers 2
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import cirq
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from synth import dicke_vector  # noqa: E402

OUT = os.path.join(HERE, "results", "compress")
TOL = 1e-13


# ---- reference circuit -------------------------------------------------------------------------


def zyz(u: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) with u = e^{i phi} Rz(a) Ry(b) Rz(c)."""
    b = 2 * math.atan2(abs(u[1, 0]), abs(u[0, 0]))
    s_plus = np.angle(u[1, 1]) - np.angle(u[0, 0]) if abs(u[0, 0]) > 1e-12 else 0.0  # a + c
    s_minus = np.angle(u[1, 0]) - np.angle(-u[0, 1]) if abs(u[1, 0]) > 1e-12 else 0.0  # a - c
    # a +- c are only known mod 2 pi, so a and c are ambiguous by pi: keep the branch that
    # reconstructs u up to a global phase.
    best = None
    for a, c in (((s_plus + s_minus) / 2, (s_plus - s_minus) / 2),
                 ((s_plus + s_minus) / 2 + math.pi, (s_plus - s_minus) / 2 - math.pi),
                 ((s_plus + s_minus) / 2 + math.pi, (s_plus - s_minus) / 2 + math.pi)):
        for bb in (b, -b):
            err = 1 - abs(np.vdot(u3_matrix(a, bb, c), u)) / 2
            if best is None or err < best[0]:
                best = (err, (a, bb, c))
    return best[1]


def u3_matrix(a, b, c) -> np.ndarray:
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])  # noqa: E731
    return rz(a) @ cirq.unitary(cirq.ry(b)) @ rz(c)


def reference(N: int, k: int) -> dict:
    """Bryan's best exact no-DD candidate on the uniform calibration, as an abstract shape."""
    import main

    cal_d = main._uniform_calibration()
    cal = main.Calibration(*(cal_d[x] for x in ("cz_pauli_error_rate", "sq_pauli_error_rate",
                                                "zero_errors", "one_errors")))
    best = None
    for circ, qs, kind in main.generate_candidates(N, k, cal, dd=False):
        s = main.estimate_log_fidelity(circ, cal, qs, k / N)
        if best is None or s > best[0]:
            best = (s, circ, list(qs), kind)
    _, circ, qs, kind = best
    assert circ.all_qubits() <= set(qs)
    idx = {q: i for i, q in enumerate(qs)}
    cz_layers, slots = [], [[np.eye(2, dtype=complex) for _ in range(N)]]
    for moment in circ:
        czs = [op for op in moment if op.gate == cirq.CZ]
        if czs:
            assert len(czs) == len(moment), "mixed moment"
            cz_layers.append([tuple(sorted((idx[op.qubits[0]], idx[op.qubits[1]]))) for op in czs])
            slots.append([np.eye(2, dtype=complex) for _ in range(N)])
        else:
            for op in moment:
                i = idx[op.qubits[0]]
                slots[-1][i] = cirq.unitary(op) @ slots[-1][i]
    params = np.array([[zyz(u) for u in slot] for slot in slots])  # (L + 1, N, 3) as (a, b, c)
    r0, c0 = qs[0].row, qs[0].col
    return dict(N=N, k=k, kind=kind, coords=[(q.row - r0, q.col - c0) for q in qs],
                cz_layers=cz_layers, params=params.transpose(0, 2, 1).reshape(-1).tolist())


# ---- complex statevector ansatz with exact gradient ----------------------------------------


class U3Circuit:
    """Slots 0..L of U3 layers around CZ layers 0..L-1. params: (L + 1, 3, N) = (a, b, c)."""

    def __init__(self, N: int, cz_layers):
        self.N = N
        self.L = len(cz_layers)
        bits = (np.arange(2**N)[:, None] >> (N - 1 - np.arange(N))) & 1
        self.z = (1 - 2 * bits).astype(float)  # (2^N, N): eigenvalue of Z_q on each basis state
        self.signs = [
            np.prod([1 - 2 * (bits[:, i] & bits[:, j]) for i, j in layer], axis=0).astype(float)
            if layer else np.ones(2**N)
            for layer in cz_layers
        ]
        self.num_params = 3 * N * (self.L + 1)

    @staticmethod
    def _ry(psi, q, theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        v = psi.reshape(2**q, 2, -1)
        out = np.empty_like(v)
        out[:, 0] = c * v[:, 0] - s * v[:, 1]
        out[:, 1] = s * v[:, 0] + c * v[:, 1]
        return out.reshape(-1)

    def _diag(self, th, l):
        """Phase vector of block l: Rz(a_{l-1}), CZ_{l-1}, Rz(c_l) (whichever exist)."""
        ang = np.zeros(2**self.N)
        if l > 0:
            ang += self.z @ th[l - 1, 0]
        if l <= self.L:
            ang += self.z @ th[l, 2]
        d = np.exp(-0.5j * ang)
        if 0 < l <= self.L:
            d = d * self.signs[l - 1]
        return d

    def state(self, params):
        th = params.reshape(self.L + 1, 3, self.N)
        psi = np.zeros(2**self.N, dtype=complex)
        psi[0] = 1.0
        for l in range(self.L + 1):
            psi = psi * self._diag(th, l)
            for q in range(self.N):
                psi = self._ry(psi, q, th[l, 1, q])
        return psi * self._diag(th, self.L + 1)

    def loss_and_grad(self, params, target):
        th = params.reshape(self.L + 1, 3, self.N)
        psi = np.zeros(2**self.N, dtype=complex)
        psi[0] = 1.0
        diags, after_diag, before_ry = [], [], []
        for l in range(self.L + 1):
            d = self._diag(th, l)
            psi = psi * d
            diags.append(d)
            after_diag.append(psi)
            for q in range(self.N):
                before_ry.append(psi)
                psi = self._ry(psi, q, th[l, 1, q])
        d = self._diag(th, self.L + 1)
        psi = psi * d
        diags.append(d)
        after_diag.append(psi)
        a = complex(target @ psi)
        grad = np.zeros_like(th)
        phi = target.astype(complex)  # backward vector (target^dagger propagated back)
        gi = len(before_ry)
        for l in range(self.L + 1, -1, -1):
            # diagonal block l: d<phi|D|psi_in>/d theta_q = -i/2 sum_x conj(phi) (D psi_in) z_q
            gz = (-0.5j * (np.conj(phi) * after_diag[l])) @ self.z
            if l > 0:
                grad[l - 1, 0] += 2 * np.real(np.conj(a) * gz) * -1  # a-angle of slot l-1
            if l <= self.L:
                grad[l, 2] += 2 * np.real(np.conj(a) * gz) * -1  # c-angle of slot l
            phi = phi * np.conj(diags[l])
            if l == 0:
                break
            ls = l - 1  # Ry layer of slot l - 1, walked backwards
            for q in range(self.N - 1, -1, -1):
                gi -= 1
                ga = 0.5 * np.vdot(phi, self._ry(before_ry[gi], q, th[ls, 1, q] + math.pi))
                grad[ls, 1, q] = -2 * np.real(np.conj(a) * ga)
                phi = self._ry(phi, q, -th[ls, 1, q])
        return 1 - abs(a) ** 2, grad.reshape(-1)


def optimize(circ: U3Circuit, target, x0, maxiter=3000):
    res = minimize(circ.loss_and_grad, x0, args=(target,), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-20, gtol=1e-15, maxcor=50))
    return float(res.fun), res.x


def _layer_params(params, N, L):
    return np.asarray(params).reshape(L + 1, 3, N)


def delete_cz(cz_layers, params, N, li, e):
    """Drop CZ e from layer li; if the layer empties, merge the U3 slots on both sides."""
    layers = [list(l) for l in cz_layers]
    layers[li].remove(e)
    th = _layer_params(params, N, len(cz_layers))
    if layers[li]:
        return layers, th.reshape(-1)
    merged = np.array([zyz(u3_matrix(*th[li + 1, :, q]) @ u3_matrix(*th[li, :, q])) for q in range(N)])
    th = np.concatenate([th[:li], merged.T[None], th[li + 2:]])
    del layers[li]
    return layers, th.reshape(-1)


# ---- greedy compression ------------------------------------------------------------------------


def compress(N: int, k: int, seed: int = 0, log=print, save=None, budget_s: float = math.inf):
    ref = reference(N, k)
    target = dicke_vector(N, k).astype(complex)
    layers, x = ref["cz_layers"], np.array(ref["params"])
    f0 = U3Circuit(N, layers).loss_and_grad(x, target)[0]
    log(f"[{N},{k}] reference {ref['kind']}: cz={sum(map(len, layers))} layers={len(layers)} infid={f0:.1e}")
    if f0 > 1e-10:
        raise RuntimeError(f"reference circuit not exact ({f0:.1e})")
    f, x = optimize(U3Circuit(N, layers), target, x)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    def result():
        return dict(N=N, k=k, kind=ref["kind"], coords=ref["coords"], cz_layers=layers,
                    params=[float(v) for v in x], infidelity=float(f), cz=sum(map(len, layers)),
                    cz_layers_count=len(layers), reference_cz=sum(map(len, ref["cz_layers"])),
                    reference_layers=len(ref["cz_layers"]), seed=seed, seconds=round(time.time() - t0, 1))

    if save:
        save(result())
    improved = True
    while improved and time.time() - t0 < budget_s:
        improved = False
        # Fewest-CZ layers first: emptying a layer also shortens the circuit.
        order = [(li, e) for li, l in enumerate(layers) for e in l]
        rng.shuffle(order)
        order.sort(key=lambda t: len(layers[t[0]]))
        for li, e in order:
            if time.time() - t0 > budget_s:
                break
            trial, x_trial = delete_cz(layers, x, N, li, e)
            ft, xt = optimize(U3Circuit(N, trial), target, x_trial)
            if ft >= TOL:  # one perturbed retry
                ft2, xt2 = optimize(U3Circuit(N, trial), target, x_trial + rng.normal(0, 0.05, x_trial.size))
                if ft2 < ft:
                    ft, xt = ft2, xt2
            if ft < TOL:
                layers, x, f = trial, xt, ft
                log(f"[{N},{k}] deleted CZ {e} in layer {li} -> cz={sum(map(len, layers))} "
                    f"layers={len(layers)} infid={f:.1e} ({time.time() - t0:.0f}s)")
                if save:
                    save(result())
                improved = True
                break
            log(f"[{N},{k}]   keep CZ {e} in layer {li} (best infid {ft:.1e})")
    return result()


def _job(args):
    N, k, seed, budget = args
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"D{N}_{k}_s{seed}")

    def log(s):
        with open(stem + ".log", "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {s}\n")

    def save(r):
        with open(stem + ".json.tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(stem + ".json.tmp", stem + ".json")

    r = compress(N, k, seed, log=log, save=save, budget_s=budget)
    save(r)
    return f"D^{N}_{k} seed {seed}: {r['reference_cz']} -> {r['cz']} CZ, " \
           f"{r['reference_layers']} -> {r['cz_layers_count']} layers ({r['seconds']}s)"


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", required=True, help="N,k pairs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--budget", type=float, default=6 * 3600, help="seconds per case")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    jobs = [(int(c.split(",")[0]), int(c.split(",")[1]), s, args.budget) for c in args.cases for s in args.seeds]
    with Pool(args.workers) as pool:
        for line in pool.imap_unordered(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main_cli()
