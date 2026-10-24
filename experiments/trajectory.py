"""Quantum-trajectory version of evaluate.py, for N where density matrices are too big.

Each trajectory is one `cirq.Simulator` run of the noisy circuit (every channel samples
one Kraus operator), and it yields the full probability vector |psi|^2, not one shot.
The average over T trajectories is an unbiased estimate of the exact noisy distribution
that evaluate.py computes, so the same readout confusion, diagnostics and judges'-score
statistics apply unchanged and N >= 13 rows are directly comparable to N <= 12 rows.

The X basis branches off every trajectory at the end: the noisy H-equivalent PhasedXZ
layer is applied to a copy of the final state.

Estimation error: a finite T adds sampling noise to every string's probability, which
the chi^2 term reads as population imbalance (a spurious noncentrality of about
10^4 * f * (fraction of trajectories with an error) / T). `debias` removes it: within
each Hamming-weight class, deviations from the class mean are shrunk so that their mean
square drops by the sampling variance estimated from the per-trajectory spread. f and
the weight distribution (hence the X term) are untouched.

Example:
  python experiments/trajectory.py --out bign/traj_N15_n7.csv --N 15 --ns 7 --traj 1000
"""

import argparse
import functools
import json
import os
import sys
import time

import cirq
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import main  # noqa: E402
from calib import BENCH_SEEDS  # noqa: E402
from evaluate import H_PHXZ, NOISE_PRESETS, append_rows, evaluate_case, git_rev  # noqa: E402
from noise import CalibratedNoise, apply_readout  # noqa: E402
from scoring import popcounts  # noqa: E402


def trajectory_distributions(
    circuit: cirq.Circuit, qubits: set, cal: dict, t1_us=None, tphi_us=None,
    t2star_us=None, seed: int = 0, num_traj: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory estimate of the noisy Z- and X-basis distributions (cirq bit order)."""
    if t2star_us is not None:
        raise NotImplementedError("quasi-static detuning is only in evaluate.py")
    data = sorted(qubits)
    anc = sorted(circuit.all_qubits() - set(qubits))
    order = data + anc
    noise = CalibratedNoise(cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"], t1_us, tphi_us)
    noisy = circuit.with_noise(noise)
    x_layer = cirq.Circuit(cirq.Moment(H_PHXZ.on(q) for q in data)).with_noise(noise)
    sim = cirq.Simulator(dtype=np.complex128, seed=np.random.default_rng(seed))
    dim = 2 ** len(data)
    pz, pz2, px = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    for _ in range(num_traj):
        psi = sim.simulate(noisy, qubit_order=order).final_state_vector
        psi_x = sim.simulate(x_layer, initial_state=psi, qubit_order=order).final_state_vector
        p = (np.abs(psi) ** 2 / np.vdot(psi, psi).real).reshape(dim, -1).sum(axis=1)
        pz += p
        pz2 += p * p
        px += (np.abs(psi_x) ** 2 / np.vdot(psi_x, psi_x).real).reshape(dim, -1).sum(axis=1)
    pz, px = pz / num_traj, px / num_traj
    var_of_mean = np.maximum(pz2 / num_traj - pz**2, 0) / max(num_traj - 1, 1)
    pz = debias(pz, var_of_mean, len(data))
    e0 = [cal["zero_errors"][q][0] for q in data]
    e1 = [cal["one_errors"][q][0] for q in data]
    return apply_readout(pz, e0, e1), apply_readout(px, e0, e1)


def debias(p: np.ndarray, var_of_mean: np.ndarray, num_qubits: int) -> np.ndarray:
    """Shrink within-weight-class deviations by their estimated sampling variance."""
    out = p.copy()
    pc = popcounts(num_qubits)
    for w in range(num_qubits + 1):
        idx = pc == w
        dev = p[idx] - p[idx].mean()
        obs = np.mean(dev**2)
        if obs > 0:
            out[idx] = p[idx].mean() + dev * np.sqrt(max(0.0, 1 - np.mean(var_of_mean[idx]) / obs))
    return out


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV file name under experiments/results/")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BENCH_SEEDS))
    ap.add_argument("--noise", choices=["gate", "idle"], default="idle")
    ap.add_argument("--traj", type=int, default=1000, help="trajectories per case")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--label", default=None)
    ap.add_argument("--sub", default=None, metavar="LABEL=PATH",
                    help="score another submission file instead of main.build (as in compare.py)")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="keyword option for main.build, e.g. --opt bryan=false")
    args = ap.parse_args()
    opts = {k: json.loads(v.lower()) if v.lower() in ("true", "false") else json.loads(v)
            for k, v in (o.split("=", 1) for o in args.opt)}
    if args.sub:
        from compare import load

        sub_label, path = args.sub.split("=", 1)
        builder, label = load(sub_label, path).create_dicke_state, args.label or sub_label
    else:
        builder = functools.partial(main.build, **opts)
        label = args.label or git_rev()
    label += f" traj={args.traj}" + "".join(f" {k}={v}" for k, v in opts.items())
    dist = functools.partial(trajectory_distributions, num_traj=args.traj)
    for n in args.ns:
        for seed in args.seeds:
            t0 = time.time()
            r = evaluate_case(args.N, n, seed, args.noise, args.repeats, builder=builder,
                              label=label, distributions=dist)
            r["sim_s"] = round(time.time() - t0, 1)
            nan = float("nan")
            print(f"N={args.N:2d} n={n:2d} seed={seed} cz={r['cz']:4d} moments={r['moments']:4d} "
                  f"f={r.get('f', nan):.3f} chi2/(K-1)={r.get('chi2_ratio_Mall', nan):.3f} "
                  f"x={r.get('x_inf_Mall', nan):.4f} median={r.get('score_median_Mall', nan):6.3f} "
                  f"({100 * r.get('frac_ceiling_Mall', nan):.0f}% of ceiling) [{r['sim_s']}s]", flush=True)
            append_rows(os.path.join(HERE, "results", args.out), [r])  # per row: keep partial runs


if __name__ == "__main__":
    main_cli()
