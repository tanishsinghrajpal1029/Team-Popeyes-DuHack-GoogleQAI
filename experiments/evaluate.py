"""Simulate circuits from main.py under calibrated noise and score them like the judges.

For each (N, n, calibration seed) this script
  1. builds the circuit (timed),
  2. checks the challenge rules,
  3. computes the exact noisy Z- and X-basis output distributions with a density-matrix
     simulation (X basis = an H-equivalent PhasedXZ on every data qubit, noise included),
     then applies the per-qubit readout confusion,
  4. reports shot-noise-free diagnostics (f, delta_rms, lambda, x_inf) and the judges'
     score statistics over many simulated 10k-shot runs,
and appends one CSV row per case to experiments/results/<out>.

Noise presets:
  gate : calibrated depolarizing gate errors + readout errors
  idle : gate + T1 = 68 us (spec sheet) + T_phi = 100 us (assumed)
  full : idle + quasi-static detuning with T2* = 20 us (assumed), 16 samples

Example:
  python experiments/evaluate.py --out baseline.csv --nmax 8 --noise idle
"""

import argparse
import csv
import functools
import json
import math
import os
import subprocess
import sys
import time

import cirq
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import main  # noqa: E402
from calib import BENCH_SEEDS, make_calibration  # noqa: E402
from noise import CalibratedNoise, apply_readout, detuned_copies, moment_duration_ns  # noqa: E402
from scoring import ceiling, distribution_metrics, score_distribution  # noqa: E402
from validity import check_rules  # noqa: E402

NOISE_PRESETS = {
    "gate": dict(t1_us=None, tphi_us=None, t2star_us=None),
    "idle": dict(t1_us=68.0, tphi_us=100.0, t2star_us=None),
    "full": dict(t1_us=68.0, tphi_us=100.0, t2star_us=20.0),
}
H_PHXZ = cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))


def exact_distributions(
    circuit: cirq.Circuit, qubits: set, cal: dict, t1_us=None, tphi_us=None,
    t2star_us=None, detuning_samples: int = 16, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact noisy Z- and X-basis distributions over the data qubits (cirq bit order)."""
    data = sorted(qubits)
    anc = sorted(circuit.all_qubits() - set(qubits))
    order = data + anc
    noise = CalibratedNoise(cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"], t1_us, tphi_us)
    x_layer = cirq.Circuit(cirq.Moment(H_PHXZ.on(q) for q in data)).with_noise(noise)
    copies = [circuit] if t2star_us is None else detuned_copies(circuit, t2star_us, detuning_samples, seed)
    sim = cirq.DensityMatrixSimulator(dtype=np.complex128)
    pz = np.zeros(2 ** len(order))
    px = np.zeros(2 ** len(order))
    for c in copies:
        rho = sim.simulate(c.with_noise(noise), qubit_order=order).final_density_matrix
        rho_x = sim.simulate(x_layer, initial_state=rho, qubit_order=order).final_density_matrix
        pz += np.real(np.diag(rho))
        px += np.real(np.diag(rho_x))
    pz = pz.reshape(2 ** len(data), -1).sum(axis=1) / len(copies)
    px = px.reshape(2 ** len(data), -1).sum(axis=1) / len(copies)
    e0 = [cal["zero_errors"][q][0] for q in data]
    e1 = [cal["one_errors"][q][0] for q in data]
    return apply_readout(pz, e0, e1), apply_readout(px, e0, e1)


def circuit_stats(circuit: cirq.Circuit) -> dict:
    ops = list(circuit.all_operations())
    return dict(
        cz=sum(op.gate == cirq.CZ for op in ops),
        sq=sum(isinstance(op.gate, cirq.PhasedXZGate) and moment_duration_ns(cirq.Moment(op)) > 0 for op in ops),
        moments=len(circuit),
        duration_ns=sum(moment_duration_ns(m) for m in circuit),
        qubits_touched=len(circuit.all_qubits()),
    )


def evaluate_case(N: int, n: int, seed: int, noise: str, repeats: int, builder=None, label="main") -> dict:
    cal = make_calibration(seed)
    builder = builder or main.create_dicke_state
    t0 = time.time()
    circuit, qubits = builder(N, n, **cal)
    build_s = time.time() - t0
    problems = check_rules(circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"])
    row = dict(label=label, N=N, n=n, K=math.comb(N, n), seed=seed, noise=noise,
               build_s=round(build_s, 3), valid=not problems, **circuit_stats(circuit))
    if problems:
        row["problems"] = "; ".join(problems)[:300]
        return row
    pz, px = exact_distributions(circuit, qubits, cal, **NOISE_PRESETS[noise], seed=seed)
    for tag, M in (("Mall", N), ("M4", min(4, N))):
        met = distribution_metrics(pz, px, N, n, M)
        if tag == "Mall":
            row.update(f=met["f"], delta_rms=met["delta_rms"], delta_max=met["delta_max"],
                       lam=met["lam"], corr_ratio=json.dumps(met["corr_ratio"]))
        row[f"x_inf_{tag}"] = met["x_inf"]
        stats = score_distribution(pz, px, N, n, M, repeats=repeats, seed=seed)
        row.update({f"{k}_{tag}": v for k, v in stats.items()})
        ceil = ceiling(N, n, M)
        row[f"ceiling_median_{tag}"] = ceil["score_median"]
        row[f"frac_ceiling_{tag}"] = stats["score_median"] / ceil["score_median"]
    return row


def git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "main.py"], cwd=HERE) != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def append_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    exists = os.path.exists(path)
    if exists:
        with open(path) as fh:
            old_keys = next(csv.reader(fh), [])
        keys = old_keys + [k for k in keys if k not in old_keys]
        with open(path) as fh:
            old = list(csv.DictReader(fh))
        rows = old + rows
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV file name under experiments/results/")
    ap.add_argument("--label", default=None, help="row label (default: git revision of main.py)")
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--only-w", action="store_true", help="only n = 1")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BENCH_SEEDS))
    ap.add_argument("--noise", choices=NOISE_PRESETS, default="idle")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="keyword option for main.build, e.g. --opt w_tree=False")
    args = ap.parse_args()
    opts = {k: json.loads(v.lower()) if v.lower() in ("true", "false") else json.loads(v)
            for k, v in (o.split("=", 1) for o in args.opt)}
    builder = functools.partial(main.build, **opts)
    label = (args.label or git_rev()) + "".join(f" {k}={v}" for k, v in opts.items())
    cases = [(N, n) for N in range(args.nmin, args.nmax + 1)
             for n in ([1] if args.only_w else range(1, N // 2 + 1))]
    rows = []
    for N, n in cases:
        for seed in args.seeds:
            r = evaluate_case(N, n, seed, args.noise, args.repeats, builder=builder, label=label)
            rows.append(r)
            nan = float("nan")
            print(f"N={N:2d} n={n:2d} seed={seed} cz={r['cz']:4d} moments={r['moments']:4d} "
                  f"f={r.get('f', nan):.3f} chi2/(K-1)={r.get('chi2_ratio_Mall', nan):.3f} "
                  f"x={r.get('x_inf_Mall', nan):.4f} loss={r.get('loss_mean_Mall', nan):.3f} "
                  f"median={r.get('score_median_Mall', nan):6.3f} "
                  f"({100 * r.get('frac_ceiling_Mall', nan):.0f}% of ceiling)", flush=True)
    append_rows(os.path.join(HERE, "results", args.out), rows)


if __name__ == "__main__":
    main_cli()
