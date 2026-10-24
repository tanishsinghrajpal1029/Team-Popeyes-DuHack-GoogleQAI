"""Broad robustness sweep: any crash or invalid circuit scores zero, so look for them.

  A  exactness + all rules, every N <= 12 and every 0 <= n <= N, 6 calibrations
     (qubit dropout 1%, 5%, 15%)
  B  rules, build time and moment budget (no fidelity: too many qubits) for N = 13..30
  C  large W states, N up to 105

Writes experiments/results/harden.csv and prints every failure plus the worst build time
and moment count.

  python experiments/harden.py --workers 6
  python experiments/harden.py --workers 6 --sub tanish_submission.py --out harden_tanish.csv
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import cirq  # noqa: E402

import main  # noqa: E402
from calib import make_calibration  # noqa: E402
from evaluate import append_rows  # noqa: E402
from validity import MAX_MOMENTS, check_rules  # noqa: E402

CALS = [(0, 0.01), (1, 0.01), (2, 0.05), (3, 0.05), (4, 0.15), (5, 0.15)]


def _job(args):
    part, N, n, seed, dropout = args
    cal = make_calibration(seed, qubit_dropout_rate=dropout)
    row = dict(part=part, N=N, n=n, seed=seed, dropout=dropout)
    t0 = time.time()
    try:
        circuit, qubits = main.create_dicke_state(N, n, **cal)
    except NotImplementedError as e:
        row.update(status="not_implemented", detail=str(e)[:200], build_s=round(time.time() - t0, 2))
        return row
    except Exception as e:  # any other exception is a bug
        row.update(status="CRASH", detail=f"{type(e).__name__}: {e}"[:300], build_s=round(time.time() - t0, 2))
        return row
    row["build_s"] = round(time.time() - t0, 2)
    row["moments"] = len(circuit)
    row["cz"] = sum(op.gate == cirq.CZ for op in circuit.all_operations())
    if part == "A":
        problems = check_rules(circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"])
    else:  # same rules without the exponential-cost fidelity check
        problems = check_rules(circuit, qubits, N, n, cal["cz_pauli_error_rate"],
                               cal["sq_pauli_error_rate"], fidelity_atol=None)
    if row["build_s"] >= 60:
        problems.append(f"build {row['build_s']} s")
    row["status"] = "ok" if not problems else ("over_moments" if all("moments" in p for p in problems) else "INVALID")
    row["detail"] = "; ".join(problems)[:300]
    return row


def jobs():
    for N in range(1, 13):
        for n in range(N + 1):
            for seed, d in CALS:
                yield ("A", N, n, seed, d)
    for N in range(13, 31):
        for n in sorted({1, 2, 3, 4, 5, N // 4, N // 2, N - 1}):
            for seed, d in CALS[::2]:
                yield ("B", N, n, seed, d)
    for N in (40, 60, 80, 100, 105):
        for n in (1, N - 1):
            for seed, d in CALS[:2] + CALS[4:5]:
                yield ("C", N, n, seed, d)


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--sub", default=None, help="submission file to test instead of main.py")
    ap.add_argument("--out", default="harden.csv", help="CSV file name under experiments/results/")
    args = ap.parse_args()
    if args.sub:  # forked workers inherit the swapped module
        global main
        from compare import load
        main = load("sub", args.sub)
    todo = [j for j in jobs() if j[0] in args.parts]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_job, todo, chunksize=2):
            rows.append(r)
            if r["status"] not in ("ok",):
                print(f"[{r['part']}] N={r['N']} n={r['n']} seed={r['seed']} dropout={r['dropout']}: "
                      f"{r['status']} {r.get('detail', '')}", flush=True)
    append_rows(os.path.join(HERE, "results", args.out), rows)
    by = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    built = [r for r in rows if "moments" in r]
    print(f"\n{len(rows)} builds: {by}")
    print(f"max build time {max(r['build_s'] for r in rows):.1f} s; "
          f"max moments {max(r['moments'] for r in built)} (limit {MAX_MOMENTS})")


if __name__ == "__main__":
    main_cli()
