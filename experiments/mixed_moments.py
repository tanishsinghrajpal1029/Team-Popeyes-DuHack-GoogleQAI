"""Is a submission's simulated gain from moments that mix CZ and single-qubit gates?

The noise model times a moment by its slowest gate, so a 25 ns PhasedXZ inside a 42 ns
CZ moment is free. This scores each submission's circuits as built and after
cirq.stratified_circuit separates CZ from single-qubit gates (exact same gates and
qubits). Rows go to experiments/results/<out> with labels <sub> and <sub>+stratified.

  python experiments/mixed_moments.py --sub tanish=tanish_submission.py --cases 10,5 9,4 --out mixed.csv
"""

import argparse
import os
import sys
from multiprocessing import Pool

import cirq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from calib import BENCH_SEEDS  # noqa: E402
from compare import load  # noqa: E402
from evaluate import append_rows, evaluate_case  # noqa: E402


def mixed_count(circuit):
    return sum(any(op.gate == cirq.CZ for op in m) and any(len(op.qubits) == 1 for op in m) for m in circuit)


def _job(args):
    label, path, N, n, seed, stratify = args
    build = load(label, path).create_dicke_state

    def builder(*a, **kw):
        c, q = build(*a, **kw)
        if stratify:
            c = cirq.stratified_circuit(c, categories=[lambda op: op.gate == cirq.CZ])
        return c, q

    c, _ = build(N, n, **__import__("calib").make_calibration(seed))
    r = evaluate_case(N, n, seed, "idle", 2000, builder=builder,
                      label=label + ("+stratified" if stratify else ""))
    r["mixed_moments_as_built"] = mixed_count(c)
    return r


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--cases", nargs="+", required=True, help="N,n pairs")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BENCH_SEEDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(l, p, int(c.split(",")[0]), int(c.split(",")[1]), s, st)
            for l, p in (x.split("=", 1) for x in args.sub) for c in args.cases
            for s in args.seeds for st in (False, True)]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r)
            print(f"{r['label']:>20} N={r['N']:2d} n={r['n']:2d} seed={r['seed']} mixed={r['mixed_moments_as_built']:3d} "
                  f"cz={r['cz']:3d} moments={r['moments']:3d} dur={r['duration_ns']:6.0f} f={r['f']:.3f} "
                  f"x={r['x_inf_Mall']:.3f} median={r['score_median_Mall']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["N"], r["n"], r["seed"], r["label"]))
    append_rows(os.path.join(HERE, "results", args.out), rows)


if __name__ == "__main__":
    main_cli()
