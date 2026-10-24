"""Head-to-head of submission files under the same harness as evaluate.py.

Each submission is a .py file exposing create_dicke_state. Every (N, n, calibration seed)
is scored with evaluate.evaluate_case (exact density matrix, idle noise by default), in
parallel, and one row per case goes to experiments/results/<out>.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/compare.py \
      --sub richard=main.py --sub tanish=tanish_submission.py --out vs_tanish.csv --workers 8
"""

import argparse
import importlib.util
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from calib import BENCH_SEEDS  # noqa: E402
from evaluate import NOISE_PRESETS, append_rows, evaluate_case  # noqa: E402

_MODULES = {}


def load(label: str, path: str):
    if label not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"_sub_{label}", os.path.join(ROOT, path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[label] = mod
    return _MODULES[label]


def _job(args):
    label, path, N, n, seed, noise, repeats = args
    builder = load(label, path).create_dicke_state
    try:
        return evaluate_case(N, n, seed, noise, repeats, builder=builder, label=label)
    except Exception as e:  # a crash scores zero on judging day; record it, don't abort
        return dict(label=label, N=N, n=n, seed=seed, noise=noise, valid=False,
                    problems=f"{type(e).__name__}: {e}"[:300])


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", required=True, help="CSV file name under experiments/results/")
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--ns", type=int, nargs="+", default=None, help="only these n (default 1..N/2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BENCH_SEEDS))
    ap.add_argument("--noise", choices=NOISE_PRESETS, default="idle")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    subs = [s.split("=", 1) for s in args.sub]
    cases = [(N, n) for N in range(args.nmin, args.nmax + 1)
             for n in (args.ns or range(1, N // 2 + 1)) if 1 <= n <= N // 2]
    # largest N first so the slow density matrices don't trail at the end
    jobs = [(label, path, N, n, seed, args.noise, args.repeats)
            for N, n in sorted(cases, key=lambda c: -c[0]) for seed in args.seeds for label, path in subs]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r)
            nan = float("nan")
            print(f"{r['label']:>8} N={r['N']:2d} n={r['n']:2d} seed={r['seed']} valid={r['valid']} "
                  f"cz={r.get('cz', -1):4d} f={r.get('f', nan):.3f} x={r.get('x_inf_Mall', nan):.4f} "
                  f"E={r.get('score_mean_Mall', nan):7.3f} median={r.get('score_median_Mall', nan):6.3f} "
                  f"build={r.get('build_s', nan):.1f}s {r.get('problems', '')}", flush=True)
    rows.sort(key=lambda r: (r["N"], r["n"], r["seed"], r["label"]))
    append_rows(os.path.join(HERE, "results", args.out), rows)


if __name__ == "__main__":
    main_cli()
