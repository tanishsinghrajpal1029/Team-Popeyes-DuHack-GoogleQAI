"""Seed-averaged summary of result CSVs, normalized to the perfect-state ceiling.

Columns: f (right-weight fraction), chi2/(K-1) (1 = shot noise only; the excess is
population imbalance), x (X term, infinite shots), loss = mean of z + x over simulated
runs, E[score] = expected single-draw score (the objective; semi-analytic, see
scoring.expected_score), median score, the perfect-state median ceiling, and the
fraction of that ceiling.

Example:
  python experiments/summarize.py w_tree.csv baseline.csv --labels w-tree-v2 chain
"""

import argparse
import collections
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scoring import ceiling, expected_score  # noqa: E402


def load(paths, labels=None, noise=None):
    rows = []
    for p in paths:
        with open(os.path.join(HERE, "results", p)) as fh:
            rows += [r for r in csv.DictReader(fh)
                     if (not labels or r["label"] in labels) and (not noise or r.get("noise") == noise)]
    return rows


def summarize(rows, tag="Mall"):
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("valid", "True") == "True":
            groups[(r["label"], r.get("noise", ""), int(r["N"]), int(r["n"]))].append(r)
    out = []
    for (label, noise, N, n), rs in sorted(groups.items(), key=lambda t: (t[0][2], t[0][3], t[0][0])):
        g = lambda k: st.mean(float(r[k]) for r in rs)
        K = int(rs[0]["K"])
        chi2_ratio = g(f"chi2_ratio_{tag}") if f"chi2_ratio_{tag}" in rs[0] and rs[0][f"chi2_ratio_{tag}"] else 1 + g("lam") / max(K - 1, 1)
        loss = g(f"loss_mean_{tag}") if rs[0].get(f"loss_mean_{tag}") else g(f"z_mean_{tag}") + g(f"x_mean_{tag}")
        M = N if tag == "Mall" else min(4, N)
        ceil = ceiling(N, n, M)["score_median"]
        x_mean = g(f"x_mean_{tag}")
        out.append(dict(label=label, noise=noise, N=N, n=n, K=K, cz=g("cz"), moments=g("moments"),
                        expected=expected_score(K, g("f"), g("lam"), x_mean),
                        ceiling_expected=expected_score(K, 1.0, 0.0, ceiling(N, n, M)["x_mean"]),
                        f=g("f"), chi2_ratio=chi2_ratio, x=g(f"x_inf_{tag}"), loss=loss,
                        median=g(f"score_median_{tag}"), ceiling=ceil,
                        frac=g(f"score_median_{tag}") / ceil, seeds=len(rs)))
    return out


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--noise")
    ap.add_argument("--M4", action="store_true", help="use max_correlator_size = 4 instead of N")
    args = ap.parse_args()
    rows = summarize(load(args.csvs, args.labels, args.noise), "M4" if args.M4 else "Mall")
    print(f"{'label':28s} {'noise':5s} {'(N,n)':7s} {'K':>4s} {'CZ':>5s} {'f':>6s} {'chi2/(K-1)':>10s} "
          f"{'x':>7s} {'loss':>6s} {'E[score]':>8s} {'median':>7s} {'ceiling':>7s} {'frac':>5s}")
    for r in rows:
        print(f"{r['label'][:28]:28s} {r['noise']:5s} ({r['N']},{r['n']}){'':{max(0, 4 - len(str(r['N'])) - len(str(r['n'])))}s} "
              f"{r['K']:4d} {r['cz']:5.0f} {r['f']:6.3f} {r['chi2_ratio']:10.3f} {r['x']:7.4f} "
              f"{r['loss']:6.3f} {r['expected']:8.3f} {r['median']:7.3f} {r['ceiling']:7.3f} {100 * r['frac']:4.0f}%")


if __name__ == "__main__":
    main_cli()
