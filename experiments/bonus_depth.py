"""Bonus track: moments of every candidate per (N, n), exactness of each, and the total.

For each case the dispatcher's candidates (main.bonus_candidates) are built, checked
(noiseless fidelity, Willow105 routing, strict gate whitelist) and listed with their moment
count. The winner per case is what create_dicke_state_adaptive returns; the total is the
notebook's score_solution. Rows go to experiments/results/bonus_variants.csv; a summary with
the winner per case and the total to stdout.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/bonus_depth.py --workers 28
  .venv/bin/python experiments/bonus_depth.py --baseline experiments/results/notebook_bonus.csv
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import cirq_google

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import main  # noqa: E402
from evaluate import git_rev  # noqa: E402
from validity import check_rules, noiseless_fidelity  # noqa: E402

CASES = [(N, n) for N in range(2, 16) for n in range(1, N // 2 + 1)]


def _case(args):
    N, n = args
    cal = main._uniform_calibration()
    rows = []
    t0 = time.time()
    cands = main.bonus_candidates(N, n)
    build_s = time.time() - t0
    for circ, qs, label in cands:
        problems = check_rules(circ, set(qs), N, n, cal['cz_pauli_error_rate'], cal['sq_pauli_error_rate'])
        try:
            cirq_google.Willow105.validate_circuit(circ)
            routed = True
        except ValueError:
            routed = False
        fid = noiseless_fidelity(circ, set(qs), n)
        rows.append(dict(N=N, n=n, label=label, moments=len(circ), fidelity=f'{fid:.3e}',
                         exact=abs(fid - 1) < 1e-9, routed=routed, rules_ok=not problems,
                         cz=sum(1 for op in circ.all_operations() if len(op.qubits) == 2),
                         build_s=round(build_s, 2)))
    return rows


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'bonus_variants.csv'))
    ap.add_argument('--baseline', default=os.path.join(HERE, 'results', 'notebook_bonus.csv'),
                    help='CSV with N, n, moments columns to compare against (default: last notebook bench)')
    ap.add_argument('--cases', nargs='*', help='N,n pairs (default: all 56 bonus cases)')
    args = ap.parse_args()
    cases = [tuple(map(int, c.split(','))) for c in args.cases] if args.cases else CASES
    with Pool(min(args.workers, len(cases))) as pool:
        results = pool.map(_case, cases)
    rows = [r for rs in results for r in rs]
    rev = git_rev()
    for r in rows:
        r['label_rev'] = rev
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    base = {}
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as fh:
            for r in csv.DictReader(fh):
                base[(int(r['N']), int(r['n']))] = int(r['moments'])
    total = 0.0
    total_base = 0.0
    bad = []
    print(f'{"case":>7} {"winner":>34} {"moments":>7} {"before":>6} {"points":>7}')
    for N, n in cases:
        rs = [r for r in rows if r['N'] == N and r['n'] == n]
        ok = [r for r in rs if r['exact'] and r['routed'] and r['rules_ok']]
        bad += [r for r in rs if not (r['exact'] and r['routed'] and r['rules_ok'])]
        if not ok:
            print(f'D{N},{n}: NO VALID CANDIDATE')
            continue
        win = min(ok, key=lambda r: r['moments'])
        pts = N * n / win['moments']
        total += pts
        b = base.get((N, n))
        if b:
            total_base += N * n / b
        flag = '' if b is None or win['moments'] <= b else '  REGRESSION'
        print(f'D{N:>2},{n:<2} {win["label"]:>34} {win["moments"]:>7} {b if b else "-":>6} {pts:>7.3f}{flag}')
    print(f'\nscore_solution total: {total:.4f}' + (f'  (baseline {total_base:.4f})' if base else ''))
    if bad:
        print(f'\n{len(bad)} candidate(s) failed a check (not eligible as winners):')
        for r in bad:
            print(f"  D{r['N']},{r['n']} {r['label']}: exact={r['exact']} routed={r['routed']} rules_ok={r['rules_ok']} fidelity={r['fidelity']}")
    print(f'wrote {os.path.relpath(args.out, ROOT)} ({len(rows)} rows)')


if __name__ == '__main__':
    main_cli()
