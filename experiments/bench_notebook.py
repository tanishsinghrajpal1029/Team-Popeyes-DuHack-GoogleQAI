"""Benchmark submission.ipynb with the notebook's own functions, for both tracks.

The notebook's code cells are executed in one namespace, as a top-to-bottom kernel run would
(the final sanity-check cell is skipped; `make_notebook.py --run` covers it). Everything that
builds, checks or scores a circuit then comes from that namespace, not from main.py:

Standard track. For each (N, n) and calibration seed:
  1. build with the notebook's `create_dicke_state` on a seeded calibration (timed);
  2. check with `validity.check_rules` (strict gate whitelist), `Willow105.validate_circuit`
     and the notebook's `test_circuit`;
  3. compute the exact noisy Z/X distributions (`evaluate.exact_distributions`, idle noise);
  4. draw `--runs` independent 10k-shot Z and X bitstring arrays and score each with the
     notebook's `overall_score`, for M = N and M = min(4, N) (M is undisclosed).
The same draws also go through `scoring.fast_overall_score`; the largest relative difference
is recorded, which checks the fast harness behind ARCHITECTURE.md against the verbatim scorer.
Judging sums one run per case, so run r of every case adds up to one sample of the total.

Bonus track. The notebook's `score_solution` calls `validate_circuit`, which the notebook
never defines. We supply one with the checks its text states: Willow routing, N returned
qubits, the exact state (on 16 sampled branches if there are mid-circuit measurements), the
readout-line rule, and a build under 60 s.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/bench_notebook.py --workers 28

Rows go to experiments/results/notebook_standard.csv (one per case and seed),
notebook_standard_totals.csv (one per seed, plus "all") and notebook_bonus.csv (one per case).
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import cirq
import cirq_google
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from calib import BENCH_SEEDS, make_calibration  # noqa: E402
from evaluate import NOISE_PRESETS, append_rows, circuit_stats, exact_distributions, git_rev  # noqa: E402
from scoring import fast_overall_score, score_distribution  # noqa: E402
from validity import check_rules  # noqa: E402

NOTEBOOK = os.path.join(ROOT, "submission.ipynb")
CHECK_CELL_PREFIX = "# Sanity check on the fake calibration"
N_SHOTS = 10_000
BUILD_LIMIT_S = 60.0

_NS = None


def load_notebook(path: str = NOTEBOOK) -> dict:
    """Run the notebook's code cells in order in one namespace and return it."""
    with open(path) as fh:
        nb = json.load(fh)
    ns: dict = {"__name__": "__main__"}
    for k, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] != "code" or src.startswith(CHECK_CELL_PREFIX):
            continue
        exec(compile(src, f"<submission.ipynb cell {k}>", "exec"), ns)
    return ns


def _init_worker(path: str) -> None:
    global _NS
    _NS = load_notebook(path)


def sample_bitstrings(p: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """10k shots from a distribution over 2^N indices (cirq order: first qubit = MSB)."""
    idx = rng.choice(len(p), size=N_SHOTS, p=p / p.sum())
    return (idx[:, None] >> np.arange(N - 1, -1, -1)) & 1


def standard_case(job: tuple) -> dict:
    N, n, seed, noise, runs = job
    ns = _NS
    cal = make_calibration(seed)
    t0 = time.time()
    circuit, qubits = ns["create_dicke_state"](N, n, **cal)
    build_s = time.time() - t0
    problems = check_rules(circuit, qubits, N, n, cal["cz_pauli_error_rate"], cal["sq_pauli_error_rate"])
    try:
        cirq_google.Willow105.validate_circuit(circuit)
        ns["test_circuit"](circuit, n, qubits)
    except Exception as e:
        problems.append(f"{type(e).__name__}: {e}")
    if build_s > BUILD_LIMIT_S:
        problems.append(f"build took {build_s:.1f} s")
    row = dict(label=git_rev(), N=N, n=n, K=math.comb(N, n), seed=seed, noise=noise, runs=runs,
               build_s=round(build_s, 3), valid=not problems, **circuit_stats(circuit))
    if problems:
        row["problems"] = "; ".join(problems)[:300]
        return row
    pz, px = exact_distributions(circuit, qubits, cal, **NOISE_PRESETS[noise], seed=seed)
    rng = np.random.default_rng([seed, N, n])
    Ms = {"Mall": N, "M4": min(4, N)}
    scores = {tag: np.empty(runs) for tag in Ms}
    max_rel_diff = 0.0
    t0 = time.time()
    for r in range(runs):
        z_bits = sample_bitstrings(pz, N, rng)
        x_bits = sample_bitstrings(px, N, rng)
        for tag, M in Ms.items():
            s = ns["overall_score"](z_bits, x_bits, n, M)
            scores[tag][r] = s
            fast = fast_overall_score(z_bits, x_bits, n, M)
            max_rel_diff = max(max_rel_diff, abs(fast - s) / abs(s))
    row["score_s"] = round(time.time() - t0, 1)
    row["max_rel_diff_fast"] = max_rel_diff
    for tag, M in Ms.items():
        s = scores[tag]
        row.update({f"median_{tag}": float(np.median(s)), f"mean_{tag}": float(s.mean()),
                    f"p10_{tag}": float(np.percentile(s, 10)), f"p90_{tag}": float(np.percentile(s, 90))})
        # the fast harness on the same distributions, 20k runs: the reference for this row
        fast = score_distribution(pz, px, N, n, M, repeats=20_000, seed=seed)
        row[f"fast_median_{tag}"] = fast["score_median"]
        row[f"fast_mean_{tag}"] = fast["score_mean"]
    row["scores"] = {tag: s.tolist() for tag, s in scores.items()}  # stripped before the CSV
    return row


def make_validate_circuit(ns: dict):
    """The judges' `validate_circuit(circuit, num_excitations, qubits)`, from the bonus text."""
    device_qubits = cirq_google.Willow105.metadata.qubit_set

    def validate_circuit(circuit: cirq.Circuit, num_excitations: int, qubits: set) -> None:
        cirq_google.Willow105.validate_circuit(circuit)
        measured = {q for op in circuit.all_operations() if cirq.is_measurement(op) for q in op.qubits}
        lines = {q.row + q.col for q in measured}
        partial = {q for q in device_qubits if q.row + q.col in lines} - measured
        assert not partial, f"readout line measured only in part: {sorted(partial)}"
        assert not measured & set(qubits), "a returned qubit is measured"
        if measured:  # the state must be exact on every branch; sample several
            for s in range(16):
                ns["test_circuit"](circuit, num_excitations, qubits, sampler=cirq.Simulator(seed=s))
        else:
            ns["test_circuit"](circuit, num_excitations, qubits)

    return validate_circuit


def bonus(ns: dict) -> tuple[float, list[dict]]:
    """Run the notebook's `score_solution` with a timing wrapper around the bonus entry."""
    rows = []
    build = ns["create_dicke_state_adaptive"]

    def timed(num_qubits, num_excitations):
        t0 = time.time()
        circuit, qubits = build(num_qubits, num_excitations)
        build_s = time.time() - t0
        if build_s > BUILD_LIMIT_S:  # the rules count a slow build as NotImplementedError
            print(f"bonus D({num_qubits},{num_excitations}) took {build_s:.1f} s: no point", flush=True)
            raise NotImplementedError
        rows.append(dict(label=git_rev(), N=num_qubits, n=num_excitations, moments=len(circuit),
                         points=num_qubits * num_excitations / len(circuit), build_s=round(build_s, 3),
                         cz=sum(op.gate == cirq.CZ for op in circuit.all_operations()),
                         qubits_touched=len(circuit.all_qubits()),
                         measurements=sum(cirq.is_measurement(op) for op in circuit.all_operations())))
        return circuit, qubits

    ns["validate_circuit"] = make_validate_circuit(ns)
    ns["create_dicke_state_adaptive"] = timed
    try:
        total = ns["score_solution"]()
    finally:
        ns["create_dicke_state_adaptive"] = build
    return total, rows


def totals(rows: list[dict], seeds: list[int]) -> list[dict]:
    """Distribution of the judged total (one run per case, summed) per seed and pooled."""
    out = []
    for seed in seeds + ["all"]:
        sel = [r for r in rows if r["valid"] and (seed == "all" or r["seed"] == seed)]
        row = dict(label=git_rev(), seed=seed, cases=len(sel))
        for tag in ("Mall", "M4"):
            if seed == "all":  # mean over calibrations of the per-seed totals
                per_seed = [sum(np.array(r["scores"][tag]) for r in sel if r["seed"] == s) for s in seeds]
                t = np.concatenate(per_seed)
            else:
                t = sum(np.array(r["scores"][tag]) for r in sel)
            row.update({f"total_median_{tag}": float(np.median(t)), f"total_mean_{tag}": float(t.mean()),
                        f"total_p10_{tag}": float(np.percentile(t, 10)),
                        f"total_p90_{tag}": float(np.percentile(t, 90)),
                        f"sum_median_{tag}": sum(r[f"median_{tag}"] for r in sel) / (len(seeds) if seed == "all" else 1),
                        f"sum_fast_mean_{tag}": sum(r[f"fast_mean_{tag}"] for r in sel) / (len(seeds) if seed == "all" else 1)})
        out.append(row)
    return out


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=NOTEBOOK)
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BENCH_SEEDS))
    ap.add_argument("--noise", choices=NOISE_PRESETS, default="idle")
    ap.add_argument("--runs", type=int, default=1000, help="10k-shot runs per case, verbatim scorer")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-standard", action="store_true")
    ap.add_argument("--skip-bonus", action="store_true")
    args = ap.parse_args()

    pool = res = None
    if not args.skip_standard:
        cases = [(N, n) for N in range(args.nmin, args.nmax + 1) for n in range(1, N // 2 + 1)]
        jobs = [(N, n, seed, args.noise, args.runs)
                for N, n in sorted(cases, key=lambda c: -c[0]) for seed in args.seeds]
        pool = Pool(args.workers, initializer=_init_worker, initargs=(args.notebook,))
        res = pool.map_async(standard_case, jobs, chunksize=1)

    if not args.skip_bonus:  # serial, in this process, while the pool works
        t0 = time.time()
        total, brows = bonus(load_notebook(args.notebook))
        for r in brows:
            print(f"bonus D({r['N']:2d},{r['n']}): {r['moments']:4d} moments, {r['points']:.3f} points, "
                  f"built in {r['build_s']:.1f} s", flush=True)
        print(f"bonus score_solution() = {total:.4f} over {len(brows)} cases ({time.time() - t0:.0f} s)\n",
              flush=True)
        append_rows(os.path.join(HERE, "results", "notebook_bonus.csv"), brows)

    if pool is not None:
        rows = sorted(res.get(), key=lambda r: (r["N"], r["n"], r["seed"]))
        pool.close()
        nan = float("nan")
        for r in rows:
            print(f"D({r['N']:2d},{r['n']}) seed={r['seed']} valid={r['valid']} cz={r['cz']:4d} "
                  f"moments={r['moments']:4d} build={r['build_s']:.1f}s "
                  f"median={r.get('median_Mall', nan):7.3f} (fast {r.get('fast_median_Mall', nan):7.3f}) "
                  f"mean={r.get('mean_Mall', nan):7.3f} (fast {r.get('fast_mean_Mall', nan):7.3f}) "
                  f"M4 median={r.get('median_M4', nan):7.3f} "
                  f"|verbatim-fast|/score<={r.get('max_rel_diff_fast', nan):.1e} {r.get('problems', '')}")
        trows = totals(rows, args.seeds)
        for t in trows:
            print(f"total seed={t['seed']}: median {t['total_median_Mall']:.2f} "
                  f"(p10 {t['total_p10_Mall']:.2f}, p90 {t['total_p90_Mall']:.2f}), "
                  f"mean {t['total_mean_Mall']:.2f}; sum of medians {t['sum_median_Mall']:.2f}, "
                  f"sum of fast E[score] {t['sum_fast_mean_Mall']:.2f} | M4 median {t['total_median_M4']:.2f}")
        for r in rows:
            r.pop("scores", None)
        append_rows(os.path.join(HERE, "results", "notebook_standard.csv"), rows)
        append_rows(os.path.join(HERE, "results", "notebook_standard_totals.csv"), trows)


if __name__ == "__main__":
    main_cli()
