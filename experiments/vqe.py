"""Does a VQE-style (variational, noise-aware) approach buy anything beyond synth.py?

Plain VQE (tune a parametrized circuit on hardware against an energy) does not fit the
challenge: the build must finish in 60 s with no hardware access, and the noiseless
circuit must be exact. Its classical half, gradient optimization of an Ry + CZ ansatz
against the known target, is what synth.py already does. What is left to exploit is the
freedom inside that: a fixed CZ structure has many exact angle sets (different restarts
converge to different ones), and they behave differently under noise.

  spread     Exact candidates of each deployed structure under the calibrated noise model
             (cached in results/vqe/). Scored on calibration seeds 0, 1, 2.
  coherent   lambda (chi^2 noncentrality at 10k shots) per candidate from coherent errors
             the noise model lacks: Ry over-rotation (first order: real amplitudes move),
             CZ conditional-phase and stray Z phases (second order for a real target).
  fragility  The same over-rotation measure for main.build's other constructions
             (B-E routed, W tree vs chain), as a reference.
  pick       One candidate per case: minimum worst-case regret over an assumed range of
             over-rotation sizes, chosen on seed 0, reported on held-out seeds 1, 2.
  gap        The notebook's check is np.isclose(fidelity, 1), i.e. infidelity <~ 1e-5. Best
             infidelity with one CZ fewer than deployed: are approximate circuits a lever?

  python experiments/vqe.py spread --cases 4,2 5,2 6,2 6,3 --samples 24
  python experiments/vqe.py coherent --cases 4,2 5,2 6,2 6,3
  python experiments/vqe.py pick --cases 4,2 5,2 6,2 6,3
  python experiments/vqe.py fragility --cases 4,1 6,1 8,1 4,2 5,2 6,2 6,3 7,3
  python experiments/vqe.py gap --cases 4,2 5,2 6,2 6,3
  python experiments/vqe.py apply            # picks -> results/vqe/picks.json
  python experiments/synth.py --export       # deploys the picks into main.py
"""

import argparse
import json
import math
import os
import sys
import time

import cirq
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import main  # noqa: E402
from evaluate import append_rows, evaluate_case  # noqa: E402
from synth import Circuit, dicke_vector, drop_cz, optimize  # noqa: E402

TOL = 1e-13
CANDS = os.path.join(HERE, "results", "vqe")


def exact_family(N, k, cz_layers, deployed, samples, seed):
    """`samples` exact parameter sets for a fixed structure; the deployed one comes first."""
    circ = Circuit(N, cz_layers)
    target = dicke_vector(N, k)
    rng = np.random.default_rng(seed)
    found, tries = [np.array(deployed)], 0
    while len(found) < samples and tries < 20 * samples:
        tries += 1
        f, x = optimize(circ, target, rng.uniform(-math.pi, math.pi, circ.num_params))
        if f < TOL:
            found.append(x)
    return found, tries


def load_or_make_family(N, k, entry, samples, seed):
    """Exact candidates for (N, k), cached in results/vqe/ so later checks reuse them."""
    path = os.path.join(CANDS, f"D{N}_{k}_cands_s{seed}.json")
    if os.path.exists(path):
        with open(path) as fh:
            cached = json.load(fh)
        family = [np.array(p) for p in cached["params"]]
        same = ([[list(e) for e in l] for l in cached["cz_layers"]]
                == [[list(e) for e in l] for l in entry["cz_layers"]])
        if same and len(family) >= samples:  # a re-exported structure invalidates the cache
            return family[:samples]
    t0 = time.time()
    family, tries = exact_family(N, k, entry["cz_layers"], entry["params"], samples, seed)
    print(f"D^{N}_{k}: {len(family) - 1} new exact solutions from {tries} restarts "
          f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(CANDS, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dict(N=N, k=k, cz_layers=entry["cz_layers"], seed=seed,
                       params=[[float(v) for v in p] for p in family]), fh)
    return family


def synth_builder(entry):
    """main.build restricted to one synthesized entry (same placement and compile)."""
    def build(N, n, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors):
        g = main._usable_graph(cz_pauli_error_rate, sq_pauli_error_rate)
        qs = main._place_synth(g, entry, n, sq_pauli_error_rate, zero_errors, one_errors)
        return main._to_cz_phxz(main._synth_circuit(entry, qs)), set(qs)
    return build


def expected_loss(row):
    """Shot-noise-free E[z + x]: E[chi^2] = K - 1 + lambda, z = chi^2 / (K f)."""
    K = row["K"]
    return (K - 1 + row["lam"]) / (K * row["f"]) + row["x_inf_Mall"]


def run_spread(args):
    rows = []
    for N, k in args.cases:
        entry = main._SYNTH[(N, k)]
        family = load_or_make_family(N, k, entry, args.samples, args.seed)
        for i, params in enumerate(family):
            if args.only is not None and i not in args.only:
                continue
            cand = dict(entry, params=[float(v) for v in params])
            for cal_seed in args.cal_seeds:
                r = evaluate_case(N, k, cal_seed, args.noise, args.repeats,
                                  builder=synth_builder(cand), label=f"vqe-spread {args.noise} cand={i}")
                r["candidate"] = i
                r["loss_exp"] = expected_loss(r)
                rows.append(r)
            print(f"  cand {i:2d}: " + "  ".join(
                f"s{r['seed']} f={r['f']:.4f} lam={r['lam']:.3f} x={r['x_inf_Mall']:.4f} "
                f"E[loss]={r['loss_exp']:.4f}"
                for r in rows[-len(args.cal_seeds):]), flush=True)
    append_rows(os.path.join(HERE, "results", args.out), rows)
    summarize_spread(rows, args.cal_seeds, args.pick)


def summarize_spread(rows, cal_seeds, fixed_pick=None):
    """Deployed vs. the picked candidate (lowest E[loss] on the first seed unless
    `fixed_pick` names it per case), judged on every seed."""
    pick_seed = cal_seeds[0]
    print(f"\npick = {'fixed' if fixed_pick else f'lowest E[loss] on calibration seed {pick_seed}'}; "
          "values are E[loss] (lower is better)")
    for N, k in sorted({(r["N"], r["n"]) for r in rows}):
        case = [r for r in rows if (r["N"], r["n"]) == (N, k)]
        by = {(r["candidate"], r["seed"]): r for r in case}
        cands = sorted({r["candidate"] for r in case})
        pick = (fixed_pick or {}).get((N, k))
        if pick is None:
            pick = min(cands, key=lambda c: by[(c, pick_seed)]["loss_exp"])
        print(f"D^{N}_{k} ({len(cands)} exact candidates, pick = cand {pick})")
        for s in cal_seeds:
            vals = np.array([by[(c, s)]["loss_exp"] for c in cands])
            dep, chosen = by[(0, s)], by[(pick, s)]
            print(f"  seed {s}: deployed {dep['loss_exp']:.4f}  picked {chosen['loss_exp']:.4f}  "
                  f"range [{vals.min():.4f}, {vals.max():.4f}]  "
                  f"f {dep['f']:.4f}->{chosen['f']:.4f}  lam {dep['lam']:.3f}->{chosen['lam']:.3f}  "
                  f"x {dep['x_inf_Mall']:.4f}->{chosen['x_inf_Mall']:.4f}  "
                  f"%ceil {100 * dep['frac_ceiling_Mall']:.1f}->{100 * chosen['frac_ceiling_Mall']:.1f}")


def phase_error_state(N, cz_layers, params, cz_eps=0.0, z_q=None):
    """Noiseless state with coherent phase errors: every CZ is CPhase(pi + cz_eps) and
    leaves a stray Rz(z_q[q]) on each qubit q it touches (uncompensated dynamic phase).
    Exact at the Ry + CZ level: merging Ry's across layers (what the compiler does) commutes
    with these errors, which only act on qubits inside a CZ."""
    z_q = np.zeros(N) if z_q is None else np.asarray(z_q)
    bits = (np.arange(2**N)[:, None] >> (N - 1 - np.arange(N))) & 1
    th = np.array(params).reshape(-1, N)
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0
    for l in range(len(th)):
        for q in range(N):
            c, s = math.cos(th[l, q] / 2), math.sin(th[l, q] / 2)
            v = psi.reshape(2**q, 2, -1)
            psi = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1).reshape(-1)
        if l < len(cz_layers):
            phase = sum((math.pi + cz_eps) * bits[:, i] * bits[:, j]
                        + z_q[i] * bits[:, i] + z_q[j] * bits[:, j] for i, j in cz_layers[l])
            psi = psi * np.exp(1j * phase)
    return psi


def _lam(p, valid):
    """Chi^2 noncentrality at 10k shots from exact Z-basis probabilities."""
    pv = p[valid]
    return float(1e4 * pv.sum() * np.mean((pv / pv.mean() - 1) ** 2))


def compiled_rot_lambda(circuit, qubits, k, sigma, per_qubit=True):
    """lambda from over-rotating the compiled circuit's PhasedXZ gates, x_exponent ->
    x_exponent (1 + sigma). per_qubit: the expectation for independent per-qubit errors
    with sd sigma, which (lambda being quadratic) is the sum of one-qubit-at-a-time
    lambdas; otherwise every qubit is over-rotated by the same sigma."""
    data = sorted(qubits)
    order = data + sorted(circuit.all_qubits() - set(qubits))
    valid = np.bitwise_count(np.arange(2 ** len(data))) == k
    sim = cirq.Simulator(dtype=np.complex128)

    def lam(targets):
        def over_rotate(op):
            g = op.gate
            if op.qubits[0] in targets and isinstance(g, cirq.PhasedXZGate):
                return cirq.PhasedXZGate(x_exponent=g.x_exponent * (1 + sigma), z_exponent=g.z_exponent,
                                         axis_phase_exponent=g.axis_phase_exponent).on(*op.qubits)
            return op
        psi = sim.simulate(circuit.map_operations(over_rotate), qubit_order=order).final_state_vector
        return _lam((np.abs(psi) ** 2).reshape(2 ** len(data), -1).sum(axis=1), valid)

    qs = circuit.all_qubits()
    return sum(lam({q}) for q in qs) if per_qubit else lam(qs)


def run_coherent(args):
    """Imbalance (lambda at 10k shots) caused by coherent miscalibration alone, per candidate.

    Over-rotations are applied to the compiled circuit (placed for calibration seed 0),
    because the compiler merges consecutive Ry's on a qubit that skips a CZ layer and an
    over-rotated merged gate differs from merged over-rotated gates."""
    from calib import make_calibration
    cal = make_calibration(0)
    rows = []
    for N, k in args.cases:
        entry = main._SYNTH[(N, k)]
        layers = [list(map(tuple, l)) for l in entry["cz_layers"]]
        family = load_or_make_family(N, k, entry, args.samples, args.seed)
        valid = np.bitwise_count(np.arange(2**N)) == k
        print(f"D^{N}_{k}: rot = uniform {args.rot_eps:.1%} over-rotation, rot_rand = per-qubit sd "
              f"{args.rot_sigma:.1%}, cz = {args.cz_eps} rad CPhase error, z_rand = per-qubit stray "
              f"CZ phase sd {args.z_sigma} rad")
        for i, params in enumerate(family):
            circuit, qubits = synth_builder(dict(entry, params=[float(v) for v in params]))(N, k, **cal)
            lz = sum(_lam(np.abs(phase_error_state(N, layers, params, z_q=args.z_sigma * np.eye(N)[q])) ** 2, valid)
                     for q in range(N))
            row = dict(N=N, k=k, candidate=i, rot_eps=args.rot_eps, rot_sigma=args.rot_sigma,
                       cz_eps=args.cz_eps, z_sigma=args.z_sigma,
                       lam_rot=compiled_rot_lambda(circuit, qubits, k, args.rot_eps, per_qubit=False),
                       lam_rot_rand=compiled_rot_lambda(circuit, qubits, k, args.rot_sigma),
                       lam_cz=_lam(np.abs(phase_error_state(N, layers, params, cz_eps=args.cz_eps)) ** 2, valid),
                       lam_z_rand=lz)
            rows.append(row)
            print(f"  cand {i:2d}: lam rot={row['lam_rot']:7.3f} rot_rand={row['lam_rot_rand']:7.3f} "
                  f"z_rand={row['lam_z_rand']:7.3f} cz={row['lam_cz']:.3f}", flush=True)
    append_rows(os.path.join(HERE, "results", args.out), rows)


def run_fragility(args):
    """Over-rotation sensitivity of what main.build actually returns, per construction."""
    from calib import make_calibration
    rows = []
    variants = dict(default={}, **{"B-E routed": dict(synth=False, scs=False), "W chain": dict(w_tree=False)})
    for N, k in args.cases:
        cal = make_calibration(args.cal_seed)
        for name, opts in variants.items():
            if name == "W chain" and k != 1:
                continue
            circuit, qubits = main.build(N, k, **cal, **opts)
            cz = sum(op.gate == cirq.CZ for op in circuit.all_operations())
            lam = compiled_rot_lambda(circuit, qubits, k, args.rot_sigma)
            rows.append(dict(N=N, k=k, variant=name, cal_seed=args.cal_seed, cz=cz,
                             rot_sigma=args.rot_sigma, lam_rot_rand=lam))
            print(f"D^{N}_{k} {name:10s} cz={cz:3d}  lam_rot_rand={lam:8.3f}", flush=True)
    append_rows(os.path.join(HERE, "results", args.out), rows)


def run_pick(args):
    """Choose one exact candidate per case from the spread (idle E[loss]) and coherent
    (lambda_rot_rand) CSVs. Coherent over-rotation of size sigma adds about
    lambda_rot_rand (sigma / rot_sigma)^2 / (K f) to E[loss]; sigma on hardware is unknown, so
    the pick minimizes the worst regret over a sigma grid, on calibration seed 0 only.
    Seeds 1, 2 are held out and reported."""
    import csv
    with open(os.path.join(HERE, "results", args.spread)) as fh:
        spread = list(csv.DictReader(fh))
    with open(os.path.join(HERE, "results", args.coherent)) as fh:
        coh = list(csv.DictReader(fh))
    lam = {(int(r["N"]), int(r["k"]), int(r["candidate"])): float(r["lam_rot_rand"]) for r in coh}
    ref = {(int(r["N"]), int(r["k"])): float(r["rot_sigma"]) for r in coh}
    idle = {}
    for r in spread:
        idle[(int(r["N"]), int(r["n"]), int(r["candidate"]), int(r["seed"]))] = (float(r["loss_exp"]), float(r["f"]))
    rows = []
    for N, k in args.cases:
        cands = sorted({c for (n_, k_, c, _) in idle if (n_, k_) == (N, k) and (N, k, c) in lam})

        def loss(c, seed, sigma):
            e, f = idle[(N, k, c, seed)]
            return e + lam[(N, k, c)] * (sigma / ref[(N, k)]) ** 2 / (math.comb(N, k) * f)

        def regret(c):
            return max(loss(c, 0, sg) - min(loss(d, 0, sg) for d in cands) for sg in args.sigmas)

        pick = min(cands, key=regret)
        print(f"D^{N}_{k}: pick cand {pick} of {len(cands)} (lam_rot_rand at {ref[(N, k)]:.1%}: "
              f"{lam[(N, k, 0)]:.1f} -> {lam[(N, k, pick)]:.1f}); E[loss] on held-out seeds 1, 2:")
        for sg in args.sigmas:
            dep = np.mean([loss(0, s, sg) for s in (1, 2)])
            new = np.mean([loss(pick, s, sg) for s in (1, 2)])
            rows.append(dict(N=N, k=k, pick=pick, sigma=sg, loss_deployed=dep, loss_pick=new,
                             change=(new - dep) / dep))
            print(f"  sigma {sg:5.1%}: deployed {dep:.3f}  pick {new:.3f}  ({100 * (new - dep) / dep:+.1f}%)")
    append_rows(os.path.join(HERE, "results", args.out), rows)


def run_gap(args):
    rows = []
    for N, k in args.cases:
        entry = main._SYNTH[(N, k)]
        layers = [list(map(tuple, l)) for l in entry["cz_layers"]]
        x0 = np.array(entry["params"])
        target = dicke_vector(N, k)
        rng = np.random.default_rng(args.seed)
        best_overall = math.inf
        for li, layer in enumerate(layers):
            for e in layer:
                circ = Circuit(N, drop_cz(layers, li, e))
                best = optimize(circ, target, x0)[0]  # warm start: the exact solution minus one CZ
                for _ in range(args.restarts):
                    best = min(best, optimize(circ, target, rng.uniform(-math.pi, math.pi, circ.num_params))[0])
                rows.append(dict(N=N, k=k, cz=sum(map(len, layers)) - 1, layer=li, edge=f"{e[0]}-{e[1]}",
                                 restarts=args.restarts, min_infidelity=best))
                best_overall = min(best_overall, best)
                print(f"D^{N}_{k} drop CZ{e} from layer {li}: min infidelity {best:.2e}", flush=True)
        print(f"D^{N}_{k}: best with one fewer CZ = {best_overall:.2e} "
              f"(notebook check needs <~1e-5)\n", flush=True)
    append_rows(os.path.join(HERE, "results", args.out), rows)


def run_apply(args):
    """Turn pick results into results/vqe/picks.json, which synth.py --export consumes.

    Each entry keeps the deployed parameters (candidate 0) next to the pick, so the exporter
    only substitutes a pick that was computed for the structure and solution it deploys.
    """
    import csv
    picks_path = os.path.join(CANDS, "picks.json")
    picks = {}
    if os.path.exists(picks_path):
        with open(picks_path) as fh:
            picks = json.load(fh)
    for name in args.pick_csvs:
        with open(os.path.join(HERE, "results", name)) as fh:
            chosen = {(int(r["N"]), int(r["k"])): int(r["pick"]) for r in csv.DictReader(fh)}
        for (N, k), idx in sorted(chosen.items()):
            with open(os.path.join(CANDS, f"D{N}_{k}_cands_s{args.seed}.json")) as fh:
                fam = json.load(fh)
            picks[f"{N},{k}"] = dict(cz_layers=fam["cz_layers"], candidate=idx,
                                     deployed_params=fam["params"][0], params=fam["params"][idx])
            print(f"D^{N}_{k}: candidate {idx}")
    with open(picks_path, "w") as fh:
        json.dump(picks, fh)
    print(f"wrote {picks_path} ({len(picks)} cases); now run: python experiments/synth.py --export")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    apl = sub.add_parser("apply", help="write picks.json for synth.py --export")
    apl.add_argument("--pick-csvs", nargs="+", default=["vqe_pick.csv"])
    apl.add_argument("--seed", type=int, default=0)
    sp = sub.add_parser("spread")
    sp.add_argument("--samples", type=int, default=24, help="exact candidates per case, incl. deployed")
    sp.add_argument("--cal-seeds", type=int, nargs="+", default=[0, 1, 2])
    sp.add_argument("--noise", default="idle")
    sp.add_argument("--repeats", type=int, default=4000)
    sp.add_argument("--out", default="vqe_spread.csv")
    sp.add_argument("--only", type=int, nargs="+", help="evaluate only these candidate indices")
    sp.add_argument("--pick", nargs="+", help="fixed picks N,k,index (default: best on first seed)")
    gp = sub.add_parser("gap")
    gp.add_argument("--restarts", type=int, default=8)
    gp.add_argument("--out", default="vqe_gap.csv")
    cp = sub.add_parser("coherent")
    cp.add_argument("--samples", type=int, default=24)
    cp.add_argument("--rot-eps", type=float, default=0.01, help="relative Ry over-rotation")
    cp.add_argument("--cz-eps", type=float, default=0.02, help="CZ conditional-phase error (rad)")
    cp.add_argument("--rot-sigma", type=float, default=0.01, help="sd of per-qubit over-rotation")
    cp.add_argument("--z-sigma", type=float, default=0.02, help="sd of per-qubit stray CZ Z phase (rad)")
    cp.add_argument("--out", default="vqe_coherent.csv")
    fp = sub.add_parser("fragility")
    fp.add_argument("--rot-sigma", type=float, default=0.01, help="sd of per-qubit over-rotation")
    fp.add_argument("--cal-seed", type=int, default=0)
    fp.add_argument("--out", default="vqe_fragility.csv")
    pp = sub.add_parser("pick")
    pp.add_argument("--spread", default="vqe_spread.csv")
    pp.add_argument("--coherent", default="vqe_coherent.csv")
    pp.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.001, 0.002, 0.003, 0.005, 0.01])
    pp.add_argument("--out", default="vqe_pick.csv")
    for p in (sp, gp, cp, fp, pp):
        p.add_argument("--cases", nargs="+", required=True, help="N,k pairs (spread, gap, coherent: must be in main._SYNTH)")
        p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "apply":
        return run_apply(args)
    args.cases = [tuple(int(v) for v in c.split(",")) for c in args.cases]
    if getattr(args, "pick", None):
        args.pick = {tuple(int(v) for v in p.split(",")[:2]): int(p.split(",")[2]) for p in args.pick}
    {"spread": run_spread, "gap": run_gap, "coherent": run_coherent,
     "fragility": run_fragility, "pick": run_pick}[args.cmd](args)


if __name__ == "__main__":
    main_cli()
