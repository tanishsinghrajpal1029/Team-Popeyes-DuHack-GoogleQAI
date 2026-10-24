"""Controlled design comparisons on the noisy emulator (results are quoted in the notebook).

Every comparison uses the same calibration instances and the shot-noise-free expected official score
(`exp_score`), so differences are not artefacts of the 10,000-shot sampling noise.
"""
import sys
import time

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
import numpy as np

import dicke_solution as D
from fake_calibration import make_calibration
from noise_sim import evaluate

NOISE_MODELS = {
    'incoherent': {},
    'detuning50kHz': {'detuning_sigma_hz': 50e3, 'n_detuning_samples': 6},
    'spectator0.02': {'spectator_sigma': 0.02},
}


def select(N, n, cal_obj, dd=True, kinds=None, pick='best', rng=None):
    """Run the solution's candidate generator with a given Calibration object and selection rule."""
    cands = []
    for circ, qs, kind in D.generate_candidates(N, n, cal_obj, dd=dd):
        if kinds and kind not in kinds:
            continue
        cands.append((D.estimate_log_fidelity(circ, cal_obj, qs, n / N), circ, qs, kind))
    if pick == 'random':
        return cands[rng.integers(len(cands))]
    return max(cands, key=lambda t: t[0])


def perturbed(cal, rng):
    """'True' error rates = reported mean + N(0, reported sigma): the calibration itself is a noisy estimate."""
    out = []
    for d in cal:
        out.append({k: (float(max(v[0] + rng.normal(0, v[1]), 1e-5)), v[1]) for k, v in d.items()})
    return tuple(out)


def uniform_like(cal):
    """Topology-only calibration: same usable qubits/couplers, identical error rates."""
    return tuple({k: (1e-3, 0.0) for k in d} for d in cal)


def exp_dd(instances, seeds):
    print('== A. dynamical decoupling on/off ==', flush=True)
    for seed in seeds:
        cal = make_calibration(seed)
        cobj = D.Calibration(*cal)
        for N, n in instances:
            row = []
            for dd in (False, True):
                _, circ, qs, kind = select(N, n, cobj, dd=dd)
                for name, kw in NOISE_MODELS.items():
                    r = evaluate(circ, qs, n, cal, rng=np.random.default_rng(1), **kw)
                    row.append((name, dd, r['exp_score']))
            txt = '  '.join(f"{name}:{'DD' if dd else 'no'}={s:.3f}" for name, dd, s in row)
            print(f'seed {seed} ({N},{n}) {txt}', flush=True)


def exp_construction(instances, seeds):
    print('== B. construction families ==', flush=True)
    for seed in seeds:
        cal = make_calibration(seed)
        cobj = D.Calibration(*cal)
        for N, n in instances:
            kinds = ['tree'] if min(n, N - n) == 1 else ['path', 'split']
            out = []
            for kind in kinds:
                _, circ, qs, _ = select(N, n, cobj, kinds=[kind])
                ncz = sum(1 for op in circ.all_operations() if len(op.qubits) == 2)
                r = evaluate(circ, qs, n, cal, rng=np.random.default_rng(1))
                out.append(f'{kind}: CZ={ncz} moments={len(circ)} E[score]={r["exp_score"]:.3f}')
            print(f'seed {seed} ({N},{n}) ' + ' | '.join(out), flush=True)


def exp_placement(instances, seeds):
    print('== C. placement strategies (scored on perturbed "true" rates) ==', flush=True)
    for seed in seeds:
        cal = make_calibration(seed)
        truth = perturbed(cal, np.random.default_rng(100 + seed))
        for N, n in instances:
            res = {}
            for name, cobj, pick in [('uncertainty-aware(b=1)', D.Calibration(*cal, beta=1.0), 'best'),
                                     ('mean-only(b=0)', D.Calibration(*cal, beta=0.0), 'best'),
                                     ('topology-only', D.Calibration(*uniform_like(cal), beta=0.0), 'best'),
                                     ('random-candidate', D.Calibration(*cal), 'random')]:
                _, circ, qs, _ = select(N, n, cobj, pick=pick, rng=np.random.default_rng(seed))
                res[name] = evaluate(circ, qs, n, cal, rng=np.random.default_rng(1), true_cal=truth)['exp_score']
            print(f'seed {seed} ({N},{n}) ' + '  '.join(f'{k}={v:.3f}' for k, v in res.items()), flush=True)


if __name__ == '__main__':
    t = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    inst = [(6, 1), (6, 3), (8, 2), (8, 4)]
    if which in ('all', 'dd'):
        exp_dd(inst, [0, 1])
    if which in ('all', 'construction'):
        exp_construction([(6, 2), (6, 3), (8, 2), (8, 3), (8, 4), (9, 3)], [0, 1])
    if which in ('all', 'placement'):
        exp_placement([(6, 1), (6, 3), (8, 2)], [0, 1, 2])
    print(f'done in {time.time() - t:.0f}s')
