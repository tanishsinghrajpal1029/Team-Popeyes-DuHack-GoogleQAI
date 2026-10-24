"""Freeze the N <= 8 submission: every built circuit must match tests/golden_small_n.json.

Big-N work must not change small-N circuits by accident. Regenerate only deliberately,
for a case that measured strictly better (before/after numbers in the commit message):
  python tests/test_golden_small_n.py --regen
"""

import functools
import hashlib
import json
import os
import sys

import cirq
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]

from calib import make_calibration  # noqa: E402
from main import create_dicke_state  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden_small_n.json")
SEEDS = (0, 1, 2)
CASES = [(seed, N, n) for seed in SEEDS for N in range(1, 9) for n in range(N + 1)]


@functools.lru_cache(maxsize=None)
def _cal(seed):
    return make_calibration(seed)


def snapshot(seed, N, n):
    circuit, qubits = create_dicke_state(N, n, **_cal(seed))
    return dict(
        cz=sum(op.gate == cirq.CZ for op in circuit.all_operations()),
        moments=len(circuit),
        qubits=sorted([q.row, q.col] for q in qubits),
        repr_sha256=hashlib.sha256(repr(circuit).encode()).hexdigest(),
    )


def _key(seed, N, n):
    return f"seed{seed}/D{N},{n}"


@functools.lru_cache(maxsize=None)
def _golden():
    with open(GOLDEN) as fh:
        return json.load(fh)


@pytest.mark.parametrize("seed,N,n", CASES)
def test_small_n_unchanged(seed, N, n):
    assert snapshot(seed, N, n) == _golden()[_key(seed, N, n)]


if __name__ == "__main__":
    if "--regen" not in sys.argv:
        sys.exit(__doc__)
    table = {_key(*c): snapshot(*c) for c in CASES}
    with open(GOLDEN, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    print(f"wrote {len(table)} entries to {GOLDEN}")
