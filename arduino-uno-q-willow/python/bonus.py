"""Checked bonus depths for the UNO Q exhibit; no quantum simulation on board."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "assets" / "bonus-depths.json").read_text())
DEPTHS = {(row["N"], row["n"]): row for row in DATA["cases"]}


def select_depths(N: int, n: int) -> dict:
    if isinstance(N, bool) or isinstance(n, bool) or not isinstance(N, int) or not isinstance(n, int):
        raise ValueError("N and n must be integers")
    row = DEPTHS.get((N, n))
    if row is None:
        raise ValueError("Bonus range: 2 ≤ N ≤ 15 and 1 ≤ n ≤ floor(N/2)")
    a, b = row["candidate_a"], row["candidate_b"]
    return {
        "N": N,
        "n": n,
        "candidate_a": a,
        "candidate_b": b,
        "selected": min(a, b),
        "winner": "tie" if a == b else ("a" if a < b else "b"),
        "main_choice": "crossover" if (N, n) in {(7, 3), (9, 4)} else "portfolio",
    }
