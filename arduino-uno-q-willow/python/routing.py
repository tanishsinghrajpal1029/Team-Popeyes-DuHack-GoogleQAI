"""Small, dependency-free routing model for the UNO Q exhibit.

The Willow topology is real. Edge errors are deterministic teaching values, not
Willow calibration. A path's last edge is the final CZ; preceding edges are
approximated as SWAPs costing three CZs each.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = json.loads((ROOT / "assets" / "topology.json").read_text())
NODES = {f"{row},{col}" for row, col in TOPOLOGY["qubits"]}
NEIGHBORS: dict[str, set[str]] = {node: set() for node in NODES}
for left, right in TOPOLOGY["couplers"]:
    a, b = f"{left[0]},{left[1]}", f"{right[0]},{right[1]}"
    NEIGHBORS[a].add(b)
    NEIGHBORS[b].add(a)

SCENARIOS = {
    "crossbar": {
        "name": "Across the chip",
        "start": "6,2",
        "end": "6,12",
        "weak": "6,7",
    },
    "upper": {
        "name": "Upper corridor",
        "start": "5,2",
        "end": "5,11",
        "weak": "5,7",
    },
    "diagonal": {
        "name": "Diagonal crossing",
        "start": "4,3",
        "end": "8,11",
        "weak": "6,7",
    },
}

# The onboard 8x13 matrix shows this central *window* of the 105-qubit graph.
MATRIX_ROW0 = 3
MATRIX_COL0 = 1
MATRIX_HEIGHT = 8
MATRIX_WIDTH = 13


def base_error(a: str, b: str) -> float:
    """Match the website's stable illustrative per-coupler error pattern."""
    token = "|".join(sorted((a, b)))
    value = 13
    for char in token:
        value = (value * 31 + ord(char)) % 9973
    return 0.0022 + (value % 29) * 0.00011


def _edge_error(a: str, b: str, weak: str, weak_error: float) -> float:
    return weak_error if weak in (a, b) else base_error(a, b)


def _path(start: str, end: str, weak: str, weak_error: float,
          blocked: bool, risk_aware: bool) -> list[str]:
    excluded = {weak} if blocked else set()
    if start in excluded or end in excluded:
        return []
    distances = {start: 0.0}
    previous: dict[str, str] = {}
    queue = [(0.0, start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current] + 1e-12:
            continue
        if current == end:
            break
        for nxt in sorted(NEIGHBORS[current]):
            if nxt in excluded:
                continue
            if risk_aware:
                # The terminal interaction costs one CZ; every earlier move
                # costs approximately one SWAP, or three CZs.
                multiplicity = 1 if nxt == end else 3
                cost = multiplicity * -math.log1p(
                    -_edge_error(current, nxt, weak, weak_error)
                )
            else:
                cost = 1.0
            candidate = distance + cost
            if candidate < distances.get(nxt, math.inf) - 1e-12:
                distances[nxt] = candidate
                previous[nxt] = current
                heapq.heappush(queue, (candidate, nxt))
    if end not in distances:
        return []
    result = [end]
    while result[0] != start:
        result.insert(0, previous[result[0]])
    return result


def _survival(path: list[str], weak: str, weak_error: float) -> float:
    if len(path) < 2:
        return 0.0
    log_survival = 0.0
    for index, (a, b) in enumerate(zip(path, path[1:])):
        multiplicity = 1 if index == len(path) - 2 else 3
        log_survival += multiplicity * math.log1p(
            -_edge_error(a, b, weak, weak_error)
        )
    return math.exp(log_survival)


def _mask(nodes: list[str] | set[str]) -> list[int]:
    rows = [0] * MATRIX_HEIGHT
    for node in nodes:
        row, col = map(int, node.split(","))
        y, x = row - MATRIX_ROW0, col - MATRIX_COL0
        if 0 <= y < MATRIX_HEIGHT and 0 <= x < MATRIX_WIDTH:
            rows[y] |= 1 << x
    return rows


def _pixel(node: str) -> int:
    row, col = map(int, node.split(","))
    y, x = row - MATRIX_ROW0, col - MATRIX_COL0
    return y * MATRIX_WIDTH + x if 0 <= y < 8 and 0 <= x < 13 else -1


def solve(scenario: str = "crossbar", weak_error_percent: float = 7.0,
          block_weak: bool = False) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    try:
        percent = float(weak_error_percent)
    except (TypeError, ValueError) as exc:
        raise ValueError("Weak-link error must be a number") from exc
    if not math.isfinite(percent) or not 0.2 <= percent <= 20:
        raise ValueError("Weak-link error must be between 0.2% and 20%")
    if not isinstance(block_weak, bool):
        raise ValueError("block_weak must be a boolean")

    item = SCENARIOS[scenario]
    start, end, weak = item["start"], item["end"], item["weak"]
    weak_error = percent / 100
    direct = _path(start, end, weak, weak_error, block_weak, False)
    aware = _path(start, end, weak, weak_error, block_weak, True)
    if not direct or not aware:
        raise ValueError("No route exists for this scenario")

    def describe(path: list[str]) -> dict:
        survival = _survival(path, weak, weak_error)
        return {
            "nodes": path,
            "hops": len(path) - 1,
            "swaps": max(0, len(path) - 2),
            "survival": survival,
            "survival_percent": round(survival * 100, 1),
        }

    direct_result, aware_result = describe(direct), describe(aware)
    return {
        "scenario": scenario,
        "scenario_name": item["name"],
        "start": start,
        "end": end,
        "weak": weak,
        "weak_error_percent": percent,
        "block_weak": block_weak,
        "direct": direct_result,
        "aware": aware_result,
        "gain_percentage_points": round(
            aware_result["survival_percent"] - direct_result["survival_percent"], 1
        ),
        "matrix": {
            "row_start": MATRIX_ROW0,
            "col_start": MATRIX_COL0,
            "nodes": _mask(NODES),
            "direct": _mask(direct),
            "aware": _mask(aware),
            "start_pixel": _pixel(start),
            "end_pixel": _pixel(end),
            "weak_pixel": _pixel(weak),
        },
    }
