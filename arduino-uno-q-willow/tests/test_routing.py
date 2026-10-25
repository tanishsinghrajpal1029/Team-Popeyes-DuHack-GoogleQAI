import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import routing  # noqa: E402
from bonus import DEPTHS, select_depths  # noqa: E402


class RoutingTests(unittest.TestCase):
    def test_topology_and_scenarios(self):
        self.assertEqual(len(routing.TOPOLOGY["qubits"]), 105)
        self.assertEqual(len(routing.TOPOLOGY["couplers"]), 182)
        for name in routing.SCENARIOS:
            with self.subTest(name=name):
                result = routing.solve(name, 12, False)
                self.assertEqual(result["direct"]["nodes"][0], result["start"])
                self.assertEqual(result["aware"]["nodes"][-1], result["end"])
                for route in (result["direct"], result["aware"]):
                    for a, b in zip(route["nodes"], route["nodes"][1:]):
                        self.assertIn(b, routing.NEIGHBORS[a])
                self.assertGreaterEqual(result["aware"]["survival"], result["direct"]["survival"] - 1e-10)
                self.assertTrue(all(0 <= mask < 8192 for rows in result["matrix"].values() if isinstance(rows, list) for mask in rows))
                self.assertTrue(all(0 <= result["matrix"][key] < 104 for key in ("start_pixel", "end_pixel", "weak_pixel")))

    def test_weak_qubit_changes_route(self):
        normal = routing.solve("crossbar", 12, False)
        self.assertIn(normal["weak"], normal["direct"]["nodes"])
        self.assertNotIn(normal["weak"], normal["aware"]["nodes"])
        self.assertGreater(normal["gain_percentage_points"], 20)
        blocked = routing.solve("crossbar", 12, True)
        self.assertNotIn(blocked["weak"], blocked["direct"]["nodes"])
        self.assertNotIn(blocked["weak"], blocked["aware"]["nodes"])

    def test_invalid_requests(self):
        for kwargs in (
            {"scenario": "missing"},
            {"weak_error_percent": float("nan")},
            {"weak_error_percent": 21},
            {"block_weak": "false"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                routing.solve(**kwargs)


class BonusTests(unittest.TestCase):
    def test_all_checked_bonus_pairs_select_the_shallower_depth(self):
        self.assertEqual(len(DEPTHS), 56)
        for N, n in DEPTHS:
            with self.subTest(N=N, n=n):
                result = select_depths(N, n)
                self.assertEqual(result["selected"], min(result["candidate_a"], result["candidate_b"]))
        self.assertEqual(select_depths(9, 4)["main_choice"], "crossover")
        self.assertEqual(select_depths(10, 3)["main_choice"], "portfolio")

    def test_bonus_range_is_explicit(self):
        for pair in ((1, 1), (9, 5), (16, 1), (9.0, 4), (True, 1)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                select_depths(*pair)


if __name__ == "__main__":
    unittest.main()
