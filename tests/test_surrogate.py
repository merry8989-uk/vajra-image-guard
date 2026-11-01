"""Checks for the numpy reference implementation. Runs with the standard library:

    python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research"))

import surrogate_pgd as sp  # noqa: E402


class GradientTest(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        w, h = 20, 16
        basis = sp.make_basis(w, h, 11, 3, (1, 2, 4))
        rng = np.random.default_rng(3)
        x = 0.2 + 0.6 * rng.random((h, w))
        g = sp.gradient(x, basis, "suppress")
        eps = 1e-6
        for (yy, xx) in [(2, 3), (8, 9), (14, 1)]:
            xp, xm = x.copy(), x.copy()
            xp[yy, xx] += eps
            xm[yy, xx] -= eps
            numeric = (sp.loss(xp, basis) - sp.loss(xm, basis)) / (2 * eps)
            self.assertLess(abs(numeric - g[yy, xx]),
                            max(1e-6, 1e-4 * max(abs(numeric), abs(g[yy, xx]))))

    def test_displace_objective_is_the_negated_loss(self):
        basis = sp.make_basis(12, 12, 2, 2, (2,))
        x = np.full((12, 12), 0.5)
        self.assertAlmostEqual(sp.loss(x, basis, "displace"), -sp.loss(x, basis, "suppress"))


class BudgetTest(unittest.TestCase):
    def test_perturbation_within_ball(self):
        w, h = 24, 24
        basis = sp.make_basis(w, h, 5, 3, (2, 4, 8))
        x = np.random.default_rng(1).random((h, w))
        mask = np.ones((h, w))
        d = sp.solve_channel(x, basis, 0.02, 10, mask, "suppress", 2)
        self.assertLessEqual(float(np.abs(d).max()), 0.02 + 1e-9)

    def test_mask_limits_flat_regions(self):
        w, h = 32, 32
        basis = sp.make_basis(w, h, 9, 3, (2, 4, 8))
        x = np.full((h, w), 0.5)
        mask = np.full((h, w), 0.2)
        d = sp.solve_channel(x, basis, 0.05, 10, mask, "suppress", 4)
        self.assertLessEqual(float(np.abs(d).max()), 0.05 * 0.2 + 1e-9)

    def test_channel_budgets_bound_rgb(self):
        b = sp.channel_budgets(0.05, 0.5)
        worst = max(b["luma"] + 1.402 * b["cr"],
                    b["luma"] + 0.344136 * b["cb"] + 0.714136 * b["cr"],
                    b["luma"] + 1.772 * b["cb"])
        self.assertLessEqual(worst, 0.05 + 1e-9)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.img = (rng.random((64, 48, 3)) * 255).astype(np.uint8)

    def test_deterministic_per_seed(self):
        a, _ = sp.protect(self.img, seed=3, iterations=6)
        b, _ = sp.protect(self.img, seed=3, iterations=6)
        np.testing.assert_array_equal(a, b)

    def test_seed_changes_output(self):
        a, _ = sp.protect(self.img, seed=3, iterations=6)
        c, _ = sp.protect(self.img, seed=4, iterations=6)
        self.assertFalse(np.array_equal(a, c))

    def test_budget_respected_in_8_bit_space(self):
        _, stats = sp.protect(self.img, iterations=6)
        self.assertLessEqual(stats["max_delta"], 7)
        self.assertGreaterEqual(stats["max_delta"], 1)

    def test_quality_floor(self):
        _, stats = sp.protect(self.img, iterations=6)
        self.assertGreater(stats["psnr"], 38.0)

    def test_self_test_suite_passes(self):
        self.assertEqual(sp.self_test(), 0)


if __name__ == "__main__":
    unittest.main()
