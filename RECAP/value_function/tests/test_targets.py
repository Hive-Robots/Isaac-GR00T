import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from value_function.targets import (
    compute_episode_returns,
    normalize_returns,
    normalized_to_bin,
    parse_success,
    terminal_reward,
)


class TestTargets(unittest.TestCase):
    def test_success_failure_returns(self):
        success_returns = compute_episode_returns(length=5, term_reward=0.0)
        failure_returns = compute_episode_returns(length=5, term_reward=-10.0)

        np.testing.assert_allclose(success_returns, np.array([-4, -3, -2, -1, 0], dtype=np.float32))
        np.testing.assert_allclose(failure_returns, np.array([-14, -13, -12, -11, -10], dtype=np.float32))

    def test_parse_success_and_terminal_reward(self):
        success_meta = {"success": True}
        failure_meta = {"trajectory_type": "failed"}

        self.assertTrue(parse_success(success_meta, "success"))
        self.assertFalse(parse_success(failure_meta, "success"))
        self.assertEqual(terminal_reward(success_meta, True, "", 200.0), 0.0)
        self.assertEqual(terminal_reward(failure_meta, False, "", 200.0), -200.0)

    def test_binning_boundaries(self):
        values = np.array([-2.0, -1.0, -0.5, 0.0, 1.0], dtype=np.float32)
        normalized = np.clip(values, -1.0, 0.0)
        bins = normalized_to_bin(normalized, num_bins=201)
        self.assertTrue(np.all(bins >= 0))
        self.assertTrue(np.all(bins <= 200))
        self.assertEqual(int(bins[1]), 0)
        self.assertEqual(int(bins[3]), 200)

    def test_normalize_returns(self):
        returns = np.array([-200.0, -100.0, 0.0], dtype=np.float32)
        normalized = normalize_returns(returns, scale=200.0)
        np.testing.assert_allclose(normalized, np.array([-1.0, -0.5, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
