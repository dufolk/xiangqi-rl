import unittest

try:
    import numpy as np
except ImportError:
    np = None

from rl.vector_env import SubprocessVectorEnv


@unittest.skipIf(np is None, "NumPy is an optional training dependency")
class SubprocessVectorEnvTests(unittest.TestCase):
    def test_parallel_reset_step_and_individual_reset(self):
        with SubprocessVectorEnv(num_envs=2, max_steps=8) as envs:
            resets = envs.reset([101, 202])
            self.assertEqual(len(resets), 2)
            actions = [int(np.flatnonzero(info["action_mask"])[0]) for _, info in resets]
            results = envs.step(actions)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(result[4]["legal_action_count"] > 0 for result in results))
            renewed = envs.reset_at([1], [303])
            self.assertEqual(len(renewed), 1)
            self.assertEqual(renewed[0][0].shape, (16, 10, 9))


if __name__ == "__main__":
    unittest.main()
