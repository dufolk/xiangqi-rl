import unittest

try:
    import numpy as np
except ImportError:
    np = None

from game_engine import RED
from rl.environment import ACTION_SIZE, XiangqiEnv, decode_action


@unittest.skipIf(np is None, "NumPy is an optional training dependency")
class XiangqiEnvironmentTests(unittest.TestCase):
    def test_reset_observation_and_mask(self):
        env = XiangqiEnv()
        observation, info = env.reset()
        self.assertEqual(observation.shape, (16, 10, 9))
        self.assertEqual(info["action_mask"].shape, (ACTION_SIZE,))
        self.assertGreater(int(info["action_mask"].sum()), 0)
        self.assertTrue((observation[14] == 1).all())

    def test_legal_action_step(self):
        env = XiangqiEnv()
        _, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        start, end = decode_action(action)
        result = env.step(action)
        self.assertEqual(env.game.turn, "black")
        self.assertEqual(env.game.last_move.start, start)
        self.assertEqual(env.game.last_move.end, end)
        self.assertEqual(result.observation.shape, (16, 10, 9))

    def test_illegal_action_is_rejected(self):
        env = XiangqiEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
