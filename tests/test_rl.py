import unittest

try:
    import numpy as np
except ImportError:
    np = None

from game_engine import BLACK, RED
from rl.environment import ACTION_SIZE, XiangqiEnv, decode_action, encode_move, rotate_square


@unittest.skipIf(np is None, "NumPy is an optional training dependency")
class XiangqiEnvironmentTests(unittest.TestCase):
    def test_reset_observation_and_mask(self):
        env = XiangqiEnv()
        observation, info = env.reset()
        self.assertEqual(observation.shape, (16, 10, 9))
        self.assertEqual(info["action_mask"].shape, (ACTION_SIZE,))
        self.assertGreater(int(info["action_mask"].sum()), 0)
        self.assertEqual(float(observation[14].sum()), 0.0)
        self.assertEqual(float(observation[15].sum()), 0.0)

    def test_red_initial_position_uses_own_and_opponent_planes(self):
        env = XiangqiEnv()
        observation, _ = env.reset()
        # Own rook plane: red rooks are on the bottom corners.
        self.assertEqual(observation[4, 9, 0], 1.0)
        self.assertEqual(observation[4, 9, 8], 1.0)
        # Opponent rook plane: black rooks are on the top corners.
        self.assertEqual(observation[11, 0, 0], 1.0)
        self.assertEqual(observation[11, 0, 8], 1.0)

    def test_black_turn_is_rotated_and_piece_channels_are_swapped(self):
        env = XiangqiEnv()
        env.reset()
        first_action = int(np.flatnonzero(env.action_mask())[0])
        env.step(first_action)
        self.assertEqual(env.game.turn, BLACK)
        observation = env.observe()
        # Raw black rook (0, 0) becomes canonical own rook (9, 8).
        self.assertEqual(observation[4, 9, 8], 1.0)
        # Raw red rook (9, 0) becomes canonical opponent rook (0, 8).
        self.assertEqual(observation[11, 0, 8], 1.0)

    def test_canonical_action_round_trip_for_black(self):
        env = XiangqiEnv()
        env.reset()
        env.step(int(np.flatnonzero(env.action_mask())[0]))
        legal_move = env.game.legal_moves()[0]
        action = encode_move(legal_move, BLACK)
        canonical_start, canonical_end = decode_action(action)
        self.assertEqual(rotate_square(canonical_start), legal_move.start)
        self.assertEqual(rotate_square(canonical_end), legal_move.end)
        env.step(action)
        self.assertEqual(env.game.last_move.start, legal_move.start)
        self.assertEqual(env.game.last_move.end, legal_move.end)

    def test_legal_action_step_and_terminal_only_reward(self):
        env = XiangqiEnv()
        _, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        result = env.step(action)
        self.assertEqual(env.game.turn, BLACK)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.observation.shape, (16, 10, 9))
        self.assertIn("move", result.info)
        self.assertIn("capture", result.info)
        self.assertIn("gives_check", result.info)
        self.assertGreater(result.info["legal_action_count"], 0)

    def test_illegal_action_is_rejected(self):
        env = XiangqiEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
