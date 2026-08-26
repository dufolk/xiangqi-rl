import importlib.util
import itertools
import queue
import unittest

TRAIN_DEPS_AVAILABLE = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"))


@unittest.skipUnless(TRAIN_DEPS_AVAILABLE, "optional training dependencies are not installed")
class MAPPOEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global np, torch
        global ALL_GEMS, GEM_KEYS, Card, Game, observation
        global ACTION_SPACE, OBS_DIM, SLOT_TAKE, action_to_slot, build_action_mask
        global encode_observation, slot_to_action_id
        global SplendorMAEnv, ActorCritic, assign_terminal_rewards
        global compute_grouped_gae, concatenate_rollouts, DistributedCollector

        import numpy as np
        import torch
        from game_ai import observation
        from game_engine import ALL_GEMS, GEM_KEYS, Card, Game
        from rl.mappo.buffer import compute_grouped_gae, concatenate_rollouts
        from rl.mappo.collector import DistributedCollector, assign_terminal_rewards
        from rl.mappo.encoders import (
            ACTION_SPACE, OBS_DIM, SLOT_TAKE, action_to_slot,
            build_action_mask, encode_observation, slot_to_action_id,
        )
        from rl.mappo.envs import SplendorMAEnv
        from rl.mappo.networks import ActorCritic

    def test_observation_is_278_and_uses_engine_color_names(self):
        game = Game()
        game.bank.update({"white": 1, "blue": 2, "green": 3, "red": 4, "black": 5, "gold": 6})
        game.players[0].reserved.append(game.decks[1].pop())
        encoded = encode_observation(observation(game))
        self.assertEqual(encoded.shape, (278,))
        np.testing.assert_array_equal(encoded[:6], np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32))
        self.assertAlmostEqual(float(encoded[34]), 1 / 3)

    def test_table_tiers_keep_four_stable_slots(self):
        game = Game()
        game.table[1].pop()
        encoded = encode_observation(observation(game))
        table_start = 80
        empty_tier_one_slot = encoded[table_start + 3 * 14: table_start + 4 * 14]
        tier_two_first_slot = encoded[table_start + 4 * 14: table_start + 5 * 14]
        self.assertTrue(np.all(empty_tier_one_slot == 0))
        self.assertGreater(float(np.abs(tier_two_first_slot).sum()), 0)

    def test_action_space_has_71_unique_reversible_slots(self):
        actions = []
        for tier in (1, 2, 3):
            for position in range(4):
                actions.append(f"buy:table:{tier}:{position}")
                actions.append(f"reserve:{tier}:{position}")
        actions += [f"reserve_deck:{tier}" for tier in (1, 2, 3)]
        actions += [f"buy:reserved:0:{index}" for index in range(3)]
        colored = GEM_KEYS
        for size in (1, 2, 3):
            actions += ["take:" + ",".join(combo) for combo in itertools.combinations(colored, size)]
        actions += [f"take:{gem},{gem}" for gem in colored]
        actions += [f"noble:{index}" for index in range(5)]
        actions += [f"return:{gem}" for gem in ALL_GEMS]

        slots = [action_to_slot(action) for action in actions]
        self.assertEqual(ACTION_SPACE, 71)
        self.assertEqual(len(actions), 71)
        self.assertEqual(set(slots), set(range(71)))
        legal = [{"id": action} for action in actions]
        self.assertEqual(int(build_action_mask(legal).sum()), 71)
        for action, slot in zip(actions, slots):
            self.assertEqual(slot_to_action_id(slot, legal), action)

    def test_environment_returns_same_instant_global_state_and_team_rewards(self):
        env = SplendorMAEnv(team_mode=True, seed=3)
        obs, global_state, mask, _ = env.reset()
        np.testing.assert_array_equal(obs, global_state)
        env.game.players[0].cards.append(Card(3, "white", 15, {gem: 0 for gem in GEM_KEYS}))
        env.game.players[2].cards.append(Card(3, "blue", 15, {gem: 0 for gem in GEM_KEYS}))
        env.game.turn = 3
        obs, global_state, mask, info = env._observe()
        legal = info["legal"]
        take = next(action for action in legal if action["id"].startswith("take:"))
        next_obs, next_global, _, _, terminated, next_info = env.step(action_to_slot(take["id"]))
        self.assertTrue(terminated)
        np.testing.assert_array_equal(next_obs, next_global)
        self.assertEqual(next_info["terminal_rewards"], [10.0, -10.0, 10.0, -10.0])

    def test_terminal_rewards_reach_each_teammate_trajectory(self):
        rewards = [0.0, 0.0, 0.0, 0.0]
        agent_ids = [0, 1, 2, 3]
        episode_ids = [9, 9, 9, 9]
        assign_terminal_rewards(rewards, agent_ids, episode_ids, 9, [10, -10, 10, -10])
        self.assertEqual(rewards, [10.0, -10.0, 10.0, -10.0])

    def test_grouped_gae_does_not_cross_workers_or_episodes_and_bootstraps(self):
        first = {
            "obs": np.zeros((1, OBS_DIM), dtype=np.float32),
            "global_state": np.zeros((1, OBS_DIM), dtype=np.float32),
            "agent_id": np.asarray([0]), "action": np.asarray([0]),
            "logp": np.asarray([0.0], dtype=np.float32),
            "reward": np.asarray([0.0], dtype=np.float32),
            "mask": np.ones((1, ACTION_SPACE), dtype=np.float32),
            "value": np.asarray([1.0], dtype=np.float32),
            "episode_id": np.asarray([0]),
            "bootstrap_values": {"0": [2.0, 0.0, 0.0, 0.0]},
        }
        second = dict(first)
        second = {key: value.copy() if hasattr(value, "copy") else value for key, value in second.items()}
        second["reward"] = np.asarray([100.0], dtype=np.float32)
        second["bootstrap_values"] = {"0": [0.0, 0.0, 0.0, 0.0]}
        merged = concatenate_rollouts([first, second])
        self.assertAlmostEqual(float(merged["advantages"][0]), 0.98, places=5)
        self.assertAlmostEqual(float(merged["advantages"][1]), 99.0, places=5)

        advantages, _ = compute_grouped_gae(
            np.asarray([1.0, 50.0]), np.asarray([0.0, 0.0]),
            np.asarray([0, 0]), np.asarray([1, 2]), {(1, 0): 0.0, (2, 0): 0.0},
        )
        self.assertAlmostEqual(float(advantages[0]), 1.0)
        self.assertAlmostEqual(float(advantages[1]), 50.0)

    def test_critic_uses_278_current_state_and_four_value_heads(self):
        actor_critic = ActorCritic(OBS_DIM, ACTION_SPACE, 4, OBS_DIM)
        values = actor_critic.critic(torch.zeros((2, OBS_DIM)))
        self.assertEqual(tuple(values.shape), (2, 4))
        collector = DistributedCollector(n_workers=2)
        self.assertIsNot(collector.weight_queues[0], collector.weight_queues[1])

    def test_worker_loads_initial_weights_and_bootstraps_horizon(self):
        actor_critic = ActorCritic(OBS_DIM, ACTION_SPACE, 4, OBS_DIM)
        collector = DistributedCollector(n_workers=1, horizon=4, seed=19)
        try:
            collector.start(actor_critic.state_dict())
            payload = collector.collect(timeout=30)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["policy_version"], 0)
            self.assertEqual(payload["global_state"].shape[1], OBS_DIM)
            np.testing.assert_array_equal(payload["obs"], payload["global_state"])
            active_episode = str(int(payload["episode_id"][-1]))
            self.assertEqual(len(payload["bootstrap_values"][active_episode]), 4)
        finally:
            collector.stop()

    def test_historical_opponents_only_emit_current_policy_trajectories(self):
        actor_critic = ActorCritic(OBS_DIM, ACTION_SPACE, 4, OBS_DIM)
        state_dict = actor_critic.state_dict()
        for team_mode, expected_seats in ((False, {0}), (True, {0, 2})):
            with self.subTest(team_mode=team_mode):
                collector = DistributedCollector(
                    n_workers=1, horizon=4, seed=31, team_mode=team_mode,
                    history_opponent_probability=1.0,
                )
                try:
                    collector.start(state_dict, opponent_pool=[state_dict])
                    payload = collector.collect(timeout=30)
                    self.assertIsNotNone(payload)
                    self.assertEqual(set(payload["agent_id"].tolist()), expected_seats)
                finally:
                    collector.stop()

    def test_elo_convergence_requires_a_stable_full_window(self):
        from rl.mappo.history import elo_has_converged

        self.assertFalse(elo_has_converged([1000.0, 1002.0], window=3, threshold=25.0))
        self.assertFalse(elo_has_converged([1000.0, 1015.0, 1060.0], window=3, threshold=25.0))
        self.assertTrue(elo_has_converged([1000.0, 1003.0, 998.0, 1001.0], window=3, threshold=25.0))
        self.assertFalse(elo_has_converged([1000.0, 1001.0, 1002.0], window=0, threshold=25.0))

    def test_train_loop_honors_stop_callback(self):
        from rl.mappo.algo import MAPPO

        mappo = MAPPO(OBS_DIM, ACTION_SPACE, 4, OBS_DIM, device=torch.device("cpu"))
        collector = DistributedCollector(n_workers=1, horizon=4, seed=47)
        callback_updates = []

        def stop_after_first_update(_state_dict, collected, update):
            callback_updates.append((collected, update))
            return {"stop": True}

        collector.train_loop(
            mappo, total_timesteps=10_000, rollouts_per_update=1, n_epochs=1,
            on_update=stop_after_first_update,
        )
        self.assertEqual(len(callback_updates), 1)
        self.assertEqual(callback_updates[0][1], 1)


if __name__ == "__main__":
    unittest.main()
