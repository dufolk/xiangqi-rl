import unittest

from ai_player import GenerativeAIPlayer
from game_ai import legal_actions, observation
from game_engine import Card, GEM_KEYS, Game
from rl.environment import SplendorEnv
from state_codec import game_from_data, game_to_data


class GameEngineTests(unittest.TestCase):
    def test_standard_deck_sizes(self):
        game = Game(rng=__import__("random").Random(1))
        self.assertEqual({tier: len(game.decks[tier]) + len(game.table[tier]) for tier in (1, 2, 3)}, {1: 40, 2: 30, 3: 20})

    def test_team_mode_uses_alternating_seats_and_thirty_points(self):
        game = Game(team_mode=True)
        self.assertEqual(game.target_score, 30)
        game.players[0].cards.append(Card(3, "white", 16, {gem: 0 for gem in GEM_KEYS}))
        game.players[2].cards.append(Card(3, "blue", 14, {gem: 0 for gem in GEM_KEYS}))
        self.assertEqual(game.team_score(0), 30)

    def test_team_heuristic_is_deterministic_and_knows_teammates(self):
        game = Game(team_mode=True)
        game.turn = 1
        state = observation(game)
        actions = legal_actions(game)
        policy = GenerativeAIPlayer()
        self.assertEqual(state["players"][3]["relation"], "teammate")
        self.assertEqual(policy.local_choice(state, actions), policy.local_choice(state, actions))

    def test_network_projection_hides_other_reserved_cards(self):
        game = Game()
        game.players[0].reserved.append(game.decks[1].pop())
        game.players[1].reserved.append(game.decks[2].pop())
        payload = game_to_data(game, viewer_seat=1)
        self.assertEqual(payload["players"][0]["reserved"], [None])
        self.assertIsInstance(payload["players"][1]["reserved"][0], dict)
        restored = game_from_data(payload)
        self.assertIsNone(restored.players[0].reserved[0])

    def test_training_environment_seed_is_reproducible(self):
        first = SplendorEnv(team_mode=True, seed=7)
        second = SplendorEnv(team_mode=True, seed=7)
        first.reset()
        second.reset()
        self.assertEqual(
            [[card.cost for card in first.game.table[tier]] for tier in (1, 2, 3)],
            [[card.cost for card in second.game.table[tier]] for tier in (1, 2, 3)],
        )
        action = first.legal_actions()[0]["id"]
        result = first.step(action)
        self.assertIn("acting_team", result.info)


if __name__ == "__main__":
    unittest.main()
