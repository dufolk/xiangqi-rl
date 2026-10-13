import unittest

from ai_player import (
    PikafishAI, SimpleAI, _parse_info, find_pikafish, move_to_uci, position_command, square_to_uci, uci_to_squares,
)
from game_engine import RED, Move, XiangqiGame

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class UciConversionTests(unittest.TestCase):
    def test_squares_use_red_bottom_left_origin(self):
        self.assertEqual(square_to_uci((9, 0)), "a0")
        self.assertEqual(square_to_uci((0, 8)), "i9")
        self.assertEqual(uci_to_squares("h2e2"), ((7, 7), (7, 4)))
        for row in range(10):
            for col in range(9):
                text = square_to_uci((row, col)) + square_to_uci((9 - row, 8 - col))
                self.assertEqual(uci_to_squares(text), ((row, col), (9 - row, 8 - col)))

    def test_rejects_malformed_moves(self):
        for text in ("", "j0a0", "a0a", "a0aa", "(none)"):
            with self.assertRaises(ValueError):
                uci_to_squares(text)

    def test_position_replays_history_from_root(self):
        game = XiangqiGame()
        for start, end in (((7, 7), (7, 4)), ((0, 7), (2, 6)), ((9, 7), (7, 6))):
            game.push(Move(start, end))
        self.assertEqual(position_command(game), f"position fen {START} - - 0 1 moves h2e2 h9g7 h0g2")
        self.assertEqual(move_to_uci(game.last_move), "h0g2")

    def test_position_falls_back_to_fen_after_editing(self):
        game = XiangqiGame()
        game.push(Move((7, 7), (7, 4)))
        game.set_piece(0, 0, None)
        self.assertEqual(position_command(game), f"position fen {game.to_fen()} - - 0 1")
        self.assertEqual(position_command(XiangqiGame()), f"position fen {START} - - 0 1")

    def test_parse_info(self):
        line = "info depth 12 seldepth 18 multipv 2 score cp -35 nodes 1 nps 1 time 1 pv b0c2 h9g7"
        self.assertEqual(_parse_info(line), (2, 12, -35, None, "b0c2"))
        mate = _parse_info("info depth 3 score mate -2 nodes 9 pv e1e0")
        self.assertEqual(mate[:2], (1, 3))
        self.assertLess(mate[2], -20_000)
        self.assertEqual(mate[3], -2)
        self.assertIsNone(_parse_info("info depth 5 score cp 10 lowerbound nodes 3 pv a0a1"))
        self.assertIsNone(_parse_info("info string NNUE evaluation using pikafish.nnue"))

    def test_sampling_stays_inside_window(self):
        ai = PikafishAI(path="/nonexistent", depth=1, seed=7)
        lines = {1: (4, 50, None, "a0a1"), 2: (4, 20, None, "b0c2"), 3: (4, -900, None, "i0i1"), 4: (3, 900, None, "h2e2")}
        picks = {ai._sample(lines, ai.level) for _ in range(200)}
        self.assertEqual(picks, {"a0a1", "b0c2"})

    def test_missing_engine_falls_back(self):
        ai = PikafishAI(path=None, depth=4, fallback=SimpleAI(depth=1, seed=1))
        ai.path = None
        move = ai.choose_move(XiangqiGame())
        self.assertIn(move, XiangqiGame().legal_moves())
        self.assertIn("Pikafish", ai.load_error)
        self.assertEqual(ai.cycle_level().name, "大师")
        self.assertEqual(ai.cycle_level().name, "萌新")


@unittest.skipIf(find_pikafish() is None, "Pikafish not installed (python tools/fetch_pikafish.py)")
class PikafishEngineTests(unittest.TestCase):
    def setUp(self):
        self.ai = PikafishAI(depth=3, threads=1, hash_mb=16)

    def tearDown(self):
        self.ai.close()

    def test_plays_legal_moves_at_every_level(self):
        game = XiangqiGame()
        for level in range(1, 4):
            self.ai.depth = level
            move = self.ai.choose_move(game.clone(with_history=True))
            self.assertIn(move, game.legal_moves())
            game.push(move)
        self.assertIsNone(self.ai.load_error)
        self.assertIsNotNone(self.ai.last_eval)

    def test_finds_mate_in_one(self):
        game = XiangqiGame.from_fen("3k5/R8/8R/9/9/9/9/9/9/4K4 w")
        move = self.ai.choose_move(game)
        game.push(move)
        self.assertEqual(game.winner, RED)
        self.assertEqual(self.ai.last_eval.mate, 1)

    def test_beats_alpha_beta_baseline(self):
        game = XiangqiGame()
        baseline = SimpleAI(depth=1, seed=3)
        self.ai.depth = 2
        while game.winner is None and game.halfmove < 160:
            player = self.ai if game.turn == RED else baseline
            game.push(player.choose_move(game.clone(with_history=True)))
        self.assertEqual(game.winner, RED)


if __name__ == "__main__":
    unittest.main()
