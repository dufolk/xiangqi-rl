import unittest

from ai_player import SimpleAI
from game_engine import BLACK, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK, Move, Piece, XiangqiGame


class XiangqiEngineTests(unittest.TestCase):
    def test_standard_position_and_fen_round_trip(self):
        game = XiangqiGame()
        self.assertEqual(sum(piece is not None for row in game.board for piece in row), 32)
        restored = XiangqiGame.from_fen(game.to_fen())
        self.assertEqual(restored.to_fen(), game.to_fen())
        self.assertEqual(restored.validate_position(), [])

    def test_horse_leg_is_blocked(self):
        game = XiangqiGame()
        starts = {move.end for move in game.legal_moves_from((9, 1))}
        self.assertIn((7, 0), starts)
        self.assertIn((7, 2), starts)
        game.set_piece(8, 1, Piece(RED, PAWN))
        self.assertEqual(game.legal_moves_from((9, 1)), [])

    def test_cannon_requires_exactly_one_screen_to_capture(self):
        game = XiangqiGame()
        game.clear()
        game.set_piece(9, 4, Piece(RED, KING))
        game.set_piece(0, 3, Piece(BLACK, KING))
        game.set_piece(5, 0, Piece(RED, CANNON))
        game.set_piece(3, 0, Piece(RED, PAWN))
        game.set_piece(1, 0, Piece(BLACK, ROOK))
        ends = {move.end for move in game.legal_moves_from((5, 0))}
        self.assertIn((1, 0), ends)
        self.assertNotIn((2, 0), ends)

    def test_elephant_cannot_cross_river(self):
        game = XiangqiGame()
        game.clear()
        game.set_piece(9, 4, Piece(RED, KING))
        game.set_piece(0, 4, Piece(BLACK, KING))
        game.set_piece(5, 4, Piece(RED, PAWN))
        game.set_piece(5, 2, Piece(RED, ELEPHANT))
        ends = {move.end for move in game.legal_moves_from((5, 2))}
        self.assertNotIn((3, 4), ends)
        self.assertIn((7, 4), ends)

    def test_move_cannot_expose_flying_generals(self):
        game = XiangqiGame()
        game.clear()
        game.set_piece(9, 4, Piece(RED, KING))
        game.set_piece(0, 4, Piece(BLACK, KING))
        game.set_piece(5, 4, Piece(RED, ROOK))
        ends = {move.end for move in game.legal_moves_from((5, 4))}
        self.assertNotIn((5, 3), ends)
        self.assertNotIn((5, 5), ends)

    def test_push_and_undo_restore_position(self):
        game = XiangqiGame()
        before = game.to_fen()
        move = next(move for move in game.legal_moves_from((6, 0)) if move.end == (5, 0))
        game.push(move)
        self.assertEqual(game.turn, BLACK)
        self.assertTrue(game.undo())
        self.assertEqual(game.to_fen(), before)

    def test_simple_ai_returns_legal_move(self):
        game = XiangqiGame()
        move = SimpleAI(depth=1, seed=1).choose_move(game)
        self.assertIsNotNone(move)
        self.assertIn(move, game.legal_moves())


if __name__ == "__main__":
    unittest.main()
