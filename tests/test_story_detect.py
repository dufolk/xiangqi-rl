import unittest

from cinema.detect import (
    CAPTURES, MATES, OPENINGS, analyze_move, classify_attack_pattern, detect_opening,
)
from game_engine import (
    ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, KINDS, PAWN, RED, ROOK, Move, Piece, XiangqiGame,
)


def opening_after(*moves):
    game = XiangqiGame()
    key = None
    for start, end in moves:
        piece = game.piece_at(start)
        played = game.push(Move(start, end))
        key = detect_opening(game, played, piece)
    return key


def position(pieces, turn=RED):
    game = XiangqiGame()
    game.clear()
    for (row, col), piece in pieces.items():
        game.set_piece(row, col, piece)
    game.turn = turn
    return game


def pattern_after(pieces, start, end):
    game = position(pieces)
    piece = game.piece_at(start)
    played = game.push(Move(start, end))
    assert game.is_in_check(BLACK) or game.winner == RED
    return classify_attack_pattern(game, played, piece), game


BK, RK = Piece(BLACK, KING), Piece(RED, KING)
R = lambda kind: Piece(RED, kind)  # noqa: E731
B = lambda kind: Piece(BLACK, kind)  # noqa: E731


class OpeningDetectionTests(unittest.TestCase):
    def test_red_first_moves(self):
        cases = {
            ((7, 7), (7, 4)): "center_cannon",
            ((7, 7), (7, 3)): "cross_palace_cannon",
            ((7, 7), (7, 5)): "advisor_cannon",
            ((7, 7), (7, 6)): "pawn_base_cannon",
            ((7, 7), (8, 7)): "retreat_cannon",
            ((7, 7), (7, 8)): "edge_cannon",
            ((7, 7), (5, 7)): "river_cannon",
            ((7, 7), (6, 7)): "hook_cannon",
            ((6, 6), (5, 6)): "immortal_pawn",
            ((6, 0), (5, 0)): "edge_pawn",
            ((9, 6), (7, 4)): "flying_elephant",
            ((9, 6), (7, 8)): "edge_elephant",
            ((9, 7), (7, 6)): "horse_opening",
            ((9, 7), (7, 8)): "edge_horse",
        }
        for move, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(opening_after(move), expected)

    def test_black_replies(self):
        self.assertEqual(opening_after(((7, 7), (7, 4)), ((2, 7), (2, 4))), "same_cannons")
        self.assertEqual(opening_after(((7, 7), (7, 4)), ((2, 1), (2, 4))), "crossed_cannons")
        self.assertEqual(opening_after(((6, 6), (5, 6)), ((2, 7), (2, 6))), "pawn_base_cannon")
        self.assertEqual(
            opening_after(((7, 7), (7, 4)), ((0, 7), (2, 6)), ((9, 7), (7, 6)), ((0, 1), (2, 2))),
            "screen_horses",
        )

    def test_later_moves_are_quiet(self):
        self.assertIsNone(opening_after(((7, 7), (7, 4)), ((2, 7), (2, 4)), ((7, 1), (7, 3))))

    def test_every_opening_key_has_text(self):
        for key, (title, subtitle) in OPENINGS.items():
            self.assertTrue(title and subtitle, key)


class CaptureTableTests(unittest.TestCase):
    def test_every_attacker_victim_pair_is_covered(self):
        for attacker in KINDS:
            for victim in KINDS:
                title, attacker_line, victim_line = CAPTURES[attacker + victim]
                self.assertTrue(title and attacker_line and victim_line)

    def test_capture_event_in_analysis(self):
        game = position({(9, 3): RK, (0, 4): BK, (5, 0): R(ROOK), (2, 0): B(HORSE)})
        piece = game.piece_at((5, 0))
        played = game.push(Move((5, 0), (2, 0)))
        events = analyze_move(game, played, piece)
        self.assertEqual([(e.kind, e.key) for e in events], [("capture", "RH")])
        self.assertEqual(events[0].title, "铁轮断蹄")


class MatePatternTests(unittest.TestCase):
    def test_double_rook_mate(self):
        key, game = pattern_after({(9, 3): RK, (0, 4): BK, (1, 8): R(ROOK), (5, 0): R(ROOK)}, (5, 0), (0, 0))
        self.assertEqual(key, "double_rook")
        self.assertEqual(game.winner, RED)
        events = analyze_move(game, game.last_move, R(ROOK))
        self.assertEqual((events[-1].kind, events[-1].title), ("mate", "双车错"))

    def test_horse_patterns(self):
        self.assertEqual(pattern_after({(9, 3): RK, (0, 4): BK, (3, 3): R(HORSE)}, (3, 3), (1, 2))[0], "wocao_horse")
        self.assertEqual(pattern_after({(9, 3): RK, (0, 4): BK, (4, 4): R(HORSE)}, (4, 4), (2, 3))[0], "corner_horse")
        self.assertEqual(pattern_after({(9, 3): RK, (1, 4): BK, (4, 3): R(HORSE)}, (4, 3), (2, 2))[0], "fishing_horse")

    def test_cannon_patterns(self):
        base = {(9, 3): RK, (0, 4): BK}
        self.assertEqual(pattern_after({**base, (3, 4): R(HORSE), (6, 0): R(CANNON)}, (6, 0), (6, 4))[0], "horse_cannon")
        self.assertEqual(pattern_after({**base, (3, 4): R(CANNON), (6, 0): R(CANNON)}, (6, 0), (6, 4))[0], "double_cannon")
        self.assertEqual(
            pattern_after({**base, (3, 4): B(PAWN), (5, 4): R(CANNON), (5, 0): R(CANNON)}, (5, 0), (0, 0))[0],
            "heaven_earth",
        )
        self.assertEqual(
            pattern_after({**base, (3, 4): B(PAWN), (5, 4): R(CANNON), (5, 3): R(ROOK)}, (5, 3), (0, 3))[0],
            "iron_bolt",
        )

    def test_heart_strike(self):
        key, game = pattern_after({(9, 3): RK, (0, 4): BK, (1, 4): B(ADVISOR), (5, 4): R(ROOK)}, (5, 4), (1, 4))
        self.assertEqual(key, "heart_strike")

    def test_facing_kings(self):
        key, _ = pattern_after({(9, 4): RK, (0, 3): BK, (1, 3): B(ELEPHANT), (1, 2): R(PAWN)}, (1, 2), (0, 2))
        self.assertEqual(key, "facing_kings")

    def test_sea_moon(self):
        key, _ = pattern_after({(9, 3): RK, (6, 3): B(PAWN), (1, 4): BK, (0, 0): R(ROOK)}, (0, 0), (0, 4))
        self.assertEqual(key, "sea_moon")

    def test_two_ghosts(self):
        key, _ = pattern_after({(9, 3): RK, (0, 4): BK, (1, 5): R(PAWN), (1, 3): R(PAWN)}, (1, 3), (0, 3))
        self.assertEqual(key, "two_ghosts")

    def test_every_mate_key_has_text(self):
        for key, (title, subtitle) in MATES.items():
            self.assertTrue(title and subtitle, key)


if __name__ == "__main__":
    unittest.main()
