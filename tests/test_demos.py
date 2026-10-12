import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cinema.demos import OPENING_MOVES, catalogue, find
from cinema.detect import (
    MATES, OPENINGS, SITUATIONS, TACTICS, analyze_move, material_balance, situation, tone_for,
)
from cinema.show import MoveShow
from cinema.view3d import BoardView
from game_engine import BLACK, RED, Move


def play(demo):
    game = demo.build()
    before = [row[:] for row in game.board]
    piece = game.piece_at(demo.move[0])
    played = game.push(Move(*demo.move))
    return game, before, piece, played, analyze_move(game, played, piece)


class DemoCatalogueTests(unittest.TestCase):
    def test_every_story_has_a_demo(self):
        keys = {(d.category, d.key) for d in catalogue()}
        for key in OPENINGS:
            self.assertIn(("opening", key), keys)
        for key in MATES:
            if key != "generic":
                self.assertIn(("mate", key), keys)
        for key in TACTICS:
            self.assertIn(("tactic", key), keys)
        self.assertEqual(set(OPENING_MOVES), set(OPENINGS))
        self.assertGreaterEqual(sum(d.category == "capture" for d in catalogue()), 30)

    def test_demos_trigger_their_event(self):
        for demo in catalogue():
            with self.subTest(demo=f"{demo.category}/{demo.key}"):
                game, _, _, played, events = play(demo)
                kinds = {(e.kind, e.key) for e in events}
                if demo.category == "opening":
                    self.assertIn(("opening", demo.key), kinds)
                elif demo.category == "capture":
                    self.assertIn(("capture", demo.key), kinds)
                elif demo.category == "tactic":
                    self.assertIn(("tactic", demo.key), kinds)
                elif demo.category == "mate":
                    self.assertEqual(game.winner, RED)
                    self.assertIn(("mate", demo.key), kinds)
                elif demo.category == "stalemate":
                    self.assertEqual(game.reason, "困毙")
                    self.assertIn(("stalemate", "stalemate"), kinds)

    def test_tone_demos_change_capture_mood(self):
        expected = {"dominant": ("dominant", "乘胜追击"), "desperate": ("desperate", "绝地反击"),
                    "comeback": ("desperate", "扳回一城")}
        for key, (tone, tag) in expected.items():
            with self.subTest(key=key):
                capture = next(e for e in play(find("tone", key))[4] if e.kind == "capture")
                self.assertEqual((capture.tone, capture.tag), (tone, tag))
        even = next(e for e in play(find("tone", "even"))[4] if e.kind == "capture")
        self.assertEqual(even.tone, "even")

    def test_check_win_names_the_prize(self):
        tactic = next(e for e in play(find("tactic", "check_win"))[4] if e.kind == "tactic")
        self.assertTrue(tactic.title.startswith("将军抽"))
        self.assertEqual(len(tactic.targets), 2)


class SituationTests(unittest.TestCase):
    def test_situation_labels(self):
        self.assertEqual(situation(0), "even")
        self.assertEqual(situation(3), "red_up")
        self.assertEqual(situation(-7), "black_big")
        self.assertEqual(set(SITUATIONS), {"red_big", "red_up", "even", "black_up", "black_big"})

    def test_tone_is_from_the_movers_view(self):
        self.assertEqual(tone_for(8, RED), "dominant")
        self.assertEqual(tone_for(8, BLACK), "desperate")
        self.assertEqual(tone_for(0.5, BLACK), "even")

    def test_standard_position_is_balanced(self):
        self.assertEqual(material_balance(find("opening", "center_cannon").build().board), 0)


class BoardShowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((16, 16))

    def test_every_demo_show_plays_to_the_end(self):
        for demo in catalogue():
            with self.subTest(demo=f"{demo.category}/{demo.key}"):
                game, before, piece, played, events = play(demo)
                for mode in ("full", "off"):
                    show = MoveShow(played, piece, before, game.board, events, mode, game.winner)
                    show.started_at = 0
                    self.assertGreater(show.duration, 0)
                    for k in range(7):
                        frame = show.frame(int(show.duration * k / 6))
                        self.assertGreaterEqual(frame.zoom, 1.0)
                    self.assertTrue(show.done(show.duration))
                    self.assertLess(abs(show.frame(show.duration).zoom - 1.0), 1e-6)

    def test_scenes_render_on_the_board(self):
        view = BoardView()
        canvas = pygame.Surface((1440, 900))
        for category, key in (("opening", "center_cannon"), ("tone", "desperate"), ("tactic", "fork"),
                              ("mate", "horse_cannon"), ("stalemate", "stalemate")):
            game, before, piece, played, events = play(find(category, key))
            show = MoveShow(played, piece, before, game.board, events, "full", game.winner)
            show.started_at = 0
            for flipped in (False, True):
                view.flipped = flipped
                for t in (0, show.duration // 3, show.duration * 2 // 3):
                    view.compose(canvas, game.board, show.frame(t), t, 0.5, (0.5, -0.5))

    def test_pick_round_trips_projection(self):
        view = BoardView()
        for flipped in (False, True):
            view.flipped = flipped
            for row in range(10):
                for col in range(9):
                    x, y = view.point(row, col, 0.1)
                    self.assertEqual(view.pick((int(x), int(y))), (row, col))


if __name__ == "__main__":
    unittest.main()
