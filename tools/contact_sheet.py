"""Render a grid of on-board story frames for quick visual review.

Usage: python tools/contact_sheet.py <category> <key> [out.png] [--flip] [--mode full|highlights|off]
       python tools/contact_sheet.py all [out_dir]

Categories come from :mod:`cinema.demos`: opening, capture, tone, tactic, mate, stalemate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((16, 16))

from cinema.demos import catalogue, find  # noqa: E402
from cinema.detect import analyze_move, material_balance  # noqa: E402
from cinema.show import MoveShow  # noqa: E402
from cinema.view3d import VIEW_W, BoardView  # noqa: E402
from game_engine import Move  # noqa: E402


def build_show(demo, mode: str = "full") -> tuple[MoveShow, object]:
    game = demo.build()
    before = [row[:] for row in game.board]
    piece = game.piece_at(demo.move[0])
    played = game.push(Move(*demo.move))
    events = analyze_move(game, played, piece)
    show = MoveShow(played, piece, before, game.board, events, mode, game.winner)
    show.started_at = 0
    return show, game


def sheet(demo, out: str, flip: bool = False, mode: str = "full", cols: int = 4, rows: int = 3) -> float:
    show, game = build_show(demo, mode)
    view = BoardView()
    view.flipped = flip
    tw, th = 430, 450
    grid = pygame.Surface((cols * tw, rows * th))
    canvas = pygame.Surface((1440, 900))
    mood = max(-1.0, min(1.0, material_balance(game.board) / 12))
    spent = 0.0
    for i in range(cols * rows):
        t = 80 + (show.duration - 160) * i / (cols * rows - 1)
        started = time.perf_counter()
        frame = show.frame(int(t))
        view.compose(canvas, game.board, frame, t, mood, (mood, -mood))
        spent += time.perf_counter() - started
        thumb = pygame.transform.smoothscale(canvas.subsurface((0, 0, VIEW_W, 900)), (tw, th))
        grid.blit(thumb, ((i % cols) * tw, (i // cols) * th))
        label = pygame.font.Font(None, 26).render(f"{int(t)} ms", True, (255, 255, 255))
        grid.blit(label, ((i % cols) * tw + 8, (i // cols) * th + 6))
    pygame.image.save(grid, out)
    return spent / (cols * rows) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("key", nargs="?")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--at", type=int, nargs="*", help="save full-size frames at these times instead of a grid")
    args = parser.parse_args()
    if args.category == "all":
        out_dir = args.key or "/tmp/sheets"
        os.makedirs(out_dir, exist_ok=True)
        for demo in catalogue():
            ms = sheet(demo, os.path.join(out_dir, f"{demo.category}_{demo.key}.png"), args.flip, args.mode)
            print(f"{demo.category:9s} {demo.key:22s} {ms:6.1f} ms/frame")
        return
    demo = find(args.category, args.key)
    if demo is None:
        sys.exit(f"unknown demo {args.category}/{args.key}")
    out = args.out or f"/tmp/sheets/{args.category}_{args.key}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.at:
        show, game = build_show(demo, args.mode)
        view = BoardView()
        view.flipped = args.flip
        canvas = pygame.Surface((1440, 900))
        mood = max(-1.0, min(1.0, material_balance(game.board) / 12))
        for t in args.at:
            view.compose(canvas, game.board, show.frame(t), t, mood, (mood, -mood))
            path = out.replace(".png", f"_{t}.png")
            pygame.image.save(canvas.subsurface((0, 0, VIEW_W, 900)), path)
            print(path)
        return
    ms = sheet(demo, out, args.flip, args.mode)
    print(out, f"{ms:.1f} ms/frame")


if __name__ == "__main__":
    main()
