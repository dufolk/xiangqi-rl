"""Render a grid of cutscene frames for quick visual review.

Usage: python tools/contact_sheet.py <capture|opening|mate> <key> [out.png] [side]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((16, 16))

from cinema.gallery import build_scene  # noqa: E402


def main() -> None:
    category, key = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/shots/{category}_{key}.png"
    side = sys.argv[4] if len(sys.argv) > 4 else "red"
    scene = build_scene(category, key, side)
    cols, rows = 4, 3
    tw, th = 480, 300
    sheet = pygame.Surface((cols * tw, rows * th))
    canvas = pygame.Surface((1440, 900))
    for i in range(cols * rows):
        t = 200 + (scene.duration - 400) * i / (cols * rows - 1)
        canvas.fill((40, 30, 60))
        scene.draw(canvas, t)
        thumb = pygame.transform.smoothscale(canvas, (tw, th))
        sheet.blit(thumb, ((i % cols) * tw, (i // cols) * th))
    pygame.image.save(sheet, out)
    print(out)


if __name__ == "__main__":
    main()
