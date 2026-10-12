"""Small choreography helpers shared by the opening and mate cutscenes."""

from __future__ import annotations

import math

import pygame

from game_engine import BLACK, RED
from cinema import fx
from cinema.actors import GOLD, INK, WOOD, actor
from cinema.stage import GROUND, title_card

R_MAIN = 90


def other(side: str) -> str:
    return BLACK if side == RED else RED


def put(stage, kind, side, x, t, R=R_MAIN, y=GROUND, **kw) -> None:
    actor(stage, kind, side, (x, y), R, t=t, **kw)


def walk(t: float, t0: float, t1: float, x0: float, x1: float, speed: float = 70.0, ease=fx.ease_in_out) -> tuple[float, float]:
    """Position and gait phase for a walk from ``x0`` to ``x1`` during ``[t0, t1]``."""
    u = fx.seg(t, t0, t1)
    phase = t / speed if 0 < u < 1 else 0.0
    return fx.lerp(x0, x1, ease(u)), phase


def say(stage, text: str, anchor, t: float, t0: float, t1: float, **kw) -> None:
    fx.speech_bubble(stage, text, anchor, fx.seg(t, t0, t1), **kw)


def boom(stage, text: str, center, t: float, t0: float, length: float = 700, size: float = 84, color=(255, 230, 120), angle: float = -8) -> None:
    fx.onomatopoeia(stage, text, center, size, color, fx.seg(t, t0, t0 + length), angle)


def title(stage, scene, t: float, start: float, center=(720, 118), size: int = 84) -> None:
    title_card(stage, scene.event.title, scene.event.subtitle, center, fx.seg(t, start, scene.duration - 120), size=size)


def drifting_clouds(stage, t: float, count: int = 3, y: float = 110, alpha: float = 220) -> None:
    for n in range(count):
        fx.cloud(stage, (n * 560 + t * 0.03) % 1800 - 180, y + n * 45, 48 + n * 12, alpha=alpha)


def crosshair(stage, center, u: float, color=(255, 70, 90)) -> None:
    if u <= 0:
        return
    r = 120 - 70 * fx.ease_out(min(1.0, u))
    x, y = center
    pygame.draw.circle(stage, INK, (int(x), int(y)), int(r) + 3, 8)
    pygame.draw.circle(stage, color, (int(x), int(y)), int(r), 4)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = (x + dx * r * 0.6, y + dy * r * 0.6)
        b = (x + dx * (r + 30), y + dy * (r + 30))
        pygame.draw.line(stage, INK, a, b, 8)
        pygame.draw.line(stage, color, a, b, 4)


def wings(stage, center, span: float, t: float, flap_speed: float = 90.0) -> None:
    flap = math.sin(t / flap_speed)
    for sgn in (-1, 1):
        base = (center[0] + sgn * 20, center[1])
        pts = [base]
        for k in range(5):
            a = -0.4 - flap * 0.6 + k * 0.32
            r = span * (1 - k * 0.12)
            pts.append((base[0] + sgn * math.cos(a) * r, base[1] - math.sin(a) * r * 0.9))
        pts.append((base[0] + sgn * span * 0.3, base[1] + 20))
        pygame.draw.polygon(stage, (255, 255, 255), pts)
        pygame.draw.polygon(stage, INK, pts, 4)
        for p in pts[2:-1]:
            pygame.draw.line(stage, (200, 210, 230), base, p, 2)


def fishing_rod(stage, hand, tip, hook, color=GOLD) -> None:
    pygame.draw.line(stage, INK, hand, tip, 9)
    pygame.draw.line(stage, WOOD, hand, tip, 5)
    pygame.draw.line(stage, (240, 240, 240), tip, hook, 2)
    pygame.draw.arc(stage, INK, (int(hook[0] - 12), int(hook[1] - 4), 24, 26), math.pi, math.tau + 1.2, 7)
    pygame.draw.arc(stage, color, (int(hook[0] - 12), int(hook[1] - 4), 24, 26), math.pi, math.tau + 1.2, 4)


def iron_bars(stage, rect: pygame.Rect, u: float, color=(110, 116, 140)) -> None:
    if u <= 0:
        return
    for k in range(6):
        x = rect.x + rect.w * (k + 0.5) / 6
        h = rect.h * fx.ease_bounce(fx.seg(u, k * 0.06, 0.6 + k * 0.06))
        pygame.draw.rect(stage, INK, (int(x - 10), rect.y, 20, int(h)))
        pygame.draw.rect(stage, color, (int(x - 6), rect.y, 12, int(h) - 3))


def question_marks(stage, center, t: float, alpha: float = 255) -> None:
    for k in range(3):
        a = t / 400.0 + k * 2.1
        fx.outlined_text(stage, "?", 46, (255, 255, 255), (center[0] + math.cos(a) * 70, center[1] + math.sin(a) * 20 - k * 10),
                         ow=3, alpha=int(alpha))
