"""Scene painters for openings, tactics, checks and mates, staged on the real squares.

Every painter has the signature ``painter(show, frame, event, lt, u, active, dur)`` where ``lt`` is
the local time in ms, ``u`` its 0..1 fraction and ``active`` whether the segment is still playing.
Painters wake pieces up through :meth:`MoveShow.cast` and paint props through the show's
ground / air / overlay hooks, so everything follows the camera, the flip state and the pieces.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import pygame

from cinema.actors import GOLD, INK
from cinema.fx import (
    _alpha_layer, bamboo, banner, clamp, cloud, confetti, dizzy_stars, ease_back, ease_out,
    ghost, hop, lerp, moon, outlined_text, radial_lines, seg, sparkles, star_points, sweat_drop, with_alpha,
)
from game_engine import ADVISOR, BLACK, CANNON, HORSE, KING, PAWN, RED, ROOK

Square = tuple[int, int]
Painter = Callable[..., None]

WATER = (90, 170, 230)
LIGHT = (255, 236, 150)
DANGER = (255, 76, 60)


# --- board helpers ----------------------------------------------------------------

def _squares(show, side: str, kind: str) -> list[Square]:
    return [(r, c) for r in range(10) for c in range(9)
            if show.after[r][c] is not None and show.after[r][c].side == side and show.after[r][c].kind == kind]


def _nearest(show, side: str, kind: str, near) -> Optional[Square]:
    found = _squares(show, side, kind)
    if not found:
        return None
    return min(found, key=lambda sq: abs(sq[0] - near[0]) + abs(sq[1] - near[1]))


def _king(show, side: str) -> Optional[Square]:
    found = _squares(show, side, KING)
    return found[0] if found else None


def _palace_rows(side: str) -> tuple[int, int]:
    return (7, 9) if side == RED else (0, 2)


def _ahead(show, sq, steps: float) -> tuple[float, float]:
    return sq[0] + show.fwd * steps, sq[1]


def _ribbon(show, f, e, lt, dur, start: float = 160) -> None:
    show.ribbon(f, e.title, e.subtitle, seg(lt, start, dur + 240), tag=e.tag)


def _on_canvas(show, f, fn) -> None:
    """Draw directly onto the canvas before the pieces (behind them), e.g. boats or sunbursts."""
    show.on_ground(f, fn)


def _screen_alpha(cv: pygame.Surface, draw: Callable[[pygame.Surface], None]) -> None:
    layer = _alpha_layer(cv)
    draw(layer)
    cv.blit(layer, (0, 0))


# --- prop primitives (cv, view, ...) -----------------------------------------------

def _crosshair(cv, v, sq, u: float, color=DANGER) -> None:
    x, y, s = v.project(sq[0], sq[1], 0.55)
    r = s * (0.95 - 0.45 * ease_out(u))
    fs = 0.75
    rect = pygame.Rect(0, 0, int(2 * r), int(2 * r * fs))
    rect.center = (int(x), int(y))
    pygame.draw.ellipse(cv, INK, rect.inflate(4, 4), 5)
    pygame.draw.ellipse(cv, color, rect, 3)
    for k in range(4):
        a = u * 2.2 + k * math.pi / 2
        a0, a1 = r * 0.55, r * 1.3
        p0 = (x + math.cos(a) * a0, y + math.sin(a) * a0 * fs)
        p1 = (x + math.cos(a) * a1, y + math.sin(a) * a1 * fs)
        pygame.draw.line(cv, INK, p0, p1, 5)
        pygame.draw.line(cv, color, p0, p1, 3)
    pygame.draw.circle(cv, color, (int(x), int(y)), max(2, int(s * 0.04)))


def _shield(cv, v, sq, u: float, text: str, color=(70, 120, 200), z: float = 1.35) -> None:
    pop = ease_back(clamp(u))
    if pop <= 0:
        return
    x, y, s = v.project(sq[0], sq[1], z)
    w = s * 0.6 * pop
    pts = [(x - w / 2, y - w * 0.55), (x, y - w * 0.68), (x + w / 2, y - w * 0.55), (x + w / 2, y), (x, y + w * 0.62), (x - w / 2, y)]
    pygame.draw.polygon(cv, INK, [(px + 3, py + 3) for px, py in pts])
    pygame.draw.polygon(cv, color, pts)
    pygame.draw.polygon(cv, GOLD, pts, max(2, int(s * 0.04)))
    outlined_text(cv, text, w * 0.55, (255, 246, 220), (x, y - w * 0.04), ow=2)


def _sandbags(cv, v, sq, fwd: int) -> None:
    for k, (dc, dz) in enumerate(((-0.28, 0), (0.0, 0), (0.28, 0), (-0.14, 0.13), (0.14, 0.13))):
        x, y, s = v.project(sq[0] + fwd * 0.5, sq[1] + dc, 0.07 + dz)
        rect = pygame.Rect(0, 0, int(s * 0.3), int(s * 0.15))
        rect.center = (int(x), int(y))
        pygame.draw.ellipse(cv, INK, rect.inflate(4, 4))
        pygame.draw.ellipse(cv, (196, 168, 112) if k % 2 else (178, 148, 96), rect)
        pygame.draw.line(cv, (130, 100, 60), (rect.centerx, rect.top + 2), (rect.centerx, rect.bottom - 2), 1)


def _telescope(cv, v, sq, fwd: int, t: float) -> None:
    x0, y0, s = v.project(sq[0], sq[1], 0.78)
    x1, y1, _ = v.project(sq[0] + fwd * 0.55, sq[1] + 0.2 * math.sin(t / 400.0), 0.92)
    pygame.draw.line(cv, INK, (x0, y0), (x1, y1), max(5, int(s * 0.11)))
    pygame.draw.line(cv, GOLD, (x0, y0), (x1, y1), max(3, int(s * 0.07)))
    pygame.draw.circle(cv, (200, 230, 255), (int(x1), int(y1)), max(2, int(s * 0.05)))
    if int(t / 300) % 3 == 0:
        pygame.draw.polygon(cv, (255, 255, 255), star_points(x1, y1, s * 0.12, s * 0.03, 4, t / 200.0))


def _boat(cv, v, sq, t: float) -> None:
    bob = math.sin(t / 260.0) * 0.03
    hull = [(sq[0] - 0.2, sq[1] - 0.62), (sq[0] - 0.2, sq[1] + 0.62), (sq[0] + 0.28, sq[1] + 0.42), (sq[0] + 0.28, sq[1] - 0.42)]
    top = [v.point(r, c, 0.18 + bob) for r, c in hull]
    low = [v.point(r, c, 0.0 + bob) for r, c in hull]
    for k in range(3):
        v.ground_ring(sq, 0.7 + k * 0.18 + (t / 900.0) % 0.18, WATER, 2, 140 - k * 40)
    pygame.draw.polygon(cv, (110, 64, 34), [top[2], top[3], low[3], low[2]])
    pygame.draw.polygon(cv, INK, [top[2], top[3], low[3], low[2]], 2)
    pygame.draw.polygon(cv, (164, 102, 56), top)
    pygame.draw.polygon(cv, INK, top, 2)


def _searchlight(cv, v, origin, fwd: int, t: float, u: float) -> None:
    sweep = math.sin(t / 420.0) * 2.6
    far = origin[0] + fwd * 3.2
    a = (far, origin[1] + sweep - 0.9)
    b = (far, origin[1] + sweep + 0.9)
    v.ground_poly([origin, a, b], LIGHT, 90 * hop(u) + 20)
    v.ground_ring(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), 0.9, LIGHT, 3, 150 * hop(u))


def _hook(cv, v, sq, fwd: int, t: float, u: float) -> None:
    tip = (sq[0] + fwd * 0.9, sq[1] + 0.3)
    x0, y0, s = v.project(sq[0], sq[1], 1.0)
    x1, y1, _ = v.project(tip[0], tip[1], 1.25)
    drop = 0.35 + 0.15 * math.sin(t / 300.0)
    x2, y2, _ = v.project(tip[0], tip[1], drop)
    pygame.draw.line(cv, INK, (x0, y0), (x1, y1), max(3, int(s * 0.06)))
    pygame.draw.line(cv, (150, 100, 60), (x0, y0), (x1, y1), max(1, int(s * 0.035)))
    pygame.draw.line(cv, (240, 240, 240), (x1, y1), (x2, y2), 1)
    r = s * 0.12
    rect = pygame.Rect(int(x2 - r), int(y2 - r * 0.2), int(2 * r), int(2 * r))
    pygame.draw.arc(cv, INK, rect.inflate(4, 4), math.pi, math.tau, max(4, int(s * 0.06)))
    pygame.draw.arc(cv, GOLD, rect, math.pi, math.tau, max(3, int(s * 0.04)))
    if int(t / 160) % 4 == 0:
        pygame.draw.polygon(cv, (255, 255, 230), star_points(x2 + r, y2 + r * 0.6, s * 0.1, s * 0.025, 4, 0))


def _fish(cv, v, sq, u: float) -> None:
    if u <= 0 or u >= 1:
        return
    x, y, s = v.project(lerp(4.5, sq[0], u * 0.4), sq[1] + 1.2 - u * 1.6, 0.1 + hop(u) * 1.4)
    a = (u - 0.5) * 2.4
    body = s * 0.22
    pts = [(x + math.cos(a) * body, y + math.sin(a) * body)]
    for k in range(1, 8):
        ang = a + math.pi * k / 4
        rr = body * (0.45 if k in (2, 6) else 0.3)
        pts.append((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
    tail = (x - math.cos(a) * body * 1.3, y - math.sin(a) * body * 1.3)
    fin = [(tail[0] + math.sin(a) * body * 0.4, tail[1] - math.cos(a) * body * 0.4), (x - math.cos(a) * body * 0.7, y - math.sin(a) * body * 0.7),
           (tail[0] - math.sin(a) * body * 0.4, tail[1] + math.cos(a) * body * 0.4)]
    pygame.draw.polygon(cv, INK, fin)
    pygame.draw.polygon(cv, (250, 150, 70), [(p[0] * 0.92 + tail[0] * 0.08, p[1] * 0.92 + tail[1] * 0.08) for p in fin])
    pygame.draw.polygon(cv, (250, 160, 70), pts)
    pygame.draw.polygon(cv, INK, pts, 2)
    pygame.draw.circle(cv, INK, (int(pts[0][0] * 0.7 + x * 0.3), int(pts[0][1] * 0.7 + y * 0.3) - 2), max(1, int(s * 0.025)))


def _splash(cv, v, sq, u: float) -> None:
    if u <= 0 or u >= 1:
        return
    rng = random.Random(9)
    for k in range(14):
        a = rng.uniform(0, math.tau)
        d = rng.uniform(0.3, 0.9) * ease_out(u)
        x, y, s = v.project(sq[0] + math.sin(a) * d * 0.6, sq[1] + math.cos(a) * d, hop(u) * rng.uniform(0.6, 1.4))
        r = max(2, int(s * 0.05 * (1 - u * 0.6)))
        pygame.draw.circle(cv, INK, (int(x), int(y)), r + 1)
        pygame.draw.circle(cv, (170, 220, 255), (int(x), int(y)), r)
    v.ground_ring(sq, 0.3 + 0.9 * ease_out(u), WATER, 3, 200 * (1 - u))


def _hand_from_cloud(cv, v, sq, fwd: int, u: float, t: float) -> None:
    descend = ease_out(seg(u, 0, 0.35))
    x, y, s = v.project(sq[0] + fwd * 0.6, sq[1] - 0.5, 3.2 - 0.8 * descend)
    cloud(cv, x, y, s * 0.42, (255, 255, 255), 240 * min(1.0, descend * 2))
    if descend <= 0.2:
        return
    fx_, fy, _ = v.project(sq[0] + fwd * 1.0, sq[1], 1.3 - 0.3 * descend)
    arm = [(x - s * 0.08, y + s * 0.1), (x + s * 0.08, y + s * 0.1), (fx_ + s * 0.05, fy), (fx_ - s * 0.05, fy)]
    pygame.draw.polygon(cv, INK, arm)
    pygame.draw.polygon(cv, (255, 222, 190), [(px * 0.94 + fx_ * 0.06, py * 0.94 + fy * 0.06) for px, py in arm])
    pygame.draw.circle(cv, INK, (int(fx_), int(fy)), max(3, int(s * 0.07)))
    pygame.draw.circle(cv, (255, 222, 190), (int(fx_), int(fy)), max(2, int(s * 0.05)))
    sparkles(cv, (fx_, fy), t, s * 0.4, LIGHT, count=6)


def _reeds(cv, v, sq, t: float, u: float) -> None:
    rng = random.Random(4)
    grow = ease_out(u)
    for _ in range(18):
        r = sq[0] + rng.uniform(-0.7, 0.7)
        c = sq[1] + rng.choice((-1, 1)) * rng.uniform(0.35, 0.8)
        x, y, s = v.project(r, c, 0)
        h = s * rng.uniform(0.6, 1.0) * grow
        sway = math.sin(t / 500.0 + r * 3 + c) * s * 0.08
        pygame.draw.line(cv, (70, 120, 60), (x, y), (x + sway, y - h), max(2, int(s * 0.03)))
        if rng.random() < 0.5:
            rect = pygame.Rect(0, 0, max(2, int(s * 0.06)), max(4, int(s * 0.2)))
            rect.center = (int(x + sway), int(y - h))
            pygame.draw.ellipse(cv, (150, 104, 60), rect)


def _wings(cv, v, sp, t: float, u: float) -> None:
    x, y, s = v.project(sp.row, sp.col, sp.z + 0.62)
    flap = math.sin(t / 90.0) * 0.5
    span = s * 0.55 * ease_back(clamp(u * 3))
    for sgn in (-1, 1):
        pts = [(x + sgn * s * 0.12, y)]
        for k in range(5):
            a = flap - 0.3 + k * 0.25
            rr = span * (1 - k * 0.12)
            pts.append((x + sgn * (s * 0.12 + math.cos(a) * rr), y - math.sin(a) * rr * 0.9 + k * s * 0.03))
        pts.append((x + sgn * s * 0.12, y + s * 0.12))
        pygame.draw.polygon(cv, (255, 255, 255), pts)
        pygame.draw.polygon(cv, INK, pts, 2)


def _feathers(cv, v, sq, t: float) -> None:
    rng = random.Random(6)
    for k in range(7):
        ph = (t / 1600.0 + rng.random()) % 1.0
        x, y, s = v.project(sq[0] + rng.uniform(-0.8, 0.8), sq[1] + rng.uniform(-0.9, 0.9) + math.sin(ph * 9) * 0.15, 1.8 * (1 - ph))
        rect = pygame.Rect(0, 0, max(3, int(s * 0.16)), max(2, int(s * 0.06)))
        rect.center = (int(x), int(y))
        pygame.draw.ellipse(cv, (255, 255, 255), rect)
        pygame.draw.ellipse(cv, INK, rect, 1)


def _sunburst(cv, v, sq, t: float, u: float) -> None:
    x, y, s = v.project(sq[0], sq[1], 0.6)
    radial_lines(cv, (x, y), t, s * 0.4, s * (0.8 + 1.0 * ease_out(u)), (255, 214, 120), count=18, alpha=170 * hop(u) + 40)


def _bamboo_grove(cv, v, sq, t: float, u: float) -> None:
    for dc, dr in ((-0.75, -0.3), (0.75, 0.2), (-0.6, 0.45)):
        x, y, s = v.project(sq[0] + dr, sq[1] + dc, 0)
        bamboo(cv, x, y, s * 1.9 * ease_out(u), t)


def _folding_screen(cv, v, row: float, c0: float, c1: float, u: float, color) -> None:
    if c1 < c0:
        c0, c1 = c1, c0
    panels = max(2, int(round(c1 - c0)) * 2)
    h = 0.95 * ease_back(clamp(u))
    if h <= 0:
        return

    def draw(layer):
        for k in range(panels):
            a = (row + (0.12 if k % 2 else -0.12), lerp(c0, c1, k / panels))
            b = (row + (-0.12 if k % 2 else 0.12), lerp(c0, c1, (k + 1) / panels))
            quad = [v.point(a[0], a[1], 0), v.point(b[0], b[1], 0), v.point(b[0], b[1], h), v.point(a[0], a[1], h)]
            pygame.draw.polygon(layer, with_alpha(color if k % 2 else tuple(min(255, c + 30) for c in color), 175), quad)
            pygame.draw.polygon(layer, with_alpha(INK, 230), quad, 2)
            mid = [(quad[0][0] * 0.5 + quad[1][0] * 0.5, quad[0][1] * 0.5 + quad[1][1] * 0.5),
                   (quad[3][0] * 0.5 + quad[2][0] * 0.5, quad[3][1] * 0.5 + quad[2][1] * 0.5)]
            pygame.draw.line(layer, with_alpha(GOLD, 200), mid[0], mid[1], 2)
    _screen_alpha(cv, draw)


def _lightning(cv, v, a, b, t: float, color=(170, 210, 255), z: float = 0.85) -> None:
    rng = random.Random(int(t / 70))
    pts = []
    for k in range(9):
        u = k / 8
        x, y, s = v.project(lerp(a[0], b[0], u), lerp(a[1], b[1], u), z)
        jitter = 0 if k in (0, 8) else rng.uniform(-1, 1) * s * 0.18
        pts.append((x + jitter * 0.4, y + jitter))
    pygame.draw.lines(cv, (255, 255, 255), False, pts, 7)
    pygame.draw.lines(cv, color, False, pts, 3)


def _swords(cv, v, sq, u: float, t: float) -> None:
    x, y, s = v.project(sq[0], sq[1], 0.7)
    close = ease_out(seg(u, 0, 0.35))
    length = s * 0.9
    for sgn in (-1, 1):
        a = -math.pi / 2 + sgn * (0.5 + 0.9 * (1 - close))
        hx, hy = x - math.cos(a) * length * 0.45, y - math.sin(a) * length * 0.45
        tx, ty = x + math.cos(a) * length * 0.55, y + math.sin(a) * length * 0.55
        pygame.draw.line(cv, INK, (hx, hy), (tx, ty), max(7, int(s * 0.12)))
        pygame.draw.line(cv, (220, 228, 240), (hx, hy), (tx, ty), max(4, int(s * 0.07)))
        gx, gy = -math.sin(a) * s * 0.14, math.cos(a) * s * 0.14
        pygame.draw.line(cv, GOLD, (hx - gx, hy - gy), (hx + gx, hy + gy), max(4, int(s * 0.06)))
    if close >= 1 and int(t / 120) % 2 == 0:
        pygame.draw.polygon(cv, (255, 250, 200), star_points(x, y - s * 0.1, s * 0.25, s * 0.07, 6, t / 300.0))


def _walls(cv, v, side: str, u: float) -> None:
    r0, r1 = _palace_rows(side)
    h = 0.65 * ease_back(clamp(u))
    if h <= 0:
        return
    corners = [(r0 - 0.45, 2.55), (r0 - 0.45, 5.45), (r1 + 0.45, 5.45), (r1 + 0.45, 2.55)]

    def draw(layer):
        for a, b in zip(corners, corners[1:] + corners[:1]):
            quad = [v.point(a[0], a[1], 0), v.point(b[0], b[1], 0), v.point(b[0], b[1], h), v.point(a[0], a[1], h)]
            pygame.draw.polygon(layer, (196, 64, 60, 120), quad)
            pygame.draw.polygon(layer, (60, 20, 20, 220), quad, 2)
            steps = 6
            for k in range(steps):
                if k % 2:
                    continue
                p0 = (lerp(a[0], b[0], k / steps), lerp(a[1], b[1], k / steps))
                p1 = (lerp(a[0], b[0], (k + 1) / steps), lerp(a[1], b[1], (k + 1) / steps))
                merlon = [v.point(p0[0], p0[1], h), v.point(p1[0], p1[1], h), v.point(p1[0], p1[1], h + 0.14), v.point(p0[0], p0[1], h + 0.14)]
                pygame.draw.polygon(layer, (176, 54, 50, 200), merlon)
    _screen_alpha(cv, draw)


def _road(v, a, b, u: float, t: float) -> None:
    v.ground_line(a, b, (90, 84, 80), 0.5, 120, ease_out(u))
    v.ground_line(a, b, (255, 255, 255), 0.05, 200, ease_out(u), dashed=True)


def _crown(cv, v, sq, u: float, z0: float = 1.0) -> None:
    if u <= 0 or u >= 1:
        return
    x, y, s = v.project(sq[0] - 0.8 * u, sq[1] + 1.1 * u, z0 + hop(u) * 1.8)
    w = s * 0.34
    spin = u * 8
    pts = [(x - w / 2, y + w * 0.3), (x - w / 2, y - w * 0.2), (x - w / 4, y + w * 0.02), (x, y - w * 0.36),
           (x + w / 4, y + w * 0.02), (x + w / 2, y - w * 0.2), (x + w / 2, y + w * 0.3)]
    c, si = math.cos(spin), math.sin(spin)
    pts = [(x + (px - x) * c - (py - y) * si, y + (px - x) * si + (py - y) * c) for px, py in pts]
    pygame.draw.polygon(cv, GOLD, pts)
    pygame.draw.polygon(cv, INK, pts, 2)


def _heart(cv, v, sq, u: float, t: float) -> None:
    x, y, s = v.project(sq[0], sq[1], 1.75 + 0.05 * math.sin(t / 120.0))
    r = s * 0.2 * ease_back(clamp(u * 3))
    if r < 2:
        return
    pts = []
    for k in range(32):
        a = k / 32 * math.tau
        hx = 16 * math.sin(a) ** 3
        hy = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
        pts.append((x + hx * r / 16, y + hy * r / 16))
    split = ease_out(seg(u, 0.35, 0.6)) * r * 0.35
    for sgn in (-1, 1):
        half = [(px + sgn * split, py) for px, py in pts if (px - x) * sgn >= -1] or pts
        pygame.draw.polygon(cv, (230, 50, 70), half)
        pygame.draw.polygon(cv, INK, half, 2)
    if split > 1:
        crack = [(x, y - r * 0.6), (x - r * 0.15, y - r * 0.2), (x + r * 0.12, y + r * 0.1), (x, y + r * 0.9)]
        pygame.draw.lines(cv, INK, False, crack, 3)


def _iron_bar(cv, v, row: float, u: float) -> None:
    slide = ease_out(clamp(u))
    a = (row, 2.6 - 2.5 * (1 - slide))
    b = (row, 5.4 - 2.5 * (1 - slide))
    pa, pb = v.point(a[0], a[1], 0.55), v.point(b[0], b[1], 0.55)
    s = v.project(row, 4, 0)[2]
    pygame.draw.line(cv, INK, pa, pb, max(9, int(s * 0.18)))
    pygame.draw.line(cv, (120, 128, 140), pa, pb, max(6, int(s * 0.12)))
    pygame.draw.line(cv, (200, 208, 220), (pa[0], pa[1] - 2), (pb[0], pb[1] - 2), 2)
    for end in (pa, pb):
        pygame.draw.circle(cv, INK, (int(end[0]), int(end[1])), max(6, int(s * 0.1)))
        pygame.draw.circle(cv, (150, 158, 170), (int(end[0]), int(end[1])), max(4, int(s * 0.07)))


def _net(cv, v, sq, u: float) -> None:
    drop = ease_out(clamp(u))
    z = 2.8 - 2.3 * drop
    pts = {}
    for i in range(5):
        for j in range(5):
            r = sq[0] + (i - 2) * 0.3
            c = sq[1] + (j - 2) * 0.3
            sag = (1 - ((i - 2) ** 2 + (j - 2) ** 2) / 8) * 0.35 * drop
            pts[i, j] = v.point(r, c, z - sag)

    def draw(layer):
        for i in range(5):
            pygame.draw.lines(layer, (240, 240, 230, 220), False, [pts[i, j] for j in range(5)], 2)
            pygame.draw.lines(layer, (240, 240, 230, 220), False, [pts[j, i] for j in range(5)], 2)
    _screen_alpha(cv, draw)


def _cage(cv, v, sq, u: float) -> None:
    drop = ease_out(clamp(u))
    z0 = 2.5 * (1 - drop)
    corners = [(sq[0] - 0.42, sq[1] - 0.42), (sq[0] - 0.42, sq[1] + 0.42), (sq[0] + 0.42, sq[1] + 0.42), (sq[0] + 0.42, sq[1] - 0.42)]
    for k in range(12):
        a, b = corners[k // 3], corners[(k // 3 + 1) % 4]
        p = (lerp(a[0], b[0], (k % 3) / 3), lerp(a[1], b[1], (k % 3) / 3))
        x0, y0 = v.point(p[0], p[1], z0)
        x1, y1 = v.point(p[0], p[1], z0 + 1.3)
        pygame.draw.line(cv, INK, (x0, y0), (x1, y1), 5)
        pygame.draw.line(cv, (150, 156, 170), (x0, y0), (x1, y1), 3)
    for z in (z0 + 0.02, z0 + 1.3):
        ring = [v.point(r, c, z) for r, c in corners]
        pygame.draw.polygon(cv, INK, ring, 5)
        pygame.draw.polygon(cv, (170, 176, 190), ring, 3)


def _fishing(cv, v, rod_sq, king_sq, lift: float, t: float) -> None:
    base = v.point(rod_sq[0], rod_sq[1], 0.9)
    tip_sq = ((rod_sq[0] + king_sq[0]) / 2, (rod_sq[1] + king_sq[1]) / 2)
    tip = v.point(tip_sq[0], tip_sq[1], 2.6)
    ctrl = v.point(rod_sq[0] * 0.7 + tip_sq[0] * 0.3, rod_sq[1] * 0.7 + tip_sq[1] * 0.3, 2.3)
    rod = [(lerp(lerp(base[0], ctrl[0], k / 8), lerp(ctrl[0], tip[0], k / 8), k / 8),
            lerp(lerp(base[1], ctrl[1], k / 8), lerp(ctrl[1], tip[1], k / 8), k / 8)) for k in range(9)]
    pygame.draw.lines(cv, INK, False, rod, 6)
    pygame.draw.lines(cv, (150, 104, 60), False, rod, 3)
    hook = v.point(king_sq[0], king_sq[1], 1.3 + lift)
    pygame.draw.line(cv, (245, 245, 245), tip, hook, 2)
    pygame.draw.circle(cv, (255, 80, 80), (int((tip[0] + hook[0]) / 2), int((tip[1] + hook[1]) / 2)), 5)


def _x_mark(v, sq, u: float) -> None:
    k = ease_back(clamp(u)) * 0.3
    if k <= 0:
        return
    v.ground_line((sq[0] - k, sq[1] - k), (sq[0] + k, sq[1] + k), DANGER, 0.09, 230)
    v.ground_line((sq[0] - k, sq[1] + k), (sq[0] + k, sq[1] - k), DANGER, 0.09, 230)


def _escape_squares(show, king: Square) -> list[Square]:
    defender = show.after[king[0]][king[1]].side if show.after[king[0]][king[1]] else show.enemy
    r0, r1 = _palace_rows(defender)
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = king[0] + dr, king[1] + dc
        if r0 <= r <= r1 and 3 <= c <= 5:
            p = show.after[r][c]
            if p is None or p.side != defender:
                out.append((r, c))
    return out


# --- openings -----------------------------------------------------------------------

def _opening_base(show, f, e, lt, dur, expr: str = "cheer", hop_height: float = 0.25):
    m = show.cast(f, show.move.end, lt, dur, expr)
    m.z = hop(seg(lt, 0, 420)) * hop_height
    _ribbon(show, f, e, lt, dur)
    return m


def _center_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "smug")
    end = show.move.end
    pawn_row = 3 if show.enemy == BLACK else 6
    target = (pawn_row, 4)
    m.look = target
    lock = seg(lt, 200, 900)
    show.beam(f, end, target, ease_out(lock), DANGER, 170, 0.05, dashed=True)
    show.in_air(f, lambda cv, v: _crosshair(cv, v, target, lock))
    pawn = show.cast(f, target, lt, dur, "scared", wake=500)
    pawn.shake = math.sin(lt / 25.0) * 2 * seg(lt, 700, 900)
    show.bubble(f, end, "中路，我镇住了！", seg(lt, 250, 1500), z=1.5)
    show.bubble(f, target, "被、被瞄上了……", seg(lt, 950, dur - 100), z=1.4, size=16)


def _cross_palace_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    _opening_base(show, f, e, lt, dur, "smug")
    r0, r1 = _palace_rows(show.side)
    glow = hop(seg(lt, 0, dur))
    show.on_ground(f, lambda cv, v: v.ground_poly([(r0 - 0.4, 2.6), (r0 - 0.4, 5.4), (r1 + 0.4, 5.4), (r1 + 0.4, 2.6)], LIGHT, 90 * glow))
    path = [show.move.start, (show.move.start[0], 4), show.move.end]
    show.on_ground(f, lambda cv, v: (v.ground_line(path[0], path[1], GOLD, 0.06, 200, 1, dashed=True),
                                     v.ground_line(path[1], path[2], GOLD, 0.06, 200, 1, dashed=True, arrow=True)))
    for sq in _squares(show, show.side, ADVISOR):
        adv = show.cast(f, sq, lt, dur, "smug", wake=300)
        adv.tilt = 0.35 * hop(seg(lt, 500, 1100))
        adv.squash = 0.1 * hop(seg(lt, 500, 1100))
        show.bubble(f, sq, "请过宫～", seg(lt, 600, 1700), z=1.3, size=15)


def _advisor_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    _opening_base(show, f, e, lt, dur, "smug", 0.1)
    end = show.move.end
    show.in_air(f, lambda cv, v: _shield(cv, v, end, seg(lt, 200, 700), "稳"))
    adv = _nearest(show, show.side, ADVISOR, end)
    if adv:
        sp = show.cast(f, adv, lt, dur, "cheer", wake=300)
        sp.look = end
        show.bubble(f, adv, "有我在侧，安心。", seg(lt, 600, 1800), z=1.3, size=15)


def _pawn_base_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "smug", 0.0)
    end = show.move.end
    pawn = None
    for k in range(1, 5):
        sq = (end[0] + show.fwd * k, end[1])
        if 0 <= sq[0] <= 9 and show.after[sq[0]][sq[1]] is not None:
            pawn = sq
            break
    peek = math.sin(lt / 160.0)
    m.squash = 0.25 + 0.1 * peek
    m.tilt = 0.25 * peek
    m.expr = "smug"
    if pawn:
        p = show.cast(f, pawn, lt, dur, "fierce", wake=300)
        p.look = _ahead(show, pawn, 2)
        show.bubble(f, pawn, "我在前面挡着！", seg(lt, 400, 1500), z=1.35, size=15)
    show.bubble(f, end, "嘘……藏好了。", seg(lt, 250, 1300), z=1.2, size=15)


def _retreat_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "sleepy", 0.0)
    end, fwd = show.move.end, show.fwd
    m.look = _ahead(show, end, 3)
    show.on_ground(f, lambda cv, v: _sandbags(cv, v, end, fwd) if seg(lt, 100, 400) > 0 else None)
    show.in_air(f, lambda cv, v: _telescope(cv, v, end, fwd, lt))
    show.bubble(f, end, "以静制动……", seg(lt, 400, 1700), z=1.4, size=15)


def _edge_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    _opening_base(show, f, e, lt, dur, "cheer")
    start = show.move.start
    horse = _nearest(show, show.side, HORSE, start)
    rook = _nearest(show, show.side, ROOK, start)
    if horse:
        show.on_ground(f, lambda cv, v: v.ground_line(horse, (horse[0] + show.fwd * 2, start[1]), GOLD, 0.08, 200, seg(lt, 200, 800), arrow=True))
        sp = show.cast(f, horse, lt, dur, "cheer", wake=300)
        sp.z = hop(seg(lt, 500, 900)) * 0.3
        show.bubble(f, horse, "多谢让路！", seg(lt, 500, 1600), z=1.35, size=15)
    if rook:
        show.cast(f, rook, lt, dur, "smug", wake=400)
        show.bubble(f, rook, "车马出动～", seg(lt, 900, dur - 100), z=1.35, size=15)


def _river_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "smug", 0.0)
    end, fwd = show.move.end, show.fwd
    m.z = 0.18 + math.sin(lt / 260.0) * 0.03
    show.on_ground(f, lambda cv, v: _boat(cv, v, end, lt))
    show.on_ground(f, lambda cv, v: _searchlight(cv, v, end, fwd, lt, u))
    show.bubble(f, end, "沿河巡逻中～", seg(lt, 300, 1600), z=1.6, size=15)


def _hook_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "sleepy", 0.0)
    end, fwd = show.move.end, show.fwd
    m.look = _ahead(show, end, 2)
    show.in_air(f, lambda cv, v: _hook(cv, v, end, fwd, lt, u))
    show.in_air(f, lambda cv, v: _fish(cv, v, _ahead(show, end, 1.4), seg(lt, 700, 1500)))
    show.bubble(f, end, "愿者上钩……", seg(lt, 300, 1500), z=1.5, size=15)
    if lt > 1100:
        show.word(f, _ahead(show, end, 1.2), "扑通!", seg(lt, 1100, 1700), z=0.8, size=0.45, color=(170, 220, 255))


def _cross_river_cannon(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "surprised", 0.0)
    end = show.move.end
    river = (4.5, end[1])
    show.in_air(f, lambda cv, v: _splash(cv, v, river, seg(lt, 0, 800)))
    m.shake = math.sin(lt / 30.0) * 3 * (1 - seg(lt, 400, 900))
    if lt > 700:
        m.expr = "fierce"
    show.bubble(f, end, "好凉！……冲过来了！", seg(lt, 300, 1700), z=1.5, size=15)


def _immortal_pawn(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "surprised", 0.0)
    end, fwd = show.move.end, show.fwd
    m.look = _ahead(show, end, 2)
    show.in_air(f, lambda cv, v: _hand_from_cloud(cv, v, end, fwd, u, lt))
    path = seg(lt, 700, 1300)
    show.on_ground(f, lambda cv, v: (v.ground_line(end, _ahead(show, end, 2.2), LIGHT, 0.32, 90, path),
                                     v.ground_line(end, _ahead(show, end, 2.2), (255, 255, 255), 0.08, 200, path, arrow=True)))
    if lt > 800:
        m.expr = "cheer"
    show.bubble(f, end, "仙人给我指路啦！", seg(lt, 900, dur - 80), z=1.4, size=15)


def _edge_pawn(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "smug", 0.0)
    end = show.move.end
    m.look = (end[0], 4)
    show.on_ground(f, lambda cv, v: _reeds(cv, v, end, lt, seg(lt, 0, 600)))
    show.bubble(f, end, "从边路摸过去……", seg(lt, 300, 1600), z=1.4, size=15)


def _center_pawn(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "fierce", 0.35)
    end = show.move.end
    m.swing = 0.6 * hop(seg(lt, 200, 600))
    color = (206, 46, 40) if show.side == RED else (36, 44, 52)

    def flag(cv, v):
        x, y, s = v.project(end[0] - show.fwd * 0.3, end[1] + 0.45, 0)
        banner(cv, x, y, s * 1.6 * ease_out(seg(lt, 0, 500)), color, lt, "冲")
    show.on_ground(f, flag)
    show.word(f, end, "冲啊!", seg(lt, 250, 1300), z=1.6, size=0.6, color=(255, 240, 160))
    for sq in _squares(show, show.side, PAWN):
        if sq != end and abs(sq[1] - end[1]) <= 2:
            p = show.cast(f, sq, lt, dur, "cheer", wake=350)
            p.z = hop(seg((lt + sq[1] * 90) % 600, 0, 600)) * 0.18


def _flying_elephant(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "cheer", 0.0)
    end = show.move.end
    m.z = 0.35 * hop(seg(lt, 0, 900)) + 0.08 * math.sin(lt / 140.0)
    sp = m
    show.in_air(f, lambda cv, v: _wings(cv, v, sp, lt, u))
    show.in_air(f, lambda cv, v: _feathers(cv, v, end, lt))
    show.bubble(f, end, "飞相镇中，稳如泰山！", seg(lt, 400, 1700), z=1.6, size=15)


def _edge_elephant(show, f, e, lt, u, active, dur):
    if not active:
        return
    _opening_base(show, f, e, lt, dur, "cheer")
    end = show.move.end
    horse = _nearest(show, show.side, HORSE, end)
    if horse:
        sp = show.cast(f, horse, lt, dur, "cheer", wake=300)
        sp.tilt = 0.3 * hop(seg(lt, 500, 900))
        show.bubble(f, horse, "谢啦，象兄！", seg(lt, 500, 1600), z=1.35, size=15)
        show.on_ground(f, lambda cv, v: v.ground_line(horse, (horse[0] + show.fwd * 2, horse[1] + (1 if horse[1] < 4 else -1)),
                                                       GOLD, 0.08, 200, seg(lt, 400, 900), arrow=True))
    show.bubble(f, end, "这边让给你走～", seg(lt, 200, 1100), z=1.4, size=15)


def _horse_opening(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "fierce", 0.0)
    end = show.move.end
    rear = hop(seg(lt, 200, 900))
    m.tilt = -0.45 * rear
    m.z = 0.15 * rear
    show.on_ground(f, lambda cv, v: _sunburst(cv, v, end, lt, u))
    show.word(f, end, "咴儿——!", seg(lt, 300, 1300), z=1.7, size=0.5, color=(255, 240, 180))


def _edge_horse(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "smug", 0.0)
    end = show.move.end
    m.look = _ahead(show, end, 2)
    show.on_ground(f, lambda cv, v: _bamboo_grove(cv, v, end, lt, seg(lt, 0, 700)))
    show.bubble(f, end, "竹林里，好藏身。", seg(lt, 400, 1600), z=1.5, size=15)


def _screen_horses(show, f, e, lt, u, active, dur):
    if not active:
        return
    _opening_base(show, f, e, lt, dur, "cheer")
    horses = sorted(_squares(show, show.side, HORSE), key=lambda sq: sq[1])
    if len(horses) >= 2:
        row = (horses[0][0] + horses[1][0]) / 2 - show.fwd * 0.55
        color = (190, 60, 50) if show.side == RED else (50, 64, 80)
        show.in_air(f, lambda cv, v: _folding_screen(cv, v, row, horses[0][1], horses[-1][1], seg(lt, 200, 800), color))
        for sq in horses:
            sp = show.cast(f, sq, lt, dur, "fierce", wake=250)
            sp.look = (sq[0], 4)
    show.bubble(f, show.move.end, "双马成屏，中路休想！", seg(lt, 700, dur - 80), z=1.5, size=15)


def _cannon_pair(show, f, e, lt, dur, crossed: bool):
    _opening_base(show, f, e, lt, dur, "fierce")
    mine = show.move.end
    theirs = _nearest(show, show.enemy, CANNON, (9 - mine[0], 4))
    if theirs is None:
        return
    rival = show.cast(f, theirs, lt, dur, "fierce", wake=250)
    rival.look = mine
    show.m.look = theirs
    if crossed:
        mid = ((mine[0] + theirs[0]) / 2, 4)
        show.in_air(f, lambda cv, v: _swords(cv, v, mid, seg(lt, 200, dur), lt))
        show.word(f, mid, "铛!", seg(lt, 600, 1300), z=1.6, size=0.5, color=(220, 236, 255))
    else:
        show.in_air(f, lambda cv, v: _lightning(cv, v, mine, theirs, lt) if 300 < lt < dur - 200 else None)
        show.word(f, ((mine[0] + theirs[0]) / 2, 4), "滋滋!", seg(lt, 400, 1300), z=1.3, size=0.45, color=(190, 220, 255))


def _same_cannons(show, f, e, lt, u, active, dur):
    if active:
        _cannon_pair(show, f, e, lt, dur, crossed=False)


def _crossed_cannons(show, f, e, lt, u, active, dur):
    if active:
        _cannon_pair(show, f, e, lt, dur, crossed=True)


def _advisor_opening(show, f, e, lt, u, active, dur):
    if not active:
        return
    _opening_base(show, f, e, lt, dur, "smug", 0.0)
    side = show.side
    show.in_air(f, lambda cv, v: _walls(cv, v, side, seg(lt, 150, 800)))
    king = _king(show, side)
    if king:
        sp = show.cast(f, king, lt, dur, "cheer", wake=300)
        show.bubble(f, king, "宫墙高筑，安心～", seg(lt, 700, dur - 80), z=1.5, size=15)
        sp.z = hop(seg(lt, 800, 1200)) * 0.2


def _rook_opening(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "fierce", 0.0)
    start, end = show.move.start, show.move.end
    far = (end[0] + show.fwd * 2, end[1]) if start[1] == end[1] else (end[0], end[1] + (1 if end[1] > start[1] else -1) * 2)
    far = (max(0, min(9, far[0])), max(0, min(8, far[1])))
    show.on_ground(f, lambda cv, v: _road(v, start, far, seg(lt, 0, 700), lt))
    m.walk = lt / 40.0
    show.bubble(f, end, "车道已开！", seg(lt, 300, 1500), z=1.5, size=15)


def _king_opening(show, f, e, lt, u, active, dur):
    if not active:
        return
    m = _opening_base(show, f, e, lt, dur, "cheer", 0.3)
    end = show.move.end
    color = (206, 46, 40) if show.side == RED else (36, 44, 52)

    def flags(cv, v):
        for dc in (-0.9, 0.9):
            x, y, s = v.project(end[0] - show.fwd * 0.4, end[1] + dc, 0)
            banner(cv, x - (90 if dc < 0 else 0), y, s * 1.5 * ease_out(seg(lt, 0, 500)), color, lt, "御" if dc < 0 else "征")
    show.on_ground(f, flags)

    def party(cv, v):
        x, y, s = v.project(end[0], end[1], 1.5)
        confetti(cv, pygame.Rect(int(x - s * 1.5), int(y - s * 1.2), int(s * 3), int(s * 2.4)), lt, count=30, alpha=220 * (1 - seg(lt, dur - 400, dur)))
    show.on_top(f, party)
    m.swing = 0.5 * hop(seg(lt, 300, 800))
    show.bubble(f, end, "本帅亲自出马！" if show.side == RED else "本将亲自出马！", seg(lt, 300, 1600), z=1.6, size=15)


OPENINGS: dict[str, Painter] = {
    "center_cannon": _center_cannon,
    "cross_palace_cannon": _cross_palace_cannon,
    "advisor_cannon": _advisor_cannon,
    "pawn_base_cannon": _pawn_base_cannon,
    "retreat_cannon": _retreat_cannon,
    "edge_cannon": _edge_cannon,
    "river_cannon": _river_cannon,
    "hook_cannon": _hook_cannon,
    "cross_river_cannon": _cross_river_cannon,
    "immortal_pawn": _immortal_pawn,
    "edge_pawn": _edge_pawn,
    "center_pawn": _center_pawn,
    "flying_elephant": _flying_elephant,
    "edge_elephant": _edge_elephant,
    "horse_opening": _horse_opening,
    "edge_horse": _edge_horse,
    "screen_horses": _screen_horses,
    "same_cannons": _same_cannons,
    "crossed_cannons": _crossed_cannons,
    "advisor_opening": _advisor_opening,
    "rook_opening": _rook_opening,
    "king_opening": _king_opening,
}


# --- tactics ------------------------------------------------------------------------

def _tactic_mood(show) -> tuple[str, tuple[int, int, int]]:
    """Expressions and beam colour shift with the balance of the game for the mover."""
    from cinema.detect import material_balance, tone_for

    tone = tone_for(material_balance(show.before), show.side)
    return tone, {"dominant": (255, 196, 70), "even": (255, 90, 70), "desperate": (255, 60, 160)}[tone]


def _scared_target(show, f, sq, lt, dur, threat, line: str, wake: float = 400) -> None:
    sp = show.cast(f, sq, lt, dur, "scared", wake=wake)
    sp.look = threat
    sp.shake = math.sin(lt / 28.0) * 2.5
    show.in_air(f, lambda cv, v: (lambda x, y, s: sweat_drop(cv, (x + s * 0.3, y), s * 0.12, lt))(*v.project(sq[0], sq[1], 1.0)))
    show.bubble(f, sq, line, seg(lt, wake + 200, wake + 1000), z=1.4, size=15, away=threat, tie_flip=True)


def _fork(show, f, e, lt, u, active, dur):
    if not active:
        return
    tone, color = _tactic_mood(show)
    end = show.move.end
    m = show.cast(f, end, lt, dur, "smug" if tone != "desperate" else "fierce")
    m.aura = color
    m.z = 0.12 * hop(seg(lt, 0, 400))
    for k, sq in enumerate(e.targets[:2]):
        grow = ease_out(seg(lt, 250 + k * 250, 750 + k * 250))
        show.beam(f, end, sq, grow, color, 220, 0.09, arrow=True)
        _scared_target(show, f, sq, lt, dur, end, ("糟了，左边！", "右边也被盯上了！")[k], wake=300 + k * 250)
        show.glow_ring(f, sq, lt / dur, color, 0.5)
    lines = {"dominant": "一网打尽～", "even": "两个都别想跑！", "desperate": "最后的机会——双捉！"}
    show.bubble(f, end, lines[tone], seg(lt, 200, 1100), z=1.55, size=15)
    show.word(f, end, "捉双!", seg(lt, 900, dur - 60), z=1.9, size=0.6, color=color)
    _ribbon(show, f, e, lt, dur, 500)


def _discovered_attack(show, f, e, lt, u, active, dur):
    if not active:
        return
    tone, color = _tactic_mood(show)
    end = show.move.end
    m = show.cast(f, end, lt, dur, "smug")
    m.tilt = 0.15 * math.sin(lt / 120.0) * (1 - seg(lt, 600, 900))
    show.bubble(f, end, "让一让～", seg(lt, 0, 800), z=1.45, size=15)
    for k, (src, dst) in enumerate(e.lines):
        hidden = show.cast(f, src, lt, dur, "fierce", wake=300)
        hidden.look = dst
        glint = seg(lt, 300, 700)
        show.in_air(f, lambda cv, v, s_=src, g=glint: (lambda x, y, s: pygame.draw.polygon(cv, (255, 255, 255), star_points(x + s * 0.25, y, s * 0.3 * hop(g), s * 0.06, 4, g * 3)) if 0 < g < 1 else None)(*v.project(s_[0], s_[1], 1.0)))
        fire = seg(lt, 650, 1000)
        show.beam(f, src, dst, ease_out(fire), color, 230, 0.11, arrow=True)
        show.bolt(f, src, dst, fire)
        _scared_target(show, f, dst, lt, dur, src, "背后有人？！", wake=700)
        show.word(f, src, "闪!", seg(lt, 600, 1300), z=1.6, size=0.55, color=color)
    _ribbon(show, f, e, lt, dur, 700)


def _check_on_king(show, f, e, lt, dur, line: str, start: float = 400) -> Optional[Square]:
    king = e.targets[0] if e.targets else _king(show, show.enemy)
    if king is None:
        return None
    sp = show.cast(f, king, lt, dur, "scared", wake=start)
    sp.shake = math.sin(lt / 24.0) * 3 * seg(lt, start, start + 200)
    sp.look = e.lines[0][0] if e.lines else show.move.end
    show.bubble(f, king, line, seg(lt, start + 200, dur - 100), z=1.5, size=15)
    return king


def _discovered_check(show, f, e, lt, u, active, dur):
    if not active:
        return
    tone, color = _tactic_mood(show)
    end = show.move.end
    m = show.cast(f, end, lt, dur, "smug")
    show.bubble(f, end, "不是我将的哦～", seg(lt, 0, 900), z=1.45, size=15)
    m.z = 0.1 * hop(seg(lt, 0, 500))
    for src, dst in e.lines:
        if src != end:
            sp = show.cast(f, src, lt, dur, "fierce", wake=300)
            sp.look = dst
        show.beam(f, src, dst, ease_out(seg(lt, 450, 850)), color, 230, 0.12, arrow=True)
        show.bolt(f, src, dst, seg(lt, 450, 850))
    _check_on_king(show, f, e, lt, dur, "哪里来的将军？！", 600)
    show.stamp(f, e.targets[0], "闪将", seg(lt, 900, dur), z=2.2, size=0.85)
    _ribbon(show, f, e, lt, dur, 800)


def _double_check(show, f, e, lt, u, active, dur):
    if not active:
        return
    tone, color = _tactic_mood(show)
    for k, (src, dst) in enumerate(e.lines):
        sp = show.cast(f, src, lt, dur, "fierce", wake=200)
        sp.look = dst
        show.beam(f, src, dst, ease_out(seg(lt, 250 + k * 200, 650 + k * 200)), color, 230, 0.12, arrow=True)
        show.bolt(f, src, dst, seg(lt, 250 + k * 200, 650 + k * 200))
    king = _check_on_king(show, f, e, lt, dur, "两边都来？！只能跑了！", 500)
    if king:
        show.in_air(f, lambda cv, v: (lambda x, y, s: dizzy_stars(cv, (x, y), lt, s * 0.35))(*v.project(king[0], king[1], 1.2)) if lt > 800 else None)
        show.stamp(f, king, "双将", seg(lt, 900, dur), z=2.2, size=0.85)
    _ribbon(show, f, e, lt, dur, 800)


def _check_win(show, f, e, lt, u, active, dur):
    if not active:
        return
    tone, color = _tactic_mood(show)
    checks, (capturer, target) = e.lines[:-1], e.lines[-1]
    for src, dst in checks:
        sp = show.cast(f, src, lt, dur, "fierce", wake=200)
        sp.look = dst
        show.beam(f, src, dst, ease_out(seg(lt, 200, 600)), DANGER, 230, 0.12, arrow=True)
    _check_on_king(show, f, e, lt, dur, "先保命要紧……", 400)
    hooker = show.cast(f, capturer, lt, dur, "smug", wake=600)
    hooker.look = target
    pull = seg(lt, 800, 1300)

    def hook_line(cv, v):
        if pull <= 0:
            return
        a = v.point(capturer[0], capturer[1], 0.9)
        b = v.point(target[0], target[1], 0.9)
        mid = ((a[0] + b[0]) / 2, min(a[1], b[1]) - 60)
        pts = []
        for k in range(13):
            t = k / 12 * ease_out(pull)
            pts.append((lerp(lerp(a[0], mid[0], t), lerp(mid[0], b[0], t), t), lerp(lerp(a[1], mid[1], t), lerp(mid[1], b[1], t), t)))
        pygame.draw.lines(cv, INK, False, pts, 6)
        pygame.draw.lines(cv, GOLD, False, pts, 3)
        if pull >= 1:
            r = 14
            pygame.draw.arc(cv, GOLD, pygame.Rect(int(b[0] - r), int(b[1] - 4), 2 * r, 2 * r), math.pi, math.tau, 4)
    show.in_air(f, hook_line)
    _scared_target(show, f, target, lt, dur, capturer, "我……我被抽了？", wake=900)
    side = capturer[1] - target[1] or (-1 if target[1] > 4 else 1)
    seal_at = (target[0], target[1] + math.copysign(0.95, side))
    show.stamp(f, seal_at, "抽", seg(lt, 1200, dur), z=0.9, size=0.7)
    _ribbon(show, f, e, lt, dur, 1000)


TACTICS: dict[str, Painter] = {
    "fork": _fork,
    "discovered_attack": _discovered_attack,
    "discovered_check": _discovered_check,
    "double_check": _double_check,
    "check_win": _check_win,
}


# --- mate / check decor ----------------------------------------------------------------

def _checkers(e) -> list[Square]:
    return [a for a, _ in e.lines]


def _decor_double_rook(show, f, e, lt, dur, king):
    for sq in [s for s in e.cast if show.piece_at(s) is not None and show.piece_at(s).kind == ROOK]:
        sp = show.cast(f, sq, lt, dur, "fierce", wake=200)
        sp.look = king
        sp.walk = lt / 50.0
    show.word(f, king, "错!", seg(lt, 300, 1300 * dur / 1500), z=1.9, size=0.6, color=(255, 200, 90))


def _decor_wocao(show, f, e, lt, dur, king):
    for sq in e.cast:
        p = show.piece_at(sq)
        if p is not None and p.kind == HORSE:
            sp = show.cast(f, sq, lt, dur, "fierce", wake=150)
            sp.tilt = -0.5 * hop((lt / 500.0) % 1.0)
            sp.z = 0.15 * hop((lt / 500.0) % 1.0)
            show.bubble(f, sq, "卧槽在此！", seg(lt, 200, 1300 * dur / 1500), z=1.5, size=15)


def _decor_corner(show, f, e, lt, dur, king):
    for sq in e.cast:
        p = show.piece_at(sq)
        if p is not None and p.kind == HORSE:
            show.glow_ring(f, sq, lt / 1500.0, LIGHT, 0.55)
            show.cast(f, sq, lt, dur, "smug", wake=150).look = king
            show.word(f, sq, "挂角!", seg(lt, 300, 1200 * dur / 1500), z=1.6, size=0.5, color=LIGHT)


def _decor_fishing(show, f, e, lt, dur, king):
    horse = next((sq for sq in e.cast if show.piece_at(sq) is not None and show.piece_at(sq).kind == HORSE), None)
    if horse is None:
        return
    show.cast(f, horse, lt, dur, "smug", wake=150).look = king
    lift = 0.4 * hop(seg(lt, 300, 1300))
    show.in_air(f, lambda cv, v: _fishing(cv, v, horse, king, lift, lt))
    show.bubble(f, horse, "王上钩了～", seg(lt, 400, 1300 * dur / 1500), z=1.5, size=15)


def _decor_cannon_over(show, f, e, lt, dur, king, double: bool):
    cannons = [sq for sq in e.cast if show.piece_at(sq) is not None and show.piece_at(sq).kind == CANNON]
    screens = [sq for sq in e.cast if sq not in cannons and show.piece_at(sq) is not None and show.piece_at(sq).kind != KING]
    for i, sq in enumerate(cannons[:2] if double else cannons[:1]):
        sp = show.cast(f, sq, lt, dur, "fierce", wake=150)
        sp.look = king
        t0 = 250 + i * 350
        sp.squash = 0.2 * (1 - seg(lt, t0, t0 + 200)) * (lt > t0)
        show.shell(f, sq, king, seg(lt, t0, t0 + 500), 1.3 + abs(sq[0] - king[0]) * 0.1)
        show.impact(f, king, seg(lt, t0 + 500, t0 + 900), 0.8, (255, 210, 120), z=0.9)
    for sq in screens:
        sp = show.cast(f, sq, lt, dur, "scared", wake=150)
        sp.squash = 0.35 * seg(lt, 150, 300) * (1 - seg(lt, 1100, 1300))
    if double:
        show.word(f, king, "砰砰!", seg(lt, 700, 1400 * dur / 1500), z=1.9, size=0.55)


def _decor_heaven_earth(show, f, e, lt, dur, king):
    cannons = [sq for sq in e.cast if show.piece_at(sq) is not None and show.piece_at(sq).kind == CANNON]
    for i, sq in enumerate(cannons[:2]):
        show.cast(f, sq, lt, dur, "fierce", wake=150).look = king
        mid = ((sq[0] + king[0]) / 2, (sq[1] + king[1]) / 2)
        label = "天" if abs(sq[0] - king[0]) > abs(sq[1] - king[1]) else "地"
        show.beam(f, sq, king, ease_out(seg(lt, 200 + i * 250, 600 + i * 250)), (160, 200, 255) if label == "天" else (210, 160, 90), 210, 0.12)
        show.on_ground(f, lambda cv, v, m_=mid, l_=label, g=seg(lt, 500 + i * 250, 800 + i * 250): v.ground_text(m_, l_, 0.7, (255, 255, 255), 230 * g))


def _decor_iron_bolt(show, f, e, lt, dur, king):
    defender = show.enemy
    r0, r1 = _palace_rows(defender)
    door = r1 + 0.5 if defender == BLACK else r0 - 0.5
    show.in_air(f, lambda cv, v: _iron_bar(cv, v, door, seg(lt, 200, 700)))
    show.word(f, (door, 4), "咣当!", seg(lt, 600, 1300 * dur / 1500), z=1.0, size=0.5, color=(220, 226, 240))
    show.bubble(f, king, "开门！开门啊！", seg(lt, 700, 1400 * dur / 1500), z=1.5, size=15)


def _decor_heart(show, f, e, lt, dur, king):
    show.in_air(f, lambda cv, v: _heart(cv, v, king, seg(lt, 100, 1400), lt))
    show.word(f, king, "穿心!", seg(lt, 500, 1300 * dur / 1500), z=2.3, size=0.5, color=(255, 120, 140))


def _decor_facing(show, f, e, lt, dur, king):
    mine = _king(show, show.side)
    if mine is None:
        return
    sp = show.cast(f, mine, lt, dur, "smug", wake=150)
    sp.look = king
    show.beam(f, mine, king, ease_out(seg(lt, 200, 600)), (255, 255, 255), 200, 0.08, dashed=True)
    show.bubble(f, mine, "哈哈，看见你了！", seg(lt, 300, 1300 * dur / 1500), z=1.5, size=15)
    f.flash = max(f.flash, 0.5 * (1 - seg(lt, 600, 900)) * (lt > 600))


def _decor_sea_moon(show, f, e, lt, dur, king):
    def sky(cv, v):
        x, y, s = v.project(king[0], king[1], 3.0)
        moon(cv, (x + s * 0.8, y - s * 0.2), s * 0.32)
    show.in_air(f, sky)
    show.in_air(f, lambda cv, v: _net(cv, v, king, seg(lt, 300, 1100)))
    show.word(f, king, "捞!", seg(lt, 900, 1500 * dur / 1500), z=1.9, size=0.55, color=(200, 230, 255))


def _decor_two_ghosts(show, f, e, lt, dur, king):
    pawns = [sq for sq in e.cast if show.piece_at(sq) is not None and show.piece_at(sq).kind == PAWN]
    for i, sq in enumerate(pawns[:2]):
        sp = show.cast(f, sq, lt, dur, "smug", wake=150)
        sp.look = king
        knock = seg((lt + i * 250) % 500, 0, 500)
        sp.swing = hop(knock)
        show.in_air(f, lambda cv, v, s_=sq: (lambda x, y, s: ghost(cv, (x, y), (lt / 1400.0) % 1.0, s * 0.45))(*v.project(s_[0], s_[1], 1.0)))
        show.word(f, sq, "咚!", hop(knock) * 0.9 + 0.05 if lt < 1400 * dur / 1500 else 0, z=1.4, size=0.4, color=(230, 230, 255))


def _decor_generic(show, f, e, lt, dur, king):
    for i, sq in enumerate(_escape_squares(show, king)):
        show.on_ground(f, lambda cv, v, s_=sq, g=seg(lt, 200 + i * 120, 500 + i * 120): _x_mark(v, s_, g))


DECOR: dict[str, Callable] = {
    "double_rook": _decor_double_rook,
    "wocao_horse": _decor_wocao,
    "corner_horse": _decor_corner,
    "fishing_horse": _decor_fishing,
    "horse_cannon": lambda show, f, e, lt, dur, king: _decor_cannon_over(show, f, e, lt, dur, king, False),
    "double_cannon": lambda show, f, e, lt, dur, king: _decor_cannon_over(show, f, e, lt, dur, king, True),
    "heaven_earth": _decor_heaven_earth,
    "iron_bolt": _decor_iron_bolt,
    "heart_strike": _decor_heart,
    "facing_kings": _decor_facing,
    "sea_moon": _decor_sea_moon,
    "two_ghosts": _decor_two_ghosts,
    "generic": _decor_generic,
}

KING_LINES = {
    "double_rook": "两辆车……轮流来？！", "wocao_horse": "槽口怎么有马！", "corner_horse": "宫角……失守了！",
    "fishing_horse": "放、放我下来！", "horse_cannon": "马后面还有炮！", "double_cannon": "两门炮叠在一起？！",
    "heaven_earth": "天上地下都是炮……", "iron_bolt": "门被栓死了！", "heart_strike": "心……心口一凉……",
    "facing_kings": "别、别看我！", "sea_moon": "从海底捞上来了？！", "two_ghosts": "谁在拍门？！",
    "generic": "无路可走了……",
}


def _beams(show, f, e, lt, start: float = 150) -> None:
    for i, (src, dst) in enumerate(e.lines):
        grow = ease_out(seg(lt, start + i * 160, start + 400 + i * 160))
        show.beam(f, src, dst, grow, DANGER, 230, 0.12, arrow=True)
        if 0 < grow < 1:
            show.bolt(f, src, dst, grow)


def mate_painter(key: str) -> Painter:
    decor = DECOR.get(key, _decor_generic)

    def paint(show, f, e, lt, u, active, dur):
        if not active:
            return
        king = e.targets[0] if e.targets else _king(show, show.enemy)
        if king is None:
            return
        _beams(show, f, e, lt)
        decor(show, f, e, lt, 1500, king)
        fall = seg(lt, 1500, 1900)
        sp = show.cast(f, king, lt, dur + 400, "scared" if fall <= 0 else "ko", wake=150)
        if fall <= 0:
            sp.shake = math.sin(lt / 24.0) * 3
            if key == "fishing_horse":
                sp.z = 0.4 * hop(seg(lt, 300, 1300))
            show.bubble(f, king, KING_LINES.get(key, KING_LINES["generic"]), seg(lt, 350, 1450), z=1.55, size=15)
        else:
            sp.props = frozenset({"no_hat"})
            sp.tilt = -1.4 * ease_out(fall)
            sp.squash = 0.15 * hop(fall)
            show.in_air(f, lambda cv, v: _crown(cv, v, king, seg(lt, 1500, 2600)))
            show.impact(f, king, seg(lt, 1500, 2000), 1.3, (255, 220, 120))
            f.shake = max(f.shake, 10 * (1 - seg(lt, 1500, 1850)) * (lt > 1500))
            if lt > 1900:
                show.in_air(f, lambda cv, v: (lambda x, y, s: dizzy_stars(cv, (x, y), lt, s * 0.38))(*v.project(king[0], king[1], 0.75)))
        show.stamp(f, king, "将死", seg(lt, 1750, dur + 600), z=2.3, size=1.0)
        if lt > 1900:
            for sq in dict.fromkeys(list(e.cast) + [show.move.end]):
                if sq == king:
                    continue
                w = show.cast(f, sq, lt, dur, "cheer", wake=1900)
                w.z = hop(((lt - 1900 + sq[1] * 70) % 520) / 520) * 0.35

            def party(cv, v):
                confetti(cv, pygame.Rect(0, 0, 860, 760), lt - 1900, count=80, alpha=255 * (1 - seg(lt, dur - 300, dur)))
            show.on_top(f, party)
        show.ribbon(f, e.title, e.subtitle, seg(lt, 400, dur + 500), sq=king, color=(170, 30, 30))
    return paint


def check_painter(key: str) -> Painter:
    decor = DECOR.get(key) if key != "generic" else None

    def paint(show, f, e, lt, u, active, dur):
        if not active:
            return
        king = e.targets[0] if e.targets else _king(show, show.enemy)
        if king is None:
            return
        _beams(show, f, e, lt, 80)
        if decor is not None:
            decor(show, f, e, lt, dur, king)
        sp = show.cast(f, king, lt, dur, "scared", wake=250)
        sp.shake = math.sin(lt / 26.0) * 2.5 * seg(lt, 250, 450)
        sp.look = e.lines[0][0] if e.lines else show.move.end
        show.word(f, king, "将军!", seg(lt, 300, dur - 50), z=1.8, size=0.6, color=(255, 120, 90))
        show.bubble(f, king, "糟了！", seg(lt, 500, dur - 100), z=1.4, size=15)
        if key != "generic":
            show.ribbon(f, e.title, e.subtitle, seg(lt, 200, dur + 200), sq=king, color=(170, 60, 30))
    return paint


def stalemate(show, f, e, lt, u, active, dur):
    if not active:
        return
    king = e.targets[0] if e.targets else _king(show, show.enemy)
    if king is None:
        return
    show.in_air(f, lambda cv, v: _cage(cv, v, king, seg(lt, 200, 800)))
    for i, sq in enumerate(_escape_squares(show, king) or [king]):
        show.on_ground(f, lambda cv, v, s_=sq, g=seg(lt, 700 + i * 150, 1000 + i * 150): _x_mark(v, s_, g))
    sp = show.cast(f, king, lt, dur, "sad", wake=150)
    sp.shake = math.sin(lt / 40.0) * 1.5
    show.bubble(f, king, "没人将我……可我动不了……", seg(lt, 700, dur - 300), z=1.6, size=15)
    show.word(f, king, "咔嚓!", seg(lt, 700, 1300), z=1.2, size=0.5, color=(220, 226, 240))
    show.stamp(f, king, "困毙", seg(lt, 1500, dur + 600), z=2.4, size=0.95)
    show.ribbon(f, e.title, e.subtitle, seg(lt, 300, dur + 400), sq=king, color=(90, 70, 130))


def sacrifice(show, f, e, lt, u, active, dur):
    if not active:
        return
    end = show.move.end
    m = show.cast(f, end, lt, dur, "fierce")
    m.aura = (255, 110, 60)
    show.in_air(f, lambda cv, v: _heart(cv, v, end, seg(lt, 100, dur), lt))
    king = _king(show, show.enemy)
    if king:
        sp = show.cast(f, king, lt, dur, "surprised", wake=300)
        sp.look = end
        show.bubble(f, king, "车竟然闯进来了！", seg(lt, 500, dur - 100), z=1.5, size=15)
    show.word(f, end, "剜心!", seg(lt, 200, 1100), z=2.2, size=0.55, color=(255, 120, 140))
    _ribbon(show, f, e, lt, dur, 300)
