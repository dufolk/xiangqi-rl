"""2.5D perspective board: camera, wooden board, wooden pucks and the characters they turn into.

Board coordinates are game ``(row, col)`` floats plus a height ``z`` in cells. The camera is an
off-axis pinhole looking across the table, so vertical things stay vertical (billboards) while the
board recedes into the distance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import pygame

from cinema.actors import INK, blit_actor, render_actor
from cinema.fonts import font
from cinema.fx import cloud, ease_back, lerp, mountains, seg, with_alpha
from game_engine import BLACK, RED, Piece

CX = 430.0
F = 2394.0
D0 = 28.5
HC = 23.4
HY = -1265.0
MARGIN = 0.78
THICK = 0.42
TEX_S = 112
VIEW_W = 860

WOOD_TOP = (226, 182, 122)
WOOD_DARK = (168, 116, 66)
WOOD_SIDE = (120, 74, 40)
GRID_INK = (74, 42, 22)
PUCK_FACE = (240, 214, 168)
PUCK_SIDE = (186, 136, 82)
GLYPH = {RED: (186, 32, 30), BLACK: (32, 36, 34)}
RING = {RED: (196, 52, 44), BLACK: (40, 52, 48)}

ACTOR_R = 0.38
PUCK_R = 0.43
PUCK_H = 0.17
HORIZON = 196
SUN = (640, 150)


@dataclass
class Sprite:
    """A piece drawn by a show instead of the static board: a puck, a character, or halfway between."""

    piece: Piece
    row: float
    col: float
    z: float = 0.0
    awake: float = 1.0
    expr: str = "idle"
    facing: int = 0
    walk: float = 0.0
    swing: float = 0.0
    squash: float = 0.0
    tilt: float = 0.0
    alpha: int = 255
    flash: float = 0.0
    props: frozenset = frozenset()
    shake: float = 0.0
    scale: float = 1.0
    aura: Optional[tuple[int, int, int]] = None
    shadow: bool = True
    look: Optional[tuple[float, float]] = None


@dataclass
class Frame:
    """Everything a show wants on screen this instant, layered by depth."""

    hidden: set = field(default_factory=set)
    sprites: list = field(default_factory=list)
    ground: list = field(default_factory=list)
    air: list = field(default_factory=list)
    overlay: list = field(default_factory=list)
    hud: list = field(default_factory=list)  # screen-space, drawn after the camera push
    shake: float = 0.0
    flash: float = 0.0
    now: float = 0.0
    zoom: float = 1.0
    focus: Optional[tuple[float, float]] = None

    def add(self, sprite: Sprite, hide: Optional[tuple[int, int]] = None) -> Sprite:
        self.sprites.append(sprite)
        if hide is not None:
            self.hidden.add(hide)
        return sprite


def _mix(a, b, u: float) -> tuple[int, int, int]:
    return tuple(int(lerp(a[i], b[i], u)) for i in range(3))


@lru_cache(maxsize=8)
def _wood(w: int, h: int, seed: int, base: tuple, dark: tuple, streak: int = 3) -> pygame.Surface:
    rng = random.Random(seed)
    surf = pygame.Surface((w, h))
    surf.fill(base)
    for i in range(0, h, streak):
        y = i + rng.random() * streak
        color = _mix(base, dark, rng.random() ** 2 * 0.55)
        freq = 70 + rng.random() * 120
        amp = 1.5 + rng.random() * 3.5
        phase = rng.random() * 6.28
        pts = [(x, y + math.sin(x / freq + phase) * amp + math.sin(x / 23.0 + i) * 0.6) for x in range(-20, w + 40, 18)]
        pygame.draw.lines(surf, color, False, pts, 1 + (rng.random() < 0.25))
    for _ in range(max(1, w * h // 160000)):
        kx, ky = rng.random() * w, rng.random() * h
        for k in range(6, 0, -1):
            pygame.draw.ellipse(surf, _mix(base, dark, 0.15 + k * 0.06), (kx - k * 7, ky - k * 2.2, k * 14, k * 4.4), 1)
    return surf


class BoardView:
    def __init__(self) -> None:
        self.flipped = False
        self.offset = (0.0, 0.0)
        self._board: dict[bool, tuple[pygame.Surface, tuple[int, int]]] = {}
        self._pucks: dict[tuple, pygame.Surface] = {}
        self._backdrops: dict[int, pygame.Surface] = {}
        self._layer = pygame.Surface((VIEW_W, 900), pygame.SRCALPHA)
        self._layer_dirty = False

    # --- camera ------------------------------------------------------------------
    def disp(self, row: float, col: float) -> tuple[float, float]:
        return (9 - row, 8 - col) if self.flipped else (row, col)

    def _proj(self, dr: float, dc: float, z: float) -> tuple[float, float, float]:
        d = D0 + (9 - dr)
        return CX + F * (dc - 4) / d + self.offset[0], HY + F * (HC - z) / d + self.offset[1], F / d

    def project(self, row: float, col: float, z: float = 0.0) -> tuple[float, float, float]:
        dr, dc = self.disp(row, col)
        return self._proj(dr, dc, z)

    def point(self, row: float, col: float, z: float = 0.0) -> tuple[float, float]:
        x, y, _ = self.project(row, col, z)
        return x, y

    def fore(self, row: float) -> float:
        dr, _ = self.disp(row, 0)
        return HC / (D0 + (9 - dr))

    def pick(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        best, best_d = None, 1e9
        for row in range(10):
            for col in range(9):
                x, y, s = self.project(row, col, PUCK_H * 0.6)
                d = math.hypot((pos[0] - x) / s, (pos[1] - y) / (s * self.fore(row)))
                if d < best_d:
                    best, best_d = (row, col), d
        return best if best_d < 0.62 else None

    # --- board ---------------------------------------------------------------------
    def _texture(self) -> pygame.Surface:
        m = int(MARGIN * TEX_S)
        w, h = 8 * TEX_S + 2 * m, 9 * TEX_S + 2 * m
        tex = _wood(w, h, 11, WOOD_TOP, WOOD_DARK).copy()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        for k in range(14):
            pygame.draw.rect(shade, (90, 50, 20, 7), (k * 5, k * 5, w - k * 10, h - k * 10), 6, border_radius=10)
        tex.blit(shade, (0, 0))

        def g(col: float, row: float) -> tuple[float, float]:
            return m + col * TEX_S, m + row * TEX_S

        pad = int(TEX_S * 0.12)
        outer = pygame.Rect(m - pad, m - pad, 8 * TEX_S + 2 * pad, 9 * TEX_S + 2 * pad)
        pygame.draw.rect(tex, GRID_INK, outer, 6)
        lw = 3
        for row in range(10):
            pygame.draw.line(tex, GRID_INK, g(0, row), g(8, row), lw)
        for col in range(9):
            if col in (0, 8):
                pygame.draw.line(tex, GRID_INK, g(col, 0), g(col, 9), lw)
            else:
                pygame.draw.line(tex, GRID_INK, g(col, 0), g(col, 4), lw)
                pygame.draw.line(tex, GRID_INK, g(col, 5), g(col, 9), lw)
        for top in (0, 7):
            pygame.draw.line(tex, GRID_INK, g(3, top), g(5, top + 2), lw)
            pygame.draw.line(tex, GRID_INK, g(5, top), g(3, top + 2), lw)
        marks = [(2, 1), (2, 7), (7, 1), (7, 7)] + [(r, c) for r in (3, 6) for c in range(0, 9, 2)]
        for row, col in marks:
            x, y = g(col, row)
            a, b = int(TEX_S * 0.07), int(TEX_S * 0.18)
            for sx in (-1, 1):
                if (col == 0 and sx < 0) or (col == 8 and sx > 0):
                    continue
                for sy in (-1, 1):
                    pygame.draw.lines(tex, GRID_INK, False, [(x + sx * b, y + sy * a), (x + sx * a, y + sy * a), (x + sx * a, y + sy * b)], 3)
        river = font(int(TEX_S * 0.6), True)
        for text, cx in (("楚  河", 2.0), ("漢  界", 6.0)):
            img = river.render(text, True, (96, 54, 26))
            tex.blit(img, img.get_rect(center=g(cx, 4.5)))
        small = font(int(TEX_S * 0.26), True)
        bottom = "九八七六五四三二一" if not self.flipped else "987654321"
        top = "123456789" if not self.flipped else "一二三四五六七八九"
        for col in range(9):
            for text, row in ((bottom[col], 9 + MARGIN * 0.62), (top[col], -MARGIN * 0.62)):
                img = small.render(text, True, (110, 66, 32))
                tex.blit(img, img.get_rect(center=g(col, row)))
        return tex

    def _build_board(self) -> tuple[pygame.Surface, tuple[int, int]]:
        tex = self._texture()
        tw, th = tex.get_size()
        d_far, d_near = D0 + 9 + MARGIN, D0 - MARGIN
        y_top = int(HY + F * HC / d_far)
        y_bot = int(HY + F * HC / d_near)
        face_bot = int(HY + F * (HC + THICK) / d_near)
        left = int(CX - F * (4 + MARGIN) / d_near) - 30
        surf = pygame.Surface((VIEW_W - left + 30, face_bot - y_top + 60), pygame.SRCALPHA)
        ox, oy = left, y_top - 10

        far_l = (CX - F * (4 + MARGIN) / d_far, y_top)
        far_r = (CX + F * (4 + MARGIN) / d_far, y_top)
        near_l = (CX - F * (4 + MARGIN) / d_near, y_bot)
        near_r = (CX + F * (4 + MARGIN) / d_near, y_bot)
        bot_l = (near_l[0], face_bot)
        bot_r = (near_r[0], face_bot)
        for k in range(10, 0, -1):
            spread = k * 3.2
            pts = [(far_l[0] - spread * 0.6 + 18 - ox, far_l[1] + 22 - oy), (far_r[0] + spread * 0.6 + 18 - ox, far_r[1] + 22 - oy),
                   (bot_r[0] + spread + 26 - ox, bot_r[1] + spread * 0.8 + 18 - oy), (bot_l[0] - spread + 26 - ox, bot_l[1] + spread * 0.8 + 18 - oy)]
            pygame.draw.polygon(surf, (12, 6, 2, 16), pts)

        for y in range(y_top, y_bot + 1):
            d = F * HC / (y + 0.5 - HY)
            dr = 9 - (d - D0)
            v = int((dr + MARGIN) * TEX_S)
            if not 0 <= v < th:
                continue
            half = F * (4 + MARGIN) / d
            width = max(1, int(round(2 * half)))
            strip = pygame.transform.smoothscale(tex.subsurface((0, v, tw, 1)), (width, 1))
            surf.blit(strip, (int(round(CX - half)) - ox, y - oy))

        side = _wood(int(near_r[0] - near_l[0]) + 2, max(2, face_bot - y_bot + 2), 5, WOOD_SIDE, (78, 44, 22), 2)
        surf.blit(side, (int(near_l[0]) - ox, y_bot - oy))
        pygame.draw.line(surf, (255, 226, 170), (near_l[0] - ox, y_bot - oy), (near_r[0] - ox, y_bot - oy), 2)
        pygame.draw.line(surf, (60, 32, 14), (bot_l[0] - ox, face_bot - oy), (bot_r[0] - ox, face_bot - oy), 3)
        rim = [(far_l[0] - ox, far_l[1] - oy), (far_r[0] - ox, far_r[1] - oy), (near_r[0] - ox, near_r[1] - oy),
               (bot_r[0] - ox, bot_r[1] - oy), (bot_l[0] - ox, bot_l[1] - oy), (near_l[0] - ox, near_l[1] - oy)]
        pygame.draw.polygon(surf, (58, 30, 12), rim, 2)
        return surf, (ox, oy)

    def draw_board(self, canvas: pygame.Surface) -> None:
        if self.flipped not in self._board:
            self._board[self.flipped] = self._build_board()
        surf, (ox, oy) = self._board[self.flipped]
        canvas.blit(surf, (ox + self.offset[0], oy + self.offset[1]))

    # --- scenery -------------------------------------------------------------------
    def _static_backdrop(self, level: int) -> pygame.Surface:
        cached = self._backdrops.get(level)
        if cached is not None:
            return cached
        mood = level / 10
        storm, dawn = max(0.0, -mood), max(0.0, mood)
        surf = pygame.Surface((1440, 900))
        top = _mix(_mix((46, 52, 92), (34, 30, 46), storm), (236, 132, 84), dawn)
        low = _mix(_mix((232, 160, 120), (96, 98, 124), storm), (255, 222, 150), dawn)
        for y in range(0, HORIZON, 4):
            pygame.draw.rect(surf, _mix(top, low, y / HORIZON), (0, y, 1440, 4))
        if dawn > 0.05:
            glow = pygame.Surface((420, 300), pygame.SRCALPHA)
            for k in range(8, 0, -1):
                pygame.draw.circle(glow, (255, 236, 170, int(20 * dawn)), (210, 150), int(40 + k * 18))
            surf.blit(glow, (SUN[0] - 210, SUN[1] - 150))
            pygame.draw.circle(surf, (255, 214, 120), SUN, int(28 + 14 * dawn))
        else:
            pygame.draw.circle(surf, _mix((255, 244, 214), (200, 205, 220), storm), (SUN[0], 92), 26)
        far = _mix(_mix((92, 88, 132), (40, 40, 58), storm), (196, 120, 110), dawn)
        near = _mix(_mix((60, 54, 96), (26, 26, 38), storm), (160, 86, 82), dawn)
        mountains(surf, pygame.Rect(-40, 70, 1520, HORIZON - 60), far, seed=3, peaks=9, height=0.8, outline=False)
        mountains(surf, pygame.Rect(-60, 110, 1560, HORIZON - 100), near, seed=8, peaks=12, height=0.75, outline=False)
        surf.blit(_wood(1440, 900 - HORIZON, 21, (70, 40, 24), (40, 22, 12), 4), (0, HORIZON))
        pygame.draw.line(surf, (30, 16, 8), (0, HORIZON), (1440, HORIZON), 3)
        light = pygame.Surface((VIEW_W + 200, 820), pygame.SRCALPHA)
        for k in range(10, 0, -1):
            pygame.draw.ellipse(light, (255, 210, 150, 7), (100 - k * 20, 220 - k * 16, VIEW_W + k * 40 - 200, 560 + k * 32))
        surf.blit(light, (-100, 0))
        if len(self._backdrops) > 4:
            self._backdrops.pop(next(iter(self._backdrops)))
        self._backdrops[level] = surf
        return surf

    def draw_backdrop(self, canvas: pygame.Surface, now: float, mood: float) -> None:
        """Sky and table. ``mood`` runs from -1 (black crushing) through 0 (balanced) to 1 (red crushing)."""
        level = int(round(max(-1.0, min(1.0, mood)) * 10))
        canvas.blit(self._static_backdrop(level), (0, 0))
        storm, dawn = max(0.0, -level / 10), max(0.0, level / 10)
        if dawn > 0.05:
            rays = pygame.Surface((1440, HORIZON), pygame.SRCALPHA)
            sx, sy = SUN
            for k in range(12):
                a = k / 12 * math.tau + now / 9000.0
                pygame.draw.polygon(rays, (255, 240, 190, int(34 * dawn)), [
                    (sx + math.cos(a) * 60, sy + math.sin(a) * 60),
                    (sx + math.cos(a - 0.05) * 260, sy + math.sin(a - 0.05) * 260),
                    (sx + math.cos(a + 0.05) * 260, sy + math.sin(a + 0.05) * 260)])
            canvas.blit(rays, (0, 0))
        cloud_col = _mix(_mix((250, 238, 236), (120, 120, 140), storm), (255, 240, 220), dawn)
        for k in range(5):
            x = (k * 337 + now * (0.006 + k * 0.002) * (1 + storm * 2)) % 1700 - 130
            cloud(canvas, x, 40 + (k * 53) % 90, 26 + k * 4, cloud_col, 150 + 60 * storm, outline=False)
        if storm > 0.55:
            flash_t = (now % 5200) / 5200
            if flash_t < 0.04:
                bolt = [(300 + k * 18 * (-1) ** k, k * 28) for k in range(7)]
                pygame.draw.lines(canvas, (240, 240, 255), False, bolt, 3)
                sheet = pygame.Surface((1440, HORIZON), pygame.SRCALPHA)
                sheet.fill((230, 230, 255, int(80 * (1 - flash_t / 0.04))))
                canvas.blit(sheet, (0, 0))

    def draw_flag(self, canvas: pygame.Surface, side: str, now: float, morale: float) -> None:
        """A banner beside the board; it rises and flies proudly when ``side`` is winning (morale -1..1)."""
        near_side = (side == RED) != self.flipped
        dr, dc = (-0.95, -0.95) if near_side else (-0.95, 8.95)
        x, y, s = self._proj(dr, dc, 0)
        height = (1.75 + 0.55 * morale) * s
        top = y - height
        pygame.draw.line(canvas, INK, (x, y), (x, top), max(3, int(s * 0.07)))
        pygame.draw.line(canvas, (150, 104, 60), (x, y), (x, top), max(1, int(s * 0.04)))
        pygame.draw.circle(canvas, (250, 198, 92), (int(x), int(top)), max(3, int(s * 0.07)))
        width = s * 1.05
        drop = max(0.0, -morale)
        wave = 7 * (0.4 + 0.6 * (1 - drop))
        color = {RED: (206, 46, 40), BLACK: (36, 44, 52)}[side]
        color = _mix(color, (120, 110, 100), drop * 0.5)
        flag_h = s * 0.62
        direction = -1 if near_side else 1
        pts_top, pts_bot = [], []
        for k in range(9):
            u = k / 8
            sway = math.sin(now / 180.0 + u * 5) * wave * u
            sag = drop * u * flag_h * 0.9
            fx_ = x + direction * width * u * (1 - drop * 0.35)
            pts_top.append((fx_, top + 4 + sway + sag))
            pts_bot.append((fx_, top + 4 + flag_h + sway + sag * 1.2 - (u * 10 if drop > 0.4 and k % 2 else 0)))
        poly = pts_top + pts_bot[::-1]
        pygame.draw.polygon(canvas, color, poly)
        pygame.draw.polygon(canvas, INK, poly, 2)
        glyph = font(int(s * 0.42), True).render("帥" if side == RED else "將", True, (255, 236, 200))
        mid = pts_top[4][0], (pts_top[4][1] + pts_bot[4][1]) / 2
        canvas.blit(glyph, glyph.get_rect(center=(int(mid[0]), int(mid[1]))))

    # --- ground helpers (drawn on a shared alpha layer) -----------------------------
    def _alpha(self) -> pygame.Surface:
        if not self._layer_dirty:
            self._layer.fill((0, 0, 0, 0))
            self._layer_dirty = True
        return self._layer

    def flush(self, canvas: pygame.Surface) -> None:
        if self._layer_dirty:
            canvas.blit(self._layer, (0, 0))
            self._layer_dirty = False

    def ground_ring(self, sq, radius: float, color, width: int = 3, alpha: float = 255, z: float = 0.0) -> None:
        x, y, s = self.project(sq[0], sq[1], z)
        fs = self.fore(sq[0])
        rect = pygame.Rect(0, 0, max(2, int(radius * s * 2)), max(2, int(radius * s * 2 * fs)))
        rect.center = (int(x), int(y))
        pygame.draw.ellipse(self._alpha(), with_alpha(color, alpha), rect, width)

    def ground_line(self, a, b, color, width: float = 0.08, alpha: float = 255, progress: float = 1.0,
                    dashed: bool = False, z: float = 0.02, arrow: bool = False) -> None:
        if progress <= 0:
            return
        layer = self._alpha()
        steps = max(2, int(abs(a[0] - b[0]) + abs(a[1] - b[1])) * 4)
        pts = []
        for k in range(steps + 1):
            u = k / steps * progress
            pts.append(self.project(lerp(a[0], b[0], u), lerp(a[1], b[1], u), z))
        for k in range(len(pts) - 1):
            if dashed and k % 2:
                continue
            (x0, y0, s0), (x1, y1, _) = pts[k], pts[k + 1]
            pygame.draw.line(layer, with_alpha(color, alpha), (x0, y0), (x1, y1), max(2, int(width * s0)))
        if arrow and len(pts) >= 2:
            (x0, y0, _), (x1, y1, s1) = pts[-2], pts[-1]
            ang = math.atan2(y1 - y0, x1 - x0)
            size = s1 * 0.32
            head = [(x1 + math.cos(ang) * size * 0.6, y1 + math.sin(ang) * size * 0.6),
                    (x1 + math.cos(ang + 2.5) * size, y1 + math.sin(ang + 2.5) * size),
                    (x1 + math.cos(ang - 2.5) * size, y1 + math.sin(ang - 2.5) * size)]
            pygame.draw.polygon(layer, with_alpha(color, alpha), head)

    def ground_poly(self, squares, color, alpha: float = 120, z: float = 0.01, width: int = 0) -> None:
        pts = [self.point(r, c, z) for r, c in squares]
        if len(pts) >= 3:
            pygame.draw.polygon(self._alpha(), with_alpha(color, alpha), pts, width)

    def ground_text(self, sq, text: str, size: float, color, alpha: float = 255) -> None:
        x, y, s = self.project(sq[0], sq[1], 0.0)
        img = font(int(size * s), True).render(text, True, color)
        img = pygame.transform.smoothscale(img, (img.get_width(), max(1, int(img.get_height() * self.fore(sq[0])))))
        img.set_alpha(int(alpha))
        self._alpha().blit(img, img.get_rect(center=(int(x), int(y))))

    # --- pieces ----------------------------------------------------------------------
    def puck(self, piece: Piece, s: float, fs: float, glow: Optional[tuple] = None) -> pygame.Surface:
        key = (piece.side, piece.kind, int(s), int(fs * 24), glow)
        cached = self._pucks.get(key)
        if cached is not None:
            return cached
        r = PUCK_R * int(s)
        ry = r * (int(fs * 24) / 24)
        th = PUCK_H * int(s)
        pad = 8
        w, h = int(2 * r + pad * 2 + s * 0.12), int(2 * ry + th + pad * 2 + s * 0.08)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        cx = pad + r
        top_y = pad + ry
        bot_y = top_y + th
        shadow = pygame.Rect(0, 0, int(2 * r * 1.06), int(2 * ry * 1.06))
        shadow.center = (int(cx + s * 0.07), int(bot_y + s * 0.045))
        pygame.draw.ellipse(surf, (20, 10, 4, 110), shadow)
        if glow:
            for k in range(3, 0, -1):
                ring = pygame.Rect(0, 0, int(2 * r + k * 6), int(2 * ry + k * 6 * ry / r))
                ring.center = (int(cx), int(bot_y))
                pygame.draw.ellipse(surf, (*glow, 50), ring)
        body = pygame.Rect(int(cx - r), int(top_y - ry), int(2 * r), int(2 * ry))
        lower = body.move(0, int(th))
        pygame.draw.ellipse(surf, _mix(PUCK_SIDE, (60, 34, 14), 0.45), lower)
        pygame.draw.rect(surf, PUCK_SIDE, (body.x, int(top_y), body.w, int(th)))
        for k in range(int(r * 2)):
            u = k / (r * 2)
            shade = _mix(PUCK_SIDE, (98, 60, 30), abs(u - 0.38) * 1.3)
            pygame.draw.line(surf, shade, (body.x + k, int(top_y)), (body.x + k, int(top_y + th + ry * math.sqrt(max(0.0, 1 - (2 * u - 1) ** 2)))), 1)
        stripe_y = top_y + th * 0.55
        pygame.draw.arc(surf, RING[piece.side], pygame.Rect(body.x, int(stripe_y - ry), body.w, int(2 * ry)), math.pi * 1.02, math.pi * 1.98, max(1, int(s * 0.03)))
        pygame.draw.ellipse(surf, (90, 54, 24), lower, 1)
        pygame.draw.ellipse(surf, PUCK_FACE, body)
        inner = body.inflate(-int(r * 0.12), -int(ry * 0.12))
        pygame.draw.ellipse(surf, _mix(PUCK_FACE, (255, 236, 200), 0.5), inner.move(-int(r * 0.05), -int(ry * 0.05)))
        ring = body.inflate(-int(r * 0.3), -int(ry * 0.3))
        pygame.draw.ellipse(surf, RING[piece.side], ring, max(1, int(s * 0.028)))
        glyph = font(int(r * 1.15), True).render(piece.name, True, GLYPH[piece.side])
        emboss = font(int(r * 1.15), True).render(piece.name, True, (255, 240, 210))
        gh = max(1, int(glyph.get_height() * ry / r))
        glyph = pygame.transform.smoothscale(glyph, (glyph.get_width(), gh))
        emboss = pygame.transform.smoothscale(emboss, (emboss.get_width(), gh))
        surf.blit(emboss, emboss.get_rect(center=(int(cx + 1), int(top_y + 1))))
        surf.blit(glyph, glyph.get_rect(center=(int(cx), int(top_y))))
        pygame.draw.ellipse(surf, (120, 76, 36), body, 1)
        pygame.draw.arc(surf, (255, 250, 230), body.inflate(-3, -3), math.pi * 0.55, math.pi * 0.95, 2)
        surf = surf.convert_alpha() if pygame.display.get_surface() else surf
        self._pucks[key] = surf
        return surf

    def draw_puck(self, canvas: pygame.Surface, piece: Piece, row: float, col: float, z: float = 0.0,
                  alpha: int = 255, glow: Optional[tuple] = None, squash: float = 0.0) -> None:
        x, y, s = self.project(row, col, 0)
        fs = self.fore(row)
        surf = self.puck(piece, s, fs, glow)
        r = PUCK_R * int(s)
        ry = r * (int(fs * 24) / 24)
        pad = 8
        bottom = pad + ry + PUCK_H * int(s)
        if squash:
            w, h = surf.get_size()
            surf = pygame.transform.smoothscale(surf, (max(1, int(w * (1 + squash * 0.3))), max(1, int(h * (1 - squash * 0.4)))))
            bottom *= 1 - squash * 0.4
            r *= 1 + squash * 0.3
        if alpha < 255:
            surf = surf.copy()
            surf.set_alpha(alpha)
        canvas.blit(surf, (int(x - pad - r), int(y - bottom - z * s)))

    def actor_radius(self, row: float) -> float:
        return ACTOR_R * self.project(row, 4, 0)[2]

    def draw_actor(self, canvas: pygame.Surface, sp: Sprite, now: float) -> None:
        x, y, s = self.project(sp.row, sp.col, 0)
        if sp.awake < 0.42:
            u = sp.awake / 0.42
            self.draw_puck(canvas, sp.piece, sp.row, sp.col, z=sp.z + 0.3 * math.sin(u * math.pi * 0.5),
                           alpha=sp.alpha, squash=-0.4 * u)
            return
        R = ACTOR_R * s * sp.scale
        lift = sp.z * s
        x += sp.shake
        if sp.shadow and sp.alpha > 40:
            scale = max(0.3, 1 - sp.z / 3.2)
            sw, sh = int(R * 2.4 * scale), max(2, int(R * 2.4 * scale * self.fore(sp.row) * 0.55))
            shadow = pygame.Surface((sw, sh), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (16, 8, 2, int(120 * scale * sp.alpha / 255)), shadow.get_rect())
            canvas.blit(shadow, (int(x - sw / 2), int(y - sh / 2)))
        if sp.aura is not None:
            self._aura(canvas, x, y - lift, R, sp.aura, now)
        pop = ease_back(seg(sp.awake, 0.42, 1.0))
        squash = sp.squash + (1 - pop) * 0.45
        facing = sp.facing or self.default_facing(sp)
        surf, spec = render_actor(sp.piece.kind, sp.piece.side, R, sp.expr, now, sp.walk, sp.swing, set(sp.props))
        blit_actor(canvas, surf, spec, (x, y - lift), facing, squash, sp.tilt, sp.alpha, sp.flash)
        if sp.awake < 0.75:
            puff = seg(sp.awake, 0.42, 0.75)
            for k in range(6):
                a = k / 6 * math.tau
                pr = R * (0.3 + 0.9 * puff)
                pygame.draw.circle(canvas, (255, 246, 226), (int(x + math.cos(a) * pr * 1.3), int(y - R * 0.5 + math.sin(a) * pr * 0.5)), max(1, int(R * 0.22 * (1 - puff))))

    def default_facing(self, sp: Sprite) -> int:
        if sp.look is not None:
            tx = self.project(sp.look[0], sp.look[1])[0]
            here = self.project(sp.row, sp.col)[0]
            if abs(tx - here) > 2:
                return 1 if tx > here else -1
        _, dc = self.disp(sp.row, sp.col)
        if abs(dc - 4) > 0.3:
            return 1 if dc < 4 else -1
        return 1 if (sp.piece.side == RED) != self.flipped else -1

    def _aura(self, canvas: pygame.Surface, x: float, y: float, R: float, color, now: float) -> None:
        layer = pygame.Surface((int(R * 5), int(R * 6)), pygame.SRCALPHA)
        cx, base = layer.get_width() / 2, layer.get_height() - R * 0.6
        for k in range(9):
            phase = (now / 140.0 + k * 0.7) % 1.0
            fx_ = cx + math.sin(k * 2.3) * R * 1.1
            fy = base - phase * R * 3.6
            size = R * 0.5 * (1 - phase)
            pygame.draw.polygon(layer, (*color, int(170 * (1 - phase))),
                                [(fx_, fy - size * 1.8), (fx_ + size, fy), (fx_, fy + size * 0.6), (fx_ - size, fy)])
        canvas.blit(layer, (int(x - cx), int(y - base)))

    # --- full frame ------------------------------------------------------------------
    def draw_pieces(self, canvas: pygame.Surface, board, frame: Optional[Frame], now: float,
                    lifted: Optional[tuple[int, int]] = None) -> None:
        items: list[tuple[float, int, Callable[[], None]]] = []
        hidden = frame.hidden if frame else set()
        order = 0
        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece is None or (row, col) in hidden:
                    continue
                y = self.project(row, col)[1]
                z = 0.12 + 0.05 * math.sin(now / 160.0) if lifted == (row, col) else 0.0
                items.append((y, order, lambda p=piece, r=row, c=col, zz=z: self.draw_puck(canvas, p, r, c, zz)))
                order += 1
        for sp in (frame.sprites if frame else []):
            y = self.project(sp.row, sp.col)[1]
            items.append((y + 0.5, order, lambda s=sp: self.draw_actor(canvas, s, now)))
            order += 1
        items.sort(key=lambda item: (item[0], item[1]))
        for _, _, draw in items:
            draw()

    def push_in(self, canvas: pygame.Surface, focus: tuple[float, float], zoom: float) -> None:
        """Camera dolly: magnify the board view around ``focus`` (board coordinates)."""
        fx, fy = self.point(focus[0], focus[1], 0.5)
        h = canvas.get_height()
        w, hh = VIEW_W / zoom, h / zoom
        x = max(0.0, min(VIEW_W - w, fx - w / 2))
        y = max(0.0, min(h - hh, fy - hh * 0.56))
        crop = canvas.subsurface(pygame.Rect(int(x), int(y), int(w), int(hh)))
        canvas.blit(pygame.transform.smoothscale(crop, (VIEW_W, h)), (0, 0))

    def compose(self, canvas: pygame.Surface, board, frame: Optional[Frame], now: float, mood: float = 0.0,
                morale: tuple[float, float] = (0.0, 0.0), lifted: Optional[tuple[int, int]] = None,
                ground: Optional[Callable[[pygame.Surface, "BoardView"], None]] = None) -> None:
        """One full frame: sky, flags, board, ground marks, pieces and characters, air and overlay effects."""
        self.draw_backdrop(canvas, now, mood)
        shake = frame.shake if frame else 0.0
        self.offset = (math.sin(now / 23.0) * shake, math.cos(now / 31.0) * shake * 0.6) if shake else (0.0, 0.0)
        self.draw_flag(canvas, BLACK if not self.flipped else RED, now, morale[1] if not self.flipped else morale[0])
        self.draw_flag(canvas, RED if not self.flipped else BLACK, now, morale[0] if not self.flipped else morale[1])
        self.draw_board(canvas)
        if ground is not None:
            ground(canvas, self)
        for fn in (frame.ground if frame else ()):
            fn(canvas, self)
        self.flush(canvas)
        self.draw_pieces(canvas, board, frame, now, lifted)
        for fn in (frame.air if frame else ()):
            fn(canvas, self)
            self.flush(canvas)
        for fn in (frame.overlay if frame else ()):
            fn(canvas, self)
            self.flush(canvas)
        self.offset = (0.0, 0.0)
        if frame and frame.zoom > 1.01 and frame.focus is not None:
            self.push_in(canvas, frame.focus, frame.zoom)
        for fn in (frame.hud if frame else ()):
            fn(canvas, self)
        if frame and frame.flash > 0.01:
            sheet = pygame.Surface((VIEW_W, canvas.get_height()))
            sheet.fill((255, 255, 255))
            sheet.set_alpha(int(255 * min(1.0, frame.flash)))
            canvas.blit(sheet, (0, 0))
