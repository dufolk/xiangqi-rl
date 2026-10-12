"""On-board character animation: every piece walks, gallops, leaps or charges in its own way."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pygame

from game_engine import ADVISOR, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK, Move, Piece
from cinema import fx
from cinema.actors import actor

BOARD_R = 30

MOVE_MS = {ROOK: 620, HORSE: 760, ELEPHANT: 820, ADVISOR: 600, KING: 640, CANNON: 700, PAWN: 620}
CAPTURE_EXTRA = 520
STRIKE_WORD = {ROOK: "轰!", HORSE: "踢!", CANNON: "砰!", ELEPHANT: "咚!", ADVISOR: "唰!", KING: "斩!", PAWN: "戳!"}


@dataclass
class BoardMove:
    """A single animated move; ``draw`` is called every frame until :attr:`done`."""

    move: Move
    piece: Piece
    started_at: int
    check: bool = False
    crossed_river: bool = False
    duration: int = field(init=False)

    def __post_init__(self) -> None:
        self.duration = MOVE_MS[self.piece.kind] + (CAPTURE_EXTRA if self.move.captured else 0)
        if self.check:
            self.duration += 450
        if self.crossed_river:
            self.duration += 250

    def progress(self, now: int) -> float:
        return fx.clamp((now - self.started_at) / self.duration)

    def done(self, now: int) -> bool:
        return now - self.started_at >= self.duration

    # --- path shaping ---------------------------------------------------------
    def _path(self, u: float, s, e, square_px: Callable) -> tuple[float, float, float, float, float]:
        """(x, y, lift, squash, tilt) for travel progress ``u`` (0..1)."""
        kind = self.piece.kind
        x, y = fx.lerp2(s, e, u)
        lift = squash = tilt = 0.0
        if kind == ROOK:
            pre = fx.seg(u, 0, 0.18)
            dash = fx.seg(u, 0.18, 0.85)
            settle = fx.seg(u, 0.85, 1.0)
            x, y = fx.lerp2(s, e, fx.ease_in(dash) * 1.04 - 0.04 * fx.ease_out(settle))
            squash = 0.2 * fx.hop(pre) - 0.12 * fx.hop(dash) + 0.15 * fx.hop(settle)
            tilt = -0.15 * fx.hop(pre) + 0.25 * fx.hop(dash) - 0.2 * fx.hop(settle)
        elif kind == HORSE:
            (sr, sc), (er, ec) = self.move.start, self.move.end
            if abs(er - sr) == 2:
                leg = (sr + (1 if er > sr else -1), sc)
            else:
                leg = (sr, sc + (1 if ec > sc else -1))
            m = square_px(*leg)
            if u < 0.45:
                k = u / 0.45
                x, y = fx.lerp2(s, m, fx.ease_in_out(k))
                lift = fx.hop(k) * 24
            else:
                k = (u - 0.45) / 0.55
                x, y = fx.lerp2(m, e, fx.ease_in_out(k))
                lift = fx.hop(k) * 38
            tilt = -0.25 * fx.hop(fx.seg(u, 0.4, 0.6)) + 0.12 * fx.hop(fx.seg(u, 0.75, 1.0))
            squash = 0.2 * fx.hop(fx.seg(u, 0.9, 1.0))
        elif kind == ELEPHANT:
            steps = u * 2
            k = steps % 1.0 if u < 1 else 1.0
            x, y = fx.lerp2(s, e, fx.ease_in_out(u))
            lift = fx.hop(k) * 16
            squash = 0.22 * fx.hop(fx.seg(k, 0.85, 1.0)) if u < 1 else 0
        elif kind == ADVISOR:
            x, y = fx.lerp2(s, e, fx.ease_in_out(u))
            lift = fx.hop(u) * 14
            tilt = 0.1 * math.sin(u * math.pi * 2)
        elif kind == KING:
            x, y = fx.lerp2(s, e, fx.ease_in_out(u))
            lift = fx.hop((u * 2) % 1.0) * 6 if u < 1 else 0
            tilt = 0.08 * math.sin(u * math.pi * 4)
        elif kind == CANNON:
            pre = fx.seg(u, 0, 0.18)
            fly = fx.seg(u, 0.18, 0.88)
            land = fx.seg(u, 0.88, 1.0)
            x, y = fx.lerp2(s, e, fx.ease_in_out(fly))
            lift = fx.hop(fly) * (70 + 0.15 * math.dist(s, e))
            squash = 0.25 * fx.hop(pre) - 0.15 * fx.hop(fly) + 0.3 * fx.hop(land)
            tilt = 0.25 * math.sin(fly * math.pi) * (1 if e[0] >= s[0] else -1)
        elif kind == PAWN:
            x, y = fx.lerp2(s, e, fx.ease_in_out(u))
            lift = fx.hop((u * 3) % 1.0) * 9 if u < 1 else 0
            squash = 0.1 * fx.hop(fx.seg(u, 0.9, 1.0))
        return x, y, lift, squash, tilt

    # --- drawing ------------------------------------------------------------------
    def draw(self, canvas: pygame.Surface, now: int, square_px: Callable, draw_tile: Callable) -> None:
        elapsed = now - self.started_at
        t = float(now)
        s = square_px(*self.move.start)
        e = square_px(*self.move.end)
        kind, side = self.piece.kind, self.piece.side
        facing = 1 if e[0] > s[0] + 1 else -1 if e[0] < s[0] - 1 else (1 if side == RED else -1)
        travel_ms = MOVE_MS[kind]
        victim = self.move.captured
        capture = victim is not None
        approach_end = 0.78 if capture and kind != CANNON else 1.0

        if capture:
            strike_at = travel_ms * approach_end
            if elapsed < strike_at:
                u = (elapsed / strike_at) * approach_end
            elif elapsed < strike_at + CAPTURE_EXTRA * 0.6:
                u = approach_end
            else:
                u = approach_end + (1 - approach_end) * fx.seg(elapsed, strike_at + CAPTURE_EXTRA * 0.6, travel_ms + CAPTURE_EXTRA)
            impact = strike_at + 90 if kind != CANNON else travel_ms * 0.9
        else:
            u = fx.seg(elapsed, 0, travel_ms)
            impact = None

        x, y, lift, squash, tilt = self._path(u, s, e, square_px)
        moving = 0 < u < 1
        walk_phase = t / (35 if kind in (ROOK, HORSE) else 60) if moving else 0.0
        swing = 0.0
        expr = "fierce" if capture or self.check else ("cheer" if u >= 1 else "idle")

        if capture and impact is not None:
            vt = elapsed - impact
            if vt < 0:
                actor(canvas, victim.kind, victim.side, (e[0], e[1] + BOARD_R), BOARD_R, facing=-facing,
                      expr="scared", t=t)
            else:
                fly = fx.seg(vt, 0, 700)
                dx, dy = e[0] - s[0], e[1] - s[1]
                norm = math.hypot(dx, dy) or 1.0
                vx = e[0] + dx / norm * 360 * fx.ease_out(fly)
                vy = e[1] + dy / norm * 360 * fx.ease_out(fly)
                actor(canvas, victim.kind, victim.side, (vx, vy + BOARD_R), BOARD_R * (1 - 0.4 * fly), facing=-facing,
                      expr="ko", t=t, tilt=fly * 9, pivot="center", lift=fx.hop(fly) * 120,
                      alpha=int(255 * (1 - fx.seg(fly, 0.6, 1.0))), flash=1 - fx.seg(vt, 0, 140), shadow=False)
                fx.impact_star(canvas, (e[0], e[1] - 6), fx.seg(vt, 0, 380), 62, seed=self.move.action % 31)
                fx.onomatopoeia(canvas, STRIKE_WORD[kind], (e[0] + 20, e[1] - 70), 44, (255, 230, 120), fx.seg(vt, 0, 650))
            if kind != CANNON:
                swing = fx.hop(fx.seg(elapsed, impact - 160, impact + 160))
                if kind == HORSE:
                    swing = fx.seg(elapsed, impact - 150, impact + 100)
            else:
                swing = 0.0
            expr = "fierce" if elapsed < impact + 200 else "cheer"

        if kind == ROOK and moving:
            dirx, diry = (e[0] - s[0]), (e[1] - s[1])
            norm = math.hypot(dirx, diry) or 1.0
            layer = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
            for k in range(1, 6):
                tx, ty = x - dirx / norm * k * 16, y - diry / norm * k * 16
                pygame.draw.line(layer, (255, 255, 255, 140 - k * 22), (tx - diry / norm * 18, ty + dirx / norm * 18 - 6),
                                 (tx - dirx / norm * 30, ty - diry / norm * 30 - 6), 3)
            canvas.blit(layer, (0, 0))
        if kind == ADVISOR and moving:
            fx.sparkles(canvas, (x, y - 10), t, 40, count=5)
        if kind == ELEPHANT:
            for k in (0.5, 1.0):
                fx.dust(canvas, fx.lerp2(s, e, k), fx.seg(elapsed, travel_ms * k - 40, travel_ms * k + 360), 26, seed=int(k * 10))
        if kind == CANNON:
            fx.dust(canvas, e, fx.seg(elapsed, travel_ms * 0.88, travel_ms * 0.88 + 420), 34, seed=3)
        if kind == HORSE:
            fx.dust(canvas, e, fx.seg(elapsed, travel_ms * 0.95, travel_ms * 0.95 + 380), 26, seed=4)

        settle = fx.seg(elapsed, self.duration - 140, self.duration)
        wake = fx.seg(elapsed, 0, 90)
        if wake < 1:
            draw_tile(self.piece, s, int(255 * (1 - wake)))
        actor(canvas, kind, side, (x, y + BOARD_R), BOARD_R, facing=facing, expr=expr, t=t, walk=walk_phase,
              swing=swing, squash=squash, tilt=tilt, lift=lift, alpha=int(255 * min(wake, 1 - settle)))
        if settle > 0:
            draw_tile(self.piece, e, int(255 * settle))

        if self.crossed_river:
            fx.speech_bubble(canvas, "过河啦！", (e[0] + 10, e[1] - 40), fx.seg(elapsed, travel_ms * 0.7, self.duration), size=20, max_width=200)
        if self.check:
            fx.speech_bubble(canvas, "将军！", (e[0] + 10, e[1] - 40), fx.seg(elapsed, self.duration - 650, self.duration + 300),
                             size=24, max_width=200, shout=True)


def draw_threat_marks(canvas: pygame.Surface, squares: list[tuple[int, int]], square_px: Callable, t: float, alpha: float = 1.0) -> None:
    for sq in squares:
        x, y = square_px(*sq)
        fx.sweat_drop(canvas, (x + 26, y - 30), 9, t)
        if alpha > 0.5:
            fx.alert_mark(canvas, (x - 24, y - 38 - abs(math.sin(t / 160.0)) * 5), 26)


def draw_king_in_check(canvas: pygame.Surface, pos, t: float) -> None:
    x, y = pos
    fx.sweat_drop(canvas, (x + 28, y - 32), 10, t)
    fx.sweat_drop(canvas, (x - 30, y - 26), 7, t + 300)
    fx.alert_mark(canvas, (x, y - 52 - abs(math.sin(t / 120.0)) * 6), 30)
