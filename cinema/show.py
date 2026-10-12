"""On-board story shows: every move becomes a short scene played on the real squares.

A :class:`MoveShow` is a timeline of segments (wake up → travel → capture → opening / tactic /
check flourish → back to sleep). Each frame it produces a :class:`~cinema.view3d.Frame` describing
which pucks to hide, which characters to draw where, and which effects to paint; everything is
anchored in board coordinates so it follows the pieces, the camera and the flip state.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import pygame

from cinema import board_story
from cinema.detect import StoryEvent
from cinema.fx import (
    clamp, dizzy_stars, dust, ease_in_out, ease_out, explosion, ghost, hop, impact_star, lerp,
    onomatopoeia, outlined_text, radial_lines, seg, shockwave, sparkles, speech_bubble, star_points,
    with_alpha,
)
from cinema.actors import INK
from cinema.fonts import font
from cinema.view3d import BoardView, Frame, Sprite
from game_engine import ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK, Move, Piece

Square = tuple[int, int]

WAKE_MS = 240
SLEEP_MS = 300
OPENING_MS = 2100
TACTIC_MS = 1900
CHECK_MS = 1500
MATE_MS = 3700
STALEMATE_MS = 2800
SACRIFICE_MS = 1600

STRIKE_WORD = {ROOK: "轰!", HORSE: "踢!", CANNON: "砰!", ELEPHANT: "咚!", ADVISOR: "唰!", KING: "斩!", PAWN: "戳!"}
TONE_COLOR = {"dominant": (255, 206, 90), "even": (255, 255, 255), "desperate": (255, 96, 70)}
SIDE_COLOR = {RED: (214, 52, 44), BLACK: (40, 46, 52)}


def _centroid(squares) -> tuple[float, float]:
    squares = list(squares)
    return sum(sq[0] for sq in squares) / len(squares), sum(sq[1] for sq in squares) / len(squares)


def _path(kind: str, start: Square, end: Square) -> list[tuple[float, float]]:
    (sr, sc), (er, ec) = start, end
    if kind == HORSE:
        dr, dc = er - sr, ec - sc
        leg = (sr + (dr // 2 if abs(dr) == 2 else 0), sc + (dc // 2 if abs(dc) == 2 else 0))
        return [start, leg, end]
    if kind == ELEPHANT:
        return [start, ((sr + er) / 2, (sc + ec) / 2), end]
    return [start, end]


def _unit(a, b) -> tuple[float, float]:
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dr, dc) or 1.0
    return dr / n, dc / n


def _along(points, u: float) -> tuple[float, float, int, float]:
    """Point at fraction ``u`` of a polyline, plus the leg index and progress within it."""
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    total = sum(lengths) or 1.0
    d = clamp(u) * total
    for i, length in enumerate(lengths):
        if d <= length or i == len(lengths) - 1:
            k = clamp(d / length) if length else 1.0
            a, b = points[i], points[i + 1]
            return lerp(a[0], b[0], k), lerp(a[1], b[1], k), i, k
        d -= length
    return points[-1][0], points[-1][1], len(points) - 2, 1.0


def travel_ms(kind: str, points) -> int:
    dist = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    return {
        ROOK: int(420 + 70 * dist), HORSE: 760, ELEPHANT: 820, ADVISOR: 560, KING: 560,
        CANNON: int(560 + 45 * dist), PAWN: int(380 + 180 * max(1.0, dist)),
    }[kind]


class MoveShow:
    def __init__(self, move: Move, piece: Piece, before, after, events: list[StoryEvent],
                 mode: str = "full", winner: Optional[str] = None) -> None:
        self.move, self.piece = move, piece
        self.before, self.after = before, after
        self.events = events
        self.mode = mode
        self.winner = winner
        self.started_at: Optional[int] = None
        self.side = piece.side
        self.enemy = BLACK if piece.side == RED else RED
        self.fwd = -1 if piece.side == RED else 1
        self.home = 9 if piece.side == RED else 0

        def first(kinds) -> Optional[StoryEvent]:
            return next((e for e in events if e.kind in kinds), None)

        self.capture = first(("capture",))
        self.opening = first(("opening",))
        self.tactic = first(("tactic",))
        self.finale = first(("mate", "stalemate", "sacrifice", "check"))
        self.victim = move.captured
        self.quick = self.victim is not None and (
            mode == "off" or (mode == "highlights" and self.victim.kind in (PAWN, ADVISOR, ELEPHANT)
                              and self.capture.tone == "even")
        )
        self.tone = self.capture.tone if self.capture else "even"
        self.points = _path(piece.kind, move.start, move.end)
        if self.victim is not None and piece.kind != CANNON:
            last = self.points[-2]
            d = _unit(last, move.end)
            self.approach = (move.end[0] - d[0] * 0.62, move.end[1] - d[1] * 0.62)
            self.points = self.points[:-1] + [self.approach]
        else:
            self.approach = move.end
        self.hit_dir = _unit(move.start if piece.kind == CANNON else self.points[-2], move.end)

        self.segments: list[tuple[int, int, Callable, Optional[tuple], float]] = []
        self._t = 0
        self._add(WAKE_MS, self._wake)
        if not (self.victim is not None and piece.kind == CANNON):
            self.travel_start = self._t
            self._add(travel_ms(piece.kind, self.points), self._travel)
        if self.victim is not None:
            self._plan_capture()
            self.capture_start = self._t
            push = 1.0 if self.quick else {"dominant": 1.14, "even": 1.2, "desperate": 1.32}[self.tone]
            self._add(self.cap_total, self._capture, move.end, push)
        else:
            self.capture_start = None
        if mode != "off":
            if self.opening and self.opening.key in board_story.OPENINGS:
                self._add(OPENING_MS, self._story(board_story.OPENINGS[self.opening.key], self.opening, OPENING_MS),
                          move.end, 1.06)
            if self.tactic and self.tactic.key in board_story.TACTICS:
                self._add(TACTIC_MS, self._story(board_story.TACTICS[self.tactic.key], self.tactic, TACTIC_MS),
                          _centroid([move.end, *self.tactic.targets]), 1.22)
            finale = self.finale
            if finale is not None:
                check_tactic = self.tactic is not None and self.tactic.key in ("discovered_check", "double_check", "check_win")
                king = finale.targets[0] if finale.targets else move.end
                if finale.kind == "mate":
                    self._add(MATE_MS, self._story(board_story.mate_painter(finale.key), finale, MATE_MS),
                              _centroid([king, king, *finale.cast]), 1.3)
                elif finale.kind == "stalemate":
                    self._add(STALEMATE_MS, self._story(board_story.stalemate, finale, STALEMATE_MS), king, 1.36)
                elif finale.kind == "sacrifice":
                    self._add(SACRIFICE_MS, self._story(board_story.sacrifice, finale, SACRIFICE_MS), move.end, 1.2)
                elif not check_tactic and (mode == "full" or finale.key != "generic"):
                    self._add(CHECK_MS, self._story(board_story.check_painter(finale.key), finale, CHECK_MS),
                              _centroid([king, move.end]), 1.14)
        self._add(SLEEP_MS, self._sleep)
        self.duration = self._t
        self.m: Sprite = Sprite(piece, *move.start, awake=0.0)

    # --- timeline ----------------------------------------------------------------
    def _add(self, dur: int, fn: Callable, focus: Optional[tuple] = None, zoom: float = 1.0) -> None:
        self.segments.append((self._t, dur, fn, focus, zoom))
        self._t += dur

    def _camera_at(self, t: float) -> tuple[float, Optional[tuple]]:
        for start, dur, _, focus, zoom in self.segments:
            if start <= t < start + dur:
                return zoom, focus
        return 1.0, None

    def camera(self, t: float) -> tuple[float, Optional[tuple[float, float]]]:
        """Zoom and focus eased over a short window so the camera glides between segments."""
        zoom, weight, fr, fc = 0.0, 0.0, 0.0, 0.0
        samples = 8
        for k in range(samples):
            z, focus = self._camera_at(t - 480 + 480 * k / (samples - 1))
            zoom += z / samples
            if focus is not None and z > 1.0:
                w = z - 1.0
                weight += w
                fr += focus[0] * w
                fc += focus[1] * w
        zoom = 1.0 + (zoom - 1.0) * (1.0 - ease_in_out(seg(t, self.duration - 420, self.duration)))
        return zoom, ((fr / weight, fc / weight) if weight else None)

    def done(self, now: int) -> bool:
        return self.started_at is not None and now - self.started_at >= self.duration

    def skip(self, now: int) -> None:
        self.started_at = now - self.duration

    def frame(self, now: int) -> Frame:
        t = 0 if self.started_at is None else now - self.started_at
        f = Frame(now=now)
        f.hidden.add(self.move.end)
        expr = "fierce" if self.victim is not None else ("cheer" if self.opening else "idle")
        self.m = Sprite(self.piece, *self.move.start, awake=0.0, expr=expr, look=self.move.end)
        if self.victim is not None and t < self.capture_start:
            self._victim_waiting(f, t)
        for start, dur, fn, _, _ in self.segments:
            if t < start:
                break
            lt = min(t - start, dur)
            fn(f, lt, lt / dur if dur else 1.0, t < start + dur)
        f.add(self.m)
        f.zoom, f.focus = self.camera(t)
        return f

    def _story(self, painter: Callable, event: StoryEvent, dur: int) -> Callable:
        def run(f: Frame, lt: float, u: float, active: bool) -> None:
            painter(self, f, event, lt, u, active, dur)
        return run

    # --- generic helpers used by board_story ----------------------------------------
    def piece_at(self, sq: Square) -> Optional[Piece]:
        return self.after[sq[0]][sq[1]]

    def cast(self, f: Frame, sq: Square, lt: float, dur: float, expr: str = "idle", wake: float = 220, **pose) -> Sprite:
        """Wake up the piece on ``sq`` (fully awake by ``wake`` ms) and return its character to pose further."""
        if tuple(sq) == tuple(self.move.end):
            sprite = self.m
            sprite.expr = expr
        else:
            piece = self.piece_at(sq)
            if piece is None:
                return Sprite(Piece(self.enemy, PAWN), *sq, alpha=0)
            awake = seg(lt, max(0.0, wake - 260), max(1.0, wake)) * (1 - seg(lt, dur - 240, dur))
            sprite = f.add(Sprite(piece, sq[0], sq[1], awake=awake, expr=expr), hide=tuple(sq))
        for key, value in pose.items():
            setattr(sprite, key, value)
        return sprite

    def bubble(self, f: Frame, sq, text: str, u: float, z: float = 1.45, shout: bool = False, size: int = 18) -> None:
        if not text or u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y = view.point(sq[0], sq[1], z)
            y = max(y, 96.0)
            tail = "down-left" if x < 470 else "down-right"
            speech_bubble(canvas, text, (x, y), u, size=size, max_width=230, tail=tail, shout=shout)
        f.overlay.append(draw)

    def word(self, f: Frame, sq, text: str, u: float, z: float = 1.1, size: float = 0.62,
             color=(255, 236, 120), angle: float = -8.0) -> None:
        if u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y, s = view.project(sq[0], sq[1], z)
            onomatopoeia(canvas, text, (x, y), s * size, color, u, angle)
        f.overlay.append(draw)

    def ribbon(self, f: Frame, title: str, subtitle: str, u: float, sq=None, tag: str = "", color=None) -> None:
        if u <= 0 or u >= 1 or self.mode == "off":
            return
        sq = sq or self.move.end
        color = color or SIDE_COLOR[self.side]

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y = view.point(sq[0], sq[1], 0)
            draw_ribbon(canvas, title, subtitle, (lerp(430, x, 0.4), 62 if y > 430 else 806), u, color, tag)
        f.hud.append(draw)

    def stamp(self, f: Frame, sq, text: str, u: float, z: float = 1.9, color=(206, 34, 30), size: float = 0.9) -> None:
        if u <= 0:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y, s = view.project(sq[0], sq[1], z)
            draw_seal(canvas, text, (x, y), s * size, u, color)
        f.overlay.append(draw)

    def beam(self, f: Frame, a, b, progress: float, color=(255, 80, 60), alpha: float = 220, width: float = 0.12,
             dashed: bool = False, arrow: bool = False) -> None:
        if progress <= 0 or alpha <= 0:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            view.ground_line(a, b, color, width * 2.4, alpha * 0.25, progress)
            view.ground_line(a, b, color, width, alpha, progress, dashed=dashed, arrow=arrow)
        f.ground.append(draw)

    def bolt(self, f: Frame, a, b, u: float, color=(255, 120, 80), z: float = 0.45) -> None:
        if u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            for k in range(6):
                uu = max(0.0, u - k * 0.035)
                x, y, s = view.project(lerp(a[0], b[0], uu), lerp(a[1], b[1], uu), z)
                pygame.draw.circle(canvas, with_alpha(color, 255)[:3], (int(x), int(y)), max(2, int(s * (0.16 - k * 0.022))))
            x, y, s = view.project(lerp(a[0], b[0], u), lerp(a[1], b[1], u), z)
            pygame.draw.circle(canvas, (255, 255, 240), (int(x), int(y)), max(2, int(s * 0.09)))
        f.air.append(draw)

    def shell(self, f: Frame, a, b, u: float, peak: float = 1.5) -> None:
        if u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            for k in range(6, -1, -1):
                uu = max(0.0, u - k * 0.03)
                x, y, s = view.project(lerp(a[0], b[0], uu), lerp(a[1], b[1], uu), 0.45 + hop(uu) * peak)
                r = s * 0.14 * (1 - k * 0.11)
                if k:
                    pygame.draw.circle(canvas, (255, 200 - k * 18, 110 - k * 12), (int(x), int(y)), max(1, int(r)))
                else:
                    pygame.draw.circle(canvas, INK, (int(x), int(y)), int(r) + 2)
                    pygame.draw.circle(canvas, (70, 74, 92), (int(x), int(y)), int(r))
                    pygame.draw.circle(canvas, (200, 205, 220), (int(x - r * 0.3), int(y - r * 0.3)), max(1, int(r * 0.3)))
        f.air.append(draw)

    def boom(self, f: Frame, sq, u: float, size: float = 1.0, z: float = 0.4) -> None:
        if u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y, s = view.project(sq[0], sq[1], z)
            explosion(canvas, (x, y), u, s * 0.8 * size)
        f.air.append(draw)

    def impact(self, f: Frame, sq, u: float, size: float = 1.0, color=(255, 236, 120), z: float = 0.5) -> None:
        if u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y, s = view.project(sq[0], sq[1], z)
            impact_star(canvas, (x, y), u, s * 0.75 * size, color)
            gx, gy, _ = view.project(sq[0], sq[1], 0)
            shockwave(canvas, (gx, gy), u, s * 1.4 * size, (255, 255, 255), 5, flat=view.fore(sq[0]))
        f.air.append(draw)

    def puff(self, f: Frame, sq, u: float, size: float = 0.8) -> None:
        if u <= 0 or u >= 1:
            return

        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            x, y, s = view.project(sq[0], sq[1], 0)
            dust(canvas, (x, y), u, s * 0.5 * size)
        f.air.append(draw)

    def glow_ring(self, f: Frame, sq, u: float, color=(255, 214, 90), radius: float = 0.55, alpha: float = 220) -> None:
        def draw(canvas: pygame.Surface, view: BoardView) -> None:
            pulse = 0.5 + 0.5 * math.sin(u * 18)
            view.ground_ring(sq, radius + 0.06 * pulse, color, 4, alpha * (0.6 + 0.4 * pulse))
            view.ground_ring(sq, radius * 0.75, color, 2, alpha * 0.5)
        f.ground.append(draw)

    def in_air(self, f: Frame, painter: Callable[[pygame.Surface, BoardView], None]) -> None:
        f.air.append(painter)

    def on_ground(self, f: Frame, painter: Callable[[pygame.Surface, BoardView], None]) -> None:
        f.ground.append(painter)

    def on_top(self, f: Frame, painter: Callable[[pygame.Surface, BoardView], None]) -> None:
        f.overlay.append(painter)

    # --- wake / travel / sleep ----------------------------------------------------------
    def _wake(self, f: Frame, lt: float, u: float, active: bool) -> None:
        self.m.awake = u

    def _sleep(self, f: Frame, lt: float, u: float, active: bool) -> None:
        self.m.row, self.m.col = self.move.end
        self.m.awake = 1 - u
        self.m.swing = 0
        self.m.z = 0

    def _travel(self, f: Frame, lt: float, u: float, active: bool) -> None:
        m, kind = self.m, self.piece.kind
        m.awake = 1.0
        pts = self.points
        dur = travel_ms(kind, pts)
        if kind == ROOK:
            pull = seg(u, 0, 0.18)
            go = ease_in_out(seg(u, 0.18, 0.92))
            r, c, _, _ = _along(pts, go)
            back = _unit(pts[-1], pts[0])
            r += back[0] * 0.18 * hop(pull) * (1 - go)
            c += back[1] * 0.18 * hop(pull) * (1 - go)
            m.row, m.col = r, c
            m.walk = go * 20
            m.tilt = 0.18 * hop(seg(u, 0.18, 0.9))
            m.squash = -0.12 * hop(pull) + 0.18 * hop(seg(u, 0.9, 1.0))
            if active and 0.2 < u < 0.92:
                self.in_air(f, lambda cv, v, rr=r, cc=c: _streaks(cv, v, pts[0], (rr, cc), lt))
        elif kind in (HORSE, ELEPHANT):
            r, c, leg, k = _along(pts, ease_in_out(u) if kind == ELEPHANT else u)
            legs = len(pts) - 1
            local = seg(u, leg / legs, (leg + 1) / legs)
            m.row, m.col = r, c
            m.z = hop(local) * (0.62 if kind == HORSE else 0.32)
            m.tilt = (local - 0.5) * (0.5 if kind == HORSE else 0.15)
            m.walk = u * 12
            land = seg(local, 0.85, 1.0)
            m.squash = 0.25 * hop(land) if kind == ELEPHANT else 0.12 * hop(land)
            if kind == ELEPHANT and active and leg >= 0:
                stomp = (u * legs) % 1.0
                if stomp < 0.25 and u > 0.3:
                    f.shake = max(f.shake, 4 * (1 - stomp / 0.25))
                    self.puff(f, pts[min(leg, legs)], stomp / 0.25 * 0.6 + 0.2, 0.8)
        elif kind == CANNON:
            r, c, _, _ = _along(pts, ease_in_out(u))
            dist = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            m.row, m.col = r, c
            m.z = hop(u) * min(2.0, 0.5 + 0.2 * dist)
            m.squash = 0.22 * hop(seg(u, 0.88, 1.0)) - 0.15 * hop(seg(u, 0, 0.12))
            m.tilt = math.sin(u * math.tau) * 0.2
        elif kind == PAWN:
            r, c, _, _ = _along(pts, ease_in_out(u))
            steps = max(2, int(round(math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) * 2)))
            m.row, m.col = r, c
            m.z = abs(math.sin(u * math.pi * steps)) * 0.22
            m.walk = u * 18
        else:
            r, c, _, _ = _along(pts, ease_in_out(u))
            m.row, m.col = r, c
            m.z = (0.14 + 0.04 * math.sin(lt / 90.0)) * hop(u) if kind == ADVISOR else abs(math.sin(u * math.pi * 3)) * 0.08
            m.walk = u * 14
        crossed = kind == PAWN and (self.move.start[0] - 4.5) * (self.move.end[0] - 4.5) < 0
        if crossed and self.mode == "full" and self.victim is None:
            self.bubble(f, self.move.end, "过河啦！", seg(lt, dur * 0.55, dur + 700), z=1.4)
        if kind == ELEPHANT and self.victim is None and active:
            f.shake = max(f.shake, 3 * hop(seg(u, 0.9, 1.0)))

    # --- capture ------------------------------------------------------------------------
    def _plan_capture(self) -> None:
        quick, tone = self.quick, self.tone
        self.slow = tone == "desperate" and not quick
        if self.piece.kind == CANNON:
            aim = 160 if quick else (620 if self.slow else 400)
            self.cap_impact = aim + 380
            self.cap_leap = self.cap_impact + (260 if quick else 520)
            self.cap_total = self.cap_leap + 480 + (160 if quick else 520)
        else:
            windup = 160 if quick else (460 if self.slow else 260)
            self.cap_clash = 0 if quick or tone != "even" else 440
            self.cap_impact = windup + self.cap_clash + 160
            self.cap_step = self.cap_impact + (300 if quick else 480)
            self.cap_total = self.cap_step + 300 + (180 if quick else 640)
        if self.slow:
            self.cap_total += 220

    def _vt(self, lt: float) -> float:
        """Time since impact, slowed down right after the hit for desperate comebacks."""
        d = lt - self.cap_impact
        if not self.slow or d <= 0:
            return d
        return 0.3 * min(d, 320) + max(0.0, d - 320) - 0.0

    def _victim_waiting(self, f: Frame, t: float) -> None:
        victim = self.victim
        end = self.move.end
        lead = self.capture_start - t
        expr = {"dominant": "sad", "even": "fierce", "desperate": "smug"}[self.tone]
        if self.quick:
            expr = "scared"
        awake = 0.0 if self.quick else clamp(1 - (lead - 80) / 300)
        f.add(Sprite(victim, end[0], end[1], awake=awake, expr=expr, look=self.move.start,
                     shake=math.sin(t / 30.0) * 2 * awake if expr in ("scared", "sad") else 0.0))

    def _capture(self, f: Frame, lt: float, u: float, active: bool) -> None:
        m, kind, end = self.m, self.piece.kind, self.move.end
        I = self.cap_impact
        tone_color = TONE_COLOR[self.tone]
        m.awake = 1.0
        m.look = end
        if kind == CANNON:
            self._cannon_attack(f, lt)
        else:
            m.row, m.col = self.approach
            windup_end = I - 160 - self.cap_clash
            if lt < windup_end:
                m.swing = -0.25 * seg(lt, 0, windup_end)
                m.squash = 0.12 * seg(lt, 0, windup_end)
                m.expr = "smug" if self.tone == "dominant" else "fierce"
            elif lt < I - 160:
                cu = seg(lt, windup_end, I - 160)
                m.swing = 0.6 * hop(cu * 2 % 1.0)
                m.row += self.hit_dir[0] * 0.12 * hop(cu * 2 % 1.0)
                m.col += self.hit_dir[1] * 0.12 * hop(cu * 2 % 1.0)
                mid = (lerp(self.approach[0], end[0], 0.5), lerp(self.approach[1], end[1], 0.5))
                for k in range(2):
                    spark = seg(cu, 0.2 + k * 0.5, 0.5 + k * 0.5)
                    self.impact(f, mid, spark, 0.45, (200, 230, 255), z=0.7)
                self.word(f, mid, "铛!", seg(cu, 0.15, 1.0), z=1.5, size=0.5, color=(200, 230, 255))
            elif lt < I:
                m.swing = ease_out(seg(lt, I - 160, I))
                m.row += self.hit_dir[0] * 0.25 * m.swing
                m.col += self.hit_dir[1] * 0.25 * m.swing
            else:
                step = seg(lt, self.cap_step, self.cap_step + 300)
                m.swing = 1 - seg(lt, I + 200, self.cap_step)
                m.row = lerp(self.approach[0], end[0], ease_in_out(step))
                m.col = lerp(self.approach[1], end[1], ease_in_out(step))
                m.z = hop(step) * 0.25
                if lt > self.cap_step + 300:
                    self._pose(f, lt - self.cap_step - 300)
            if self.slow and lt < windup_end:
                m.aura = (255, 110, 60)
        vt = self._vt(lt)
        if vt >= 0:
            self._victim_hit(f, vt)
            if active:
                self.impact(f, end, seg(vt, 0, 520), 1.4 if self.slow else 1.0, tone_color)
                f.shake = max(f.shake, (12 if self.slow else 7) * (1 - seg(vt, 0, 320)))
                word = STRIKE_WORD[kind]
                self.word(f, end, word, seg(vt, 0, 780), z=1.35, size=0.9 if self.slow else 0.7, color=tone_color)
                if self.slow:
                    f.flash = max(f.flash, 0.35 * (1 - seg(vt, 0, 200)))
                    self.in_air(f, lambda cv, v, uu=seg(vt, 0, 700): _speed_burst(cv, v, end, uu, lt))
        elif not self.quick and active:
            if self.capture.attacker_line and self.mode == "full":
                self.bubble(f, self.approach if kind != CANNON else self.move.start, self.capture.attacker_line,
                            seg(lt, 0, I + 120), z=1.5)
            if self.slow:
                self.in_air(f, lambda cv, v: _speed_burst(cv, v, self.approach, seg(lt, 0, I) * 0.7, lt))
        if not self.quick and self.mode == "full" and active and self.capture.victim_line:
            self.bubble(f, end, self.capture.victim_line, seg(lt, I * 0.45, I + 420), z=1.45)
        if not self.quick:
            self.ribbon(f, self.capture.title, self.capture.subtitle, seg(lt, I + 120, self.cap_total + 200),
                        tag=self.capture.tag, color=_tag_color(self.tone, self.side))

    def _cannon_attack(self, f: Frame, lt: float) -> None:
        m, start, end = self.m, self.move.start, self.move.end
        I = self.cap_impact
        fire = I - 380
        m.row, m.col = start
        m.expr = "smug" if self.tone == "dominant" else "fierce"
        screen = self._cannon_screen()
        if lt < fire:
            m.swing = 0
            lock = seg(lt, 0, fire)
            self.on_ground(f, lambda cv, v: (v.ground_ring(end, 0.9 - 0.4 * ease_out(lock), (255, 70, 60), 3, 220),
                                           v.ground_line(start, end, (255, 90, 70), 0.04, 160 * lock, lock, dashed=True)))
            if self.slow:
                m.aura = (255, 110, 60)
        elif lt < self.cap_leap:
            m.swing = 1 - seg(lt, fire + 80, fire + 420)
            m.squash = 0.2 * (1 - seg(lt, fire, fire + 200))
            self.shell(f, start, end, seg(lt, fire, I), 1.4 + 0.08 * abs(start[0] - end[0] + start[1] - end[1]))
            self.puff(f, start, seg(lt, fire, fire + 420), 0.7)
        else:
            leap = seg(lt, self.cap_leap, self.cap_leap + 480)
            m.row, m.col = lerp(start[0], end[0], ease_in_out(leap)), lerp(start[1], end[1], ease_in_out(leap))
            m.z = hop(leap) * 1.3
            m.squash = 0.2 * hop(seg(leap, 0.85, 1.0))
            if leap >= 1:
                self._pose(f, lt - self.cap_leap - 480)
        if screen is not None:
            duck = seg(lt, fire - 200, fire) * (1 - seg(lt, I + 100, I + 400))
            sp = self.cast(f, screen, lt, self.cap_total, expr="scared" if duck > 0.1 else "idle", wake=fire * 0.6)
            sp.squash = 0.35 * duck
            sp.look = start
            if duck > 0.3 and not self.quick:
                self.word(f, screen, "低头!", seg(lt, fire - 150, I + 200), z=1.2, size=0.42, color=(255, 255, 255))

    def _cannon_screen(self) -> Optional[Square]:
        (r1, c1), (r2, c2) = self.move.start, self.move.end
        dr, dc = (r2 > r1) - (r2 < r1), (c2 > c1) - (c2 < c1)
        r, c = r1 + dr, c1 + dc
        while (r, c) != (r2, c2):
            if self.before[r][c] is not None:
                return r, c
            r, c = r + dr, c + dc
        return None

    def _pose(self, f: Frame, pt: float) -> None:
        m = self.m
        m.row, m.col = self.move.end
        m.swing = 0
        if self.tone == "dominant":
            m.expr = "smug" if pt < 300 else "sleepy"
            self.in_air(f, lambda cv, v: (lambda x, y, s: sparkles(cv, (x, y), pt, s * 0.8, alpha=200))(*v.project(*self.move.end, 0.8)))
        elif self.tone == "desperate":
            m.expr = "fierce"
            m.aura = (255, 110, 60)
            m.z = hop(seg(pt, 0, 300)) * 0.3
        else:
            m.expr = "cheer"
            m.z = hop(seg(pt, 0, 320)) * 0.35

    def _victim_hit(self, f: Frame, vt: float) -> None:
        victim, end = self.victim, self.move.end
        kind = victim.kind
        d = self.hit_dir
        rs = {ROOK: 260, HORSE: 420, CANNON: 380, ELEPHANT: 360, ADVISOR: 340, PAWN: 60, KING: 400}[kind]
        if self.quick:
            rs = min(rs, 120)
        fly = seg(vt, rs, rs + 680)
        sp = Sprite(victim, end[0], end[1], expr="hurt", look=self.move.start, flash=0.8 * (1 - seg(vt, 0, 140)))
        props: set[str] = set()
        if kind == ROOK:
            props.add("no_wheels")
            sp.expr = "surprised"
            for k, ang in enumerate((-0.6, 0.9)):
                self.in_air(f, lambda cv, v, kk=k, a=ang: _wheel(cv, v, end, seg(vt, 0, 820), a, kk))
        elif kind == HORSE:
            sp.expr = "ko"
            sp.tilt = -1.2 * ease_out(seg(vt, 60, 320))
            if vt < rs:
                self.in_air(f, lambda cv, v: (lambda x, y, s: dizzy_stars(cv, (x, y), vt, s * 0.35))(*v.project(*end, 1.15)))
        elif kind == CANNON:
            props.add("soot")
            self.boom(f, end, seg(vt, 0, 640), 0.9)
            if self.mode == "full" and not self.quick:
                self.bubble(f, end, "咳咳……", seg(vt, 120, rs + 500), z=1.3, size=16)
        elif kind == ELEPHANT:
            props.add("no_tusks")
            sp.expr = "ko"
            sp.squash = 0.55 * seg(vt, 0, 120) * (1 - seg(vt, rs - 80, rs))
            self.in_air(f, lambda cv, v: _tusks(cv, v, end, seg(vt, 0, 760)))
        elif kind == ADVISOR:
            props.add("no_hat")
            sp.expr = "surprised"
            sp.tilt = 0.5 * seg(vt, 80, rs)
            self.in_air(f, lambda cv, v: _hat(cv, v, end, seg(vt, 0, 820), self.enemy))
        elif kind == PAWN:
            sp.expr = "scared"
        elif kind == KING:
            props.add("no_hat")
            sp.expr = "ko"
        sp.props = frozenset(props)
        if fly > 0:
            reach = 3.2 if kind != PAWN else 1.4
            sp.row = end[0] + d[0] * reach * ease_out(fly)
            sp.col = end[1] + d[1] * reach * ease_out(fly)
            sp.z = (math.sin(fly * math.pi * 0.85) * 1.9 + fly * 0.6) if kind != PAWN else 7 * ease_out(fly)
            sp.tilt += fly * 9 * (1 if d[1] >= 0 else -1)
            sp.alpha = int(255 * (1 - seg(fly, 0.55, 1.0)))
            sp.scale = 1 - 0.35 * fly
            sp.shadow = fly < 0.4
            if kind == PAWN and fly > 0.9:
                self.in_air(f, lambda cv, v: _twinkle(cv, v, (sp.row, sp.col), 7.0, seg(fly, 0.9, 1.0)))
        if fly < 1:
            f.add(sp)
        if kind in (HORSE, ELEPHANT, ADVISOR) and not self.quick and self.mode == "full":
            g = seg(vt, rs + 200, rs + 1300)
            if 0 < g < 1:
                self.in_air(f, lambda cv, v: (lambda x, y, s: ghost(cv, (x, y), g, s * 0.5))(*v.project(*end, 0.6)))


# --- drawing helpers ------------------------------------------------------------------

def _tag_color(tone: str, side: str) -> tuple[int, int, int]:
    if tone == "desperate":
        return (196, 60, 30)
    if tone == "dominant":
        return (176, 120, 30)
    return SIDE_COLOR[side]


def _streaks(canvas: pygame.Surface, view: BoardView, origin, pos, t: float) -> None:
    x0, y0, s = view.project(origin[0], origin[1], 0.4)
    x1, y1, _ = view.project(pos[0], pos[1], 0.4)
    ang = math.atan2(y1 - y0, x1 - x0)
    for k in range(6):
        off = (k - 2.5) * s * 0.09
        nx, ny = -math.sin(ang) * off, math.cos(ang) * off
        length = s * (0.6 + (k * 37 % 5) * 0.12)
        sx, sy = x1 - math.cos(ang) * s * 0.4 + nx, y1 - math.sin(ang) * s * 0.4 + ny
        pygame.draw.line(canvas, (255, 255, 255), (sx, sy), (sx - math.cos(ang) * length, sy - math.sin(ang) * length), 2)


def _speed_burst(canvas: pygame.Surface, view: BoardView, sq, u: float, t: float) -> None:
    if u <= 0 or u >= 1:
        return
    x, y, s = view.project(sq[0], sq[1], 0.5)
    radial_lines(canvas, (x, y), t, s * 0.9, s * 3.2, (255, 255, 255), count=22, alpha=160 * (1 - u))


def _wheel(canvas: pygame.Surface, view: BoardView, sq, u: float, ang: float, k: int) -> None:
    if u <= 0 or u >= 1:
        return
    r = sq[0] + math.sin(ang) * 2.2 * u
    c = sq[1] + math.cos(ang) * 2.2 * u
    x, y, s = view.project(r, c, 0.3 + hop(u) * 1.6)
    rad = s * 0.16
    spin = u * 20 + k
    pygame.draw.circle(canvas, INK, (int(x), int(y)), int(rad) + 2)
    pygame.draw.circle(canvas, (120, 84, 50), (int(x), int(y)), int(rad))
    for j in range(3):
        a = spin + j * 2.09
        pygame.draw.line(canvas, INK, (x, y), (x + math.cos(a) * rad, y + math.sin(a) * rad), 2)


def _tusks(canvas: pygame.Surface, view: BoardView, sq, u: float) -> None:
    if u <= 0 or u >= 1:
        return
    for sgn in (-1, 1):
        x, y, s = view.project(sq[0] - 0.3 * u, sq[1] + sgn * 1.4 * u, 0.5 + hop(u) * 1.2)
        a = u * 12 * sgn
        pts = [(x + math.cos(a + d) * s * 0.18 * w, y + math.sin(a + d) * s * 0.18 * w) for d, w in ((0, 1), (2.6, 0.5), (3.4, 0.5))]
        pygame.draw.polygon(canvas, (255, 250, 236), pts)
        pygame.draw.polygon(canvas, INK, pts, 2)


def _hat(canvas: pygame.Surface, view: BoardView, sq, u: float, side: str) -> None:
    if u <= 0 or u >= 1:
        return
    x, y, s = view.project(sq[0] - 0.8 * u, sq[1] + 0.6 * u, 1.0 + hop(u) * 1.4)
    a = u * 9
    w = s * 0.32
    pts = [(x + math.cos(a + k * 1.57) * w, y + math.sin(a + k * 1.57) * w * 0.5) for k in range(4)]
    pygame.draw.polygon(canvas, (40, 44, 70) if side == BLACK else (150, 30, 40), pts)
    pygame.draw.polygon(canvas, INK, pts, 2)


def _twinkle(canvas: pygame.Surface, view: BoardView, sq, z: float, u: float) -> None:
    x, y, s = view.project(sq[0], sq[1], z)
    r = s * 0.35 * hop(u)
    if r > 1:
        pygame.draw.polygon(canvas, (255, 250, 200), star_points(x, y, r, r * 0.3, 4, u * 3))


def draw_seal(canvas: pygame.Surface, text: str, center, size: float, u: float, color=(206, 34, 30)) -> None:
    """A square red seal that slams down (``u`` 0..1, holds once landed)."""
    slam = clamp(u / 0.25)
    scale = 1 + 1.4 * (1 - slam) ** 2
    alpha = int(255 * min(1.0, slam * 2))
    side = int(size * scale)
    if side < 4:
        return
    layer = pygame.Surface((side + 8, side + 8), pygame.SRCALPHA)
    rect = pygame.Rect(4, 4, side, side)
    pygame.draw.rect(layer, (*color, 235), rect, border_radius=max(2, side // 10))
    pygame.draw.rect(layer, (255, 236, 220, 255), rect.inflate(-side // 8, -side // 8), max(2, side // 22), border_radius=max(2, side // 12))
    chars = list(text)
    if len(chars) <= 2:
        f = font(int(side * (0.5 if len(chars) == 2 else 0.68)), True)
        img = f.render(text, True, (255, 240, 226))
        layer.blit(img, img.get_rect(center=rect.center))
    else:
        f = font(int(side * 0.34), True)
        half = (len(chars) + 1) // 2
        for i, line in enumerate(("".join(chars[:half]), "".join(chars[half:]))):
            img = f.render(line, True, (255, 240, 226))
            layer.blit(img, img.get_rect(center=(rect.centerx, rect.y + side * (0.32 + 0.36 * i))))
    layer = pygame.transform.rotozoom(layer, -8, 1.0)
    layer.set_alpha(alpha)
    canvas.blit(layer, layer.get_rect(center=(int(center[0]), int(center[1]))))


def draw_ribbon(canvas: pygame.Surface, title: str, subtitle: str, anchor, u: float, color, tag: str = "") -> None:
    """Brush-stroke title banner hovering over the board."""
    appear = clamp(u / 0.12)
    fade = 1 - seg(u, 0.85, 1.0)
    if appear <= 0 or fade <= 0:
        return
    title_font = font(40, True)
    tw = title_font.size(title)[0]
    sw = font(17, True).size(subtitle)[0] if subtitle else 0
    w, h = max(tw + 120, sw + 50), 66
    x = max(30 + w / 2, min(800 - w / 2, anchor[0]))
    y = max(56, min(820, anchor[1]))
    layer = pygame.Surface((w + 40, h + 70), pygame.SRCALPHA)
    reveal = ease_out(appear)
    pts = []
    for k in range(21):
        px = 20 + w * k / 20 * reveal
        pts.append((px, 14 + math.sin(k * 1.7) * 3))
    for k in range(20, -1, -1):
        px = 20 + w * k / 20 * reveal
        pts.append((px, 14 + h + math.cos(k * 1.3) * 4))
    pygame.draw.polygon(layer, INK, [(px + 3, py + 4) for px, py in pts])
    pygame.draw.polygon(layer, color, pts)
    for k in range(5):
        yy = 22 + k * (h - 16) / 4
        pygame.draw.line(layer, with_alpha((255, 255, 255), 36), (26, yy), (20 + w * reveal - 10, yy + 2), 2)
    if reveal > 0.6:
        outlined_text(layer, title, 40, (255, 240, 205), (20 + w / 2, 14 + h / 2), outline=INK, ow=3)
    if subtitle and reveal > 0.8:
        sub = font(17, True).render(subtitle, True, (255, 246, 230))
        plate = pygame.Rect(0, 0, sub.get_width() + 24, sub.get_height() + 8)
        plate.midtop = (20 + w / 2, 14 + h + 6)
        pygame.draw.rect(layer, (30, 18, 12, 220), plate, border_radius=10)
        layer.blit(sub, sub.get_rect(center=plate.center))
    layer.set_alpha(int(255 * fade))
    canvas.blit(layer, (int(x - w / 2 - 20), int(y - h / 2 - 14)))
    if tag and reveal > 0.9:
        draw_seal(canvas, tag, (x + w / 2 + 6, y - h / 2 + 6), 58, seg(u, 0.14, 0.4), (200, 40, 30))
