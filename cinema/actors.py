"""Hand-drawn cartoon characters for every Xiangqi piece.

Each actor is the familiar octagonal piece tile come to life: a face above the glyph,
little feet, and a kind-specific costume (crown, scholar hat, trunk, mane, chariot
wheels, cannon barrel, bamboo hat). Characters are rendered upright facing right and
transformed (flip / squash / tilt) when blitted, so scenes can animate them freely.
"""

from __future__ import annotations

import math
from typing import Optional

import pygame

from game_engine import ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK
from cinema.detect import KIND_LABEL
from cinema.fonts import font

INK = (42, 24, 52)
GOLD = (250, 198, 92)
GOLD_DARK = (190, 128, 40)
STEEL = (196, 206, 222)
STEEL_DARK = (110, 120, 142)
WOOD = (156, 100, 58)
STRAW = (236, 196, 120)
WHITE = (255, 255, 255)

TEAM = {
    RED: {"ring": (232, 76, 116), "dark": (150, 34, 70), "face": (255, 242, 230),
          "glyph": (151, 37, 73), "cloth": (255, 120, 150), "blush": (255, 150, 170)},
    BLACK: {"ring": (92, 170, 238), "dark": (30, 72, 130), "face": (233, 245, 253),
            "glyph": (30, 67, 117), "cloth": (130, 200, 255), "blush": (255, 170, 190)},
}

EXPRESSIONS = ("idle", "fierce", "smug", "scared", "hurt", "ko", "cheer", "sad", "sleepy", "surprised")


def team(side: str) -> dict[str, tuple[int, int, int]]:
    return TEAM[side]


def _ow(R: float) -> int:
    return max(2, int(R * 0.075))


def octagon(cx: float, cy: float, half: float, cut: float) -> list[tuple[float, float]]:
    return [
        (cx - half + cut, cy - half), (cx + half - cut, cy - half),
        (cx + half, cy - half + cut), (cx + half, cy + half - cut),
        (cx + half - cut, cy + half), (cx - half + cut, cy + half),
        (cx - half, cy + half - cut), (cx - half, cy - half + cut),
    ]


def ink_polygon(surf: pygame.Surface, color, points, w: int) -> None:
    pts = [(int(x), int(y)) for x, y in points]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, INK, pts, w)


def ink_circle(surf: pygame.Surface, color, center, radius: float, w: int) -> None:
    c = (int(center[0]), int(center[1]))
    r = max(1, int(radius))
    pygame.draw.circle(surf, INK, c, r + w // 2 + 1)
    pygame.draw.circle(surf, color, c, max(1, r - w // 2))


def ink_ellipse(surf: pygame.Surface, color, rect, w: int) -> None:
    r = pygame.Rect(rect)
    pygame.draw.ellipse(surf, INK, r.inflate(w, w))
    pygame.draw.ellipse(surf, color, r.inflate(-w, -w))


def ink_line(surf: pygame.Surface, color, a, b, width: float, w: int) -> None:
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    pygame.draw.line(surf, INK, a, b, int(width) + 2 * w)
    pygame.draw.circle(surf, INK, a, int(width / 2) + w)
    pygame.draw.circle(surf, INK, b, int(width / 2) + w)
    pygame.draw.line(surf, color, a, b, max(1, int(width)))
    pygame.draw.circle(surf, color, a, max(1, int(width / 2)))
    pygame.draw.circle(surf, color, b, max(1, int(width / 2)))


def rot(vx: float, vy: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return vx * c - vy * s, vx * s + vy * c


class ActorSpec:
    """Geometry of a rendered actor surface."""

    def __init__(self, R: float):
        self.R = R
        self.W = int(R * 5.2)
        self.H = int(R * 4.6)
        self.cx = self.W / 2
        self.cy = R * 2.25
        self.ground = (self.cx, self.cy + R * 1.18)
        self.glyph: Optional[pygame.Surface] = None
        self.glyph_center = (self.cx, self.cy)


def draw_face(surf: pygame.Surface, cx: float, cy: float, R: float, expr: str, blink: bool, colors, t: float) -> None:
    w = max(1, int(R * 0.05))
    fx = R * 0.1
    ey = cy - R * 0.44
    ex = (cx + fx - R * 0.3, cx + fx + R * 0.3)
    er = R * 0.15
    blush = colors["blush"]
    for x in ex:
        pygame.draw.ellipse(surf, blush, (int(x - R * 0.13), int(ey + R * 0.15), int(R * 0.26), int(R * 0.12)))

    def brows(angle: float, lift: float = 0.0) -> None:
        for i, x in enumerate(ex):
            sgn = -1 if i == 0 else 1
            a = (x - R * 0.14, ey - R * 0.26 - lift + sgn * angle * R)
            b = (x + R * 0.14, ey - R * 0.26 - lift - sgn * angle * R)
            pygame.draw.line(surf, INK, a, b, max(2, int(R * 0.07)))

    mouth_y = cy - R * 0.17
    mx = cx + fx
    if expr in ("ko",):
        for x in ex:
            d = er * 0.9
            pygame.draw.line(surf, INK, (x - d, ey - d), (x + d, ey + d), max(2, int(R * 0.07)))
            pygame.draw.line(surf, INK, (x - d, ey + d), (x + d, ey - d), max(2, int(R * 0.07)))
        pygame.draw.ellipse(surf, INK, (int(mx - R * 0.12), int(mouth_y - R * 0.04), int(R * 0.24), int(R * 0.12)))
        pygame.draw.ellipse(surf, (255, 120, 140), (int(mx - R * 0.02), int(mouth_y + R * 0.02), int(R * 0.12), int(R * 0.12)))
        return
    if expr == "hurt":
        for i, x in enumerate(ex):
            sgn = 1 if i == 0 else -1
            pts = [(x - er * sgn, ey - er), (x + er * sgn, ey), (x - er * sgn, ey + er)]
            pygame.draw.lines(surf, INK, False, pts, max(2, int(R * 0.07)))
        pts = [(mx - R * 0.16 + i * R * 0.08, mouth_y + (R * 0.04 if i % 2 else -R * 0.02)) for i in range(5)]
        pygame.draw.lines(surf, INK, False, pts, max(2, int(R * 0.05)))
        return
    if expr == "cheer":
        for x in ex:
            pygame.draw.arc(surf, INK, (int(x - er), int(ey - er * 0.6), int(er * 2), int(er * 1.6)), 0.2, math.pi - 0.2, max(2, int(R * 0.07)))
        rect = pygame.Rect(int(mx - R * 0.17), int(mouth_y - R * 0.08), int(R * 0.34), int(R * 0.22))
        pygame.draw.ellipse(surf, INK, rect)
        pygame.draw.ellipse(surf, (255, 120, 140), rect.inflate(-int(R * 0.1), -int(R * 0.1)).move(0, int(R * 0.03)))
        return
    if expr == "smug" or expr == "sleepy" or blink:
        for x in ex:
            pygame.draw.line(surf, INK, (x - er, ey), (x + er, ey), max(2, int(R * 0.07)))
            if expr == "smug":
                pygame.draw.arc(surf, INK, (int(x - er), int(ey - er), int(er * 2), int(er * 2)), math.pi, math.tau, max(1, int(R * 0.04)))
        if expr == "smug":
            brows(-0.05)
            pygame.draw.arc(surf, INK, (int(mx - R * 0.14), int(mouth_y - R * 0.12), int(R * 0.3), int(R * 0.18)), math.pi + 0.3, math.tau - 0.1, max(2, int(R * 0.05)))
        else:
            pygame.draw.arc(surf, INK, (int(mx - R * 0.1), int(mouth_y - R * 0.08), int(R * 0.2), int(R * 0.12)), math.pi + 0.4, math.tau - 0.4, max(1, int(R * 0.04)))
        return

    pupil_scale = {"scared": 0.38, "surprised": 0.45, "fierce": 0.7}.get(expr, 0.62)
    white_scale = {"scared": 1.25, "surprised": 1.2}.get(expr, 1.0)
    look = R * 0.04
    for x in ex:
        pygame.draw.circle(surf, INK, (int(x), int(ey)), int(er * white_scale) + w)
        pygame.draw.circle(surf, WHITE, (int(x), int(ey)), int(er * white_scale))
        pr = max(1, int(er * pupil_scale))
        jitter = math.sin(t * 0.09) * R * 0.02 if expr == "scared" else 0
        pygame.draw.circle(surf, INK, (int(x + look + jitter), int(ey + R * 0.01)), pr)
        pygame.draw.circle(surf, WHITE, (int(x + look + pr * 0.4 + jitter), int(ey - pr * 0.4)), max(1, pr // 3))

    if expr == "fierce":
        brows(0.09)
        pygame.draw.rect(surf, INK, (int(mx - R * 0.17), int(mouth_y - R * 0.06), int(R * 0.34), int(R * 0.15)), border_radius=int(R * 0.05))
        pygame.draw.rect(surf, WHITE, (int(mx - R * 0.13), int(mouth_y - R * 0.04), int(R * 0.26), int(R * 0.06)))
    elif expr in ("scared", "surprised"):
        brows(-0.07, R * 0.05)
        pygame.draw.ellipse(surf, INK, (int(mx - R * 0.08), int(mouth_y - R * 0.08), int(R * 0.16), int(R * 0.2)))
        if expr == "scared":
            sx, sy = cx + R * 0.7, cy - R * 0.85 + (t * 0.03 % (R * 0.3))
            pts = [(sx, sy - R * 0.18), (sx + R * 0.1, sy + R * 0.02), (sx, sy + R * 0.1), (sx - R * 0.1, sy + R * 0.02)]
            pygame.draw.polygon(surf, (150, 210, 255), pts)
            pygame.draw.polygon(surf, INK, pts, max(1, int(R * 0.03)))
    elif expr == "sad":
        brows(-0.08)
        pygame.draw.arc(surf, INK, (int(mx - R * 0.12), int(mouth_y - R * 0.02), int(R * 0.24), int(R * 0.16)), 0.3, math.pi - 0.3, max(2, int(R * 0.05)))
        for x in ex:
            ty = ey + er + (t * 0.05 % (R * 0.4))
            pygame.draw.ellipse(surf, (120, 190, 255), (int(x - R * 0.04), int(ty), int(R * 0.08), int(R * 0.13)))
    else:
        pygame.draw.arc(surf, INK, (int(mx - R * 0.12), int(mouth_y - R * 0.1), int(R * 0.24), int(R * 0.16)), math.pi + 0.3, math.tau - 0.3, max(2, int(R * 0.05)))


def render_actor(
    kind: str,
    side: str,
    R: float,
    expr: str = "idle",
    t: float = 0.0,
    walk: float = 0.0,
    swing: float = 0.0,
    props: Optional[set[str]] = None,
) -> tuple[pygame.Surface, ActorSpec]:
    """Draw the character upright, facing right.

    ``walk`` is a gait phase in radians (0 = standing), ``swing`` 0..1 animates the
    signature attack (lance thrust, sword slash, fan sweep, kick, stomp, recoil).
    ``props`` can contain: no_hat, no_weapon, no_wheels, no_barrel, no_tusks, soot.
    """
    props = props or set()
    spec = ActorSpec(R)
    surf = pygame.Surface((spec.W, spec.H), pygame.SRCALPHA)
    cx, cy = spec.cx, spec.cy
    col = team(side)
    w = _ow(R)
    gy = spec.ground[1]
    blink = (int(t) // 90) % 41 == 0 and expr in ("idle", "fierce", "cheer")
    walking = abs(walk) > 1e-6
    bob = abs(math.sin(walk)) * R * 0.08 if walking else math.sin(t / 380.0) * R * 0.025
    cy -= bob

    # --- behind the body -------------------------------------------------
    if kind == KING:
        wave = math.sin(t / 160.0) * R * 0.12 + (R * 0.2 if walking else 0)
        cape = [(cx - R * 0.5, cy - R * 0.6), (cx - R * 1.35 - wave, cy + R * 0.9), (cx - R * 0.6, cy + R * 1.05), (cx + R * 0.1, cy + R * 0.2)]
        ink_polygon(surf, col["dark"], cape, w)
        pygame.draw.line(surf, GOLD, (int(cx - R * 1.3 - wave), int(cy + R * 0.86)), (int(cx - R * 0.62), int(cy + R * 1.0)), max(2, int(R * 0.06)))
    elif kind == ELEPHANT:
        flap = math.sin(t / 200.0) * R * 0.06
        ink_ellipse(surf, col["cloth"], (cx - R * 1.55 - flap, cy - R * 0.9, R * 1.1, R * 1.45), w)
        pygame.draw.ellipse(surf, col["blush"], (int(cx - R * 1.38 - flap), int(cy - R * 0.62), int(R * 0.7), int(R * 0.9)))
        ink_ellipse(surf, col["cloth"], (cx + R * 0.35 + flap * 0.5, cy - R * 1.05, R * 0.75, R * 1.0), w)
    elif kind == HORSE:
        sway = math.sin(t / 140.0 + walk) * R * 0.2
        tail = [(cx - R * 0.9, cy + R * 0.0), (cx - R * 1.6, cy + R * 0.2 + sway), (cx - R * 1.75, cy + R * 0.85 + sway), (cx - R * 1.2, cy + R * 0.55), (cx - R * 0.9, cy + R * 0.35)]
        ink_polygon(surf, col["dark"], tail, w)
    elif kind == PAWN:
        flutter = math.sin(t / 90.0) * R * 0.12
        scarf = [(cx - R * 0.6, cy - R * 0.55), (cx - R * 1.5, cy - R * 0.55 + flutter), (cx - R * 1.4, cy - R * 0.25 - flutter), (cx - R * 0.6, cy - R * 0.3)]
        ink_polygon(surf, col["ring"], scarf, w)

    # --- legs / wheels ---------------------------------------------------
    if kind == ROOK:
        if "no_wheels" not in props:
            for i, wx in enumerate((cx - R * 0.62, cx + R * 0.62)):
                wr = R * 0.4
                wy = gy - wr
                ink_circle(surf, WOOD, (wx, wy), wr, w)
                pygame.draw.circle(surf, (110, 70, 40), (int(wx), int(wy)), int(wr * 0.62), max(1, int(R * 0.05)))
                for k in range(4):
                    a = walk * 1.6 + k * math.pi / 4 + i
                    dx, dy = math.cos(a) * wr * 0.85, math.sin(a) * wr * 0.85
                    pygame.draw.line(surf, INK, (wx - dx, wy - dy), (wx + dx, wy + dy), max(1, int(R * 0.05)))
                pygame.draw.circle(surf, GOLD, (int(wx), int(wy)), max(2, int(wr * 0.22)))
    elif kind == HORSE:
        for i, lx in enumerate((-0.6, -0.25, 0.25, 0.6)):
            phase = walk + (i % 2) * math.pi
            lift = max(0.0, math.sin(phase)) * R * 0.25 if walking else 0
            kick = 0.0
            if swing > 0 and i >= 2:
                kick = math.sin(min(1.0, swing) * math.pi) * R * 0.7
                lift += kick * 0.4
            top = (cx + lx * R, cy + R * 0.5)
            foot = (cx + lx * R + kick + math.cos(phase) * R * 0.12 * walking, gy - lift - R * 0.08)
            ink_line(surf, col["face"], top, foot, R * 0.16, max(1, w - 1))
            pygame.draw.rect(surf, INK, (int(foot[0] - R * 0.12), int(foot[1] - R * 0.02), int(R * 0.24), int(R * 0.12)), border_radius=int(R * 0.03))
    else:
        foot_w = R * (0.62 if kind == ELEPHANT else 0.44)
        foot_h = R * (0.32 if kind == ELEPHANT else 0.24)
        for i, base in enumerate((-0.36, 0.36)):
            phase = walk + i * math.pi
            fx = cx + base * R + (math.sin(phase) * R * 0.22 if walking else 0)
            lift = max(0.0, math.cos(phase)) * R * 0.2 if walking else 0
            color = (170, 170, 190) if kind == ELEPHANT else col["dark"]
            ink_ellipse(surf, color, (fx - foot_w / 2, gy - foot_h - lift, foot_w, foot_h), max(1, w - 1))

    # --- body tile -------------------------------------------------------
    ink_polygon(surf, col["ring"], octagon(cx, cy, R + w * 0.5, R * 0.38), w)
    pygame.draw.polygon(surf, col["face"], [(int(x), int(y)) for x, y in octagon(cx, cy - R * 0.02, R * 0.8, R * 0.3)])
    pygame.draw.polygon(surf, col["dark"], [(int(x), int(y)) for x, y in octagon(cx, cy, R * 0.8, R * 0.3)], max(1, int(R * 0.04)))
    pygame.draw.line(surf, (255, 255, 255), (int(cx - R * 0.5), int(cy - R * 0.92)), (int(cx + R * 0.5), int(cy - R * 0.92)), max(1, int(R * 0.05)))
    spec.glyph = font(R * 0.78).render(KIND_LABEL[(side, kind)], True, col["glyph"])
    spec.glyph_center = (cx, cy + R * 0.36)
    if "soot" in props:
        for i in range(6):
            a = i * 1.7
            pygame.draw.circle(surf, (60, 50, 60), (int(cx + math.cos(a) * R * 0.45), int(cy + math.sin(a) * R * 0.4)), int(R * (0.12 + 0.05 * (i % 3))))

    # --- face features ----------------------------------------------------
    if kind == ELEPHANT:
        draw_face(surf, cx - R * 0.08, cy, R, expr, blink, col, t)
        lift = swing * 1.6
        base = (cx + R * 0.55, cy - R * 0.25)
        pts = [base]
        for k in range(1, 6):
            u = k / 5
            ang = 0.3 + u * (1.2 - lift) + math.sin(t / 220.0) * 0.08
            px, py = pts[-1]
            pts.append((px + math.cos(ang - 0.9) * R * 0.22, py + math.sin(ang - 0.9) * R * 0.22 + R * 0.05))
        for a, b in zip(pts, pts[1:]):
            ink_line(surf, (180, 180, 200), a, b, R * 0.2 * (1 - 0.1), max(1, w - 1))
        for a, b in zip(pts, pts[1:]):
            pygame.draw.line(surf, (180, 180, 200), a, b, max(2, int(R * 0.18)))
        if "no_tusks" not in props:
            tusk = [(cx + R * 0.35, cy - R * 0.05), (cx + R * 0.65, cy + R * 0.05), (cx + R * 0.42, cy + R * 0.12)]
            ink_polygon(surf, (255, 252, 240), tusk, max(1, w - 1))
    elif kind == HORSE:
        draw_face(surf, cx - R * 0.12, cy, R, expr, blink, col, t)
        snout = (cx + R * 0.62, cy - R * 0.32, R * 0.62, R * 0.46)
        ink_ellipse(surf, col["face"], snout, max(1, w - 1))
        pygame.draw.circle(surf, INK, (int(cx + R * 1.06), int(cy - R * 0.16)), max(1, int(R * 0.05)))
        if expr in ("cheer", "fierce"):
            pygame.draw.arc(surf, INK, (int(cx + R * 0.7), int(cy - R * 0.2), int(R * 0.4), int(R * 0.22)), math.pi + 0.3, math.tau - 0.2, max(1, int(R * 0.04)))
    else:
        draw_face(surf, cx, cy, R, expr, blink, col, t)

    # --- headgear --------------------------------------------------------
    top = cy - R - w * 0.5
    if "no_hat" not in props:
        if kind == KING:
            crown = [(cx - R * 0.55, top + R * 0.1), (cx - R * 0.6, top - R * 0.5), (cx - R * 0.3, top - R * 0.22),
                     (cx, top - R * 0.68), (cx + R * 0.3, top - R * 0.22), (cx + R * 0.6, top - R * 0.5), (cx + R * 0.55, top + R * 0.1)]
            ink_polygon(surf, GOLD, crown, w)
            for jx, color in ((-0.3, (255, 90, 120)), (0.0, (90, 200, 255)), (0.3, (120, 230, 150))):
                pygame.draw.circle(surf, color, (int(cx + jx * R), int(top - R * 0.06)), max(2, int(R * 0.08)))
        elif kind == ADVISOR:
            wobble = math.sin(t / 150.0) * R * 0.05
            for sgn in (-1, 1):
                wing = (cx + sgn * R * 0.55 - (R * 0.55 if sgn < 0 else 0), top - R * 0.3 + wobble * sgn, R * 0.55, R * 0.16)
                ink_ellipse(surf, (40, 36, 54), wing, max(1, w - 1))
            ink_polygon(surf, (40, 36, 54), [(cx - R * 0.45, top + R * 0.1), (cx - R * 0.4, top - R * 0.45), (cx + R * 0.4, top - R * 0.45), (cx + R * 0.45, top + R * 0.1)], w)
            pygame.draw.rect(surf, GOLD, (int(cx - R * 0.42), int(top - R * 0.05), int(R * 0.84), max(2, int(R * 0.08))))
        elif kind == ROOK:
            dome = [(cx + math.cos(a) * R * 0.62, top + R * 0.1 - math.sin(a) * R * 0.55) for a in [k * math.pi / 12 for k in range(13)]]
            ink_polygon(surf, STEEL, dome, w)
            pygame.draw.line(surf, (235, 240, 250), (int(cx - R * 0.3), int(top - R * 0.28)), (int(cx + R * 0.05), int(top - R * 0.4)), max(1, int(R * 0.06)))
            sway = math.sin(t / 120.0) * R * 0.1
            plume = [(cx, top - R * 0.45), (cx - R * 0.5 - sway, top - R * 0.95), (cx - R * 0.15, top - R * 0.85), (cx + R * 0.15, top - R * 0.5)]
            ink_polygon(surf, col["ring"], plume, max(1, w - 1))
        elif kind == PAWN:
            hat = [(cx - R * 1.05, top + R * 0.12), (cx, top - R * 0.58), (cx + R * 1.05, top + R * 0.12)]
            ink_polygon(surf, STRAW, hat, w)
            pygame.draw.line(surf, col["ring"], (int(cx - R * 0.6), int(top - R * 0.12)), (int(cx + R * 0.6), int(top - R * 0.12)), max(2, int(R * 0.1)))
            for k in range(-2, 3):
                pygame.draw.line(surf, (190, 150, 80), (int(cx), int(top - R * 0.55)), (int(cx + k * R * 0.4), int(top + R * 0.08)), 1)
        elif kind == HORSE:
            for sgn in (-1, 1):
                ear = [(cx + sgn * R * 0.32 - R * 0.15, top + R * 0.12), (cx + sgn * R * 0.42, top - R * 0.48), (cx + sgn * R * 0.32 + R * 0.18, top + R * 0.12)]
                ink_polygon(surf, col["ring"], ear, max(1, w - 1))
            for k in range(5):
                mx = cx - R * 0.15 - k * R * 0.22
                my = top + k * R * 0.2
                flick = math.sin(t / 110.0 + k) * R * 0.08
                spike = [(mx + R * 0.12, my + R * 0.05), (mx - R * 0.3 + flick, my - R * 0.22), (mx - R * 0.08, my + R * 0.2)]
                ink_polygon(surf, col["dark"], spike, max(1, w - 1))
    if kind == CANNON and "no_barrel" not in props:
        recoil = math.sin(min(1.0, swing) * math.pi) * R * 0.35
        angle = -0.18
        base = (cx - R * 0.15 - recoil, top + R * 0.02)
        length = R * 1.45
        half = R * 0.26
        dx, dy = math.cos(angle), math.sin(angle)
        nx, ny = -dy, dx
        tip = (base[0] + dx * length, base[1] + dy * length)
        barrel = [(base[0] + nx * half, base[1] + ny * half), (tip[0] + nx * half * 0.85, tip[1] + ny * half * 0.85),
                  (tip[0] - nx * half * 0.85, tip[1] - ny * half * 0.85), (base[0] - nx * half, base[1] - ny * half)]
        ink_polygon(surf, (70, 74, 92), barrel, w)
        ink_circle(surf, (90, 94, 112), tip, half * 1.05, max(1, w - 1))
        pygame.draw.circle(surf, INK, (int(tip[0]), int(tip[1])), max(1, int(half * 0.55)))
        ink_circle(surf, (90, 94, 112), base, half * 1.1, max(1, w - 1))
        pygame.draw.line(surf, GOLD, (int(base[0] + dx * length * 0.5 + nx * half), int(base[1] + dy * length * 0.5 + ny * half)),
                         (int(base[0] + dx * length * 0.5 - nx * half), int(base[1] + dy * length * 0.5 - ny * half)), max(2, int(R * 0.07)))
        fuse_end = (base[0] - R * 0.45, base[1] - R * 0.35)
        pygame.draw.line(surf, INK, (int(base[0] - half), int(base[1])), (int(fuse_end[0]), int(fuse_end[1])), max(2, int(R * 0.06)))
        flicker = 0.7 + 0.3 * math.sin(t / 30.0)
        pygame.draw.circle(surf, (255, 200, 80), (int(fuse_end[0]), int(fuse_end[1])), max(2, int(R * 0.13 * flicker)))
        pygame.draw.circle(surf, (255, 250, 210), (int(fuse_end[0]), int(fuse_end[1])), max(1, int(R * 0.06 * flicker)))

    # --- hands & weapons -------------------------------------------------
    hand_back = (cx - R * 1.0, cy + R * 0.25)
    hand_front = (cx + R * 1.0, cy + R * 0.2)
    if "no_weapon" not in props:
        if kind == ROOK:
            s = min(1.0, swing)
            thrust = math.sin(s * math.pi) * R * 0.7
            ang = -0.75 * (1 - min(1.0, s * 3))
            hx, hy = hand_front[0] + thrust, hand_front[1] + R * 0.1
            back, front = rot(-R * 0.45, 0, ang), rot(R * 1.35, 0, ang)
            a = (hx + back[0], hy + back[1])
            b = (hx + front[0], hy + front[1])
            ink_line(surf, WOOD, a, b, R * 0.1, max(1, w - 1))
            tip, side_v = rot(R * 0.42, 0, ang), rot(0, R * 0.14, ang)
            ink_polygon(surf, STEEL, [(b[0] + side_v[0], b[1] + side_v[1]), (b[0] + tip[0], b[1] + tip[1]), (b[0] - side_v[0], b[1] - side_v[1])], max(1, w - 1))
            fl = rot(-R * 0.45, -R * 0.4 + math.sin(t / 80.0) * R * 0.08, ang)
            fl2 = rot(-R * 0.55, 0, ang)
            flag = [b, (b[0] + fl[0], b[1] + fl[1]), (b[0] + fl2[0], b[1] + fl2[1])]
            ink_polygon(surf, col["ring"], flag, max(1, w - 1))
            hand_front = (hx, hy)
        elif kind == KING:
            ang = -1.3 + swing * 2.4
            hx, hy = hand_front
            bx, by = rot(R * 1.25, 0, ang)
            ink_line(surf, STEEL, (hx, hy), (hx + bx, hy + by), R * 0.14, max(1, w - 1))
            gx, gy2 = rot(0, R * 0.22, ang)
            ink_line(surf, GOLD, (hx - gx, hy - gy2), (hx + gx, hy + gy2), R * 0.1, max(1, w - 1))
        elif kind == ADVISOR:
            ang = -0.6 + swing * 1.8
            hx, hy = hand_front
            fan = [(hx, hy)]
            for k in range(7):
                a = ang - 0.75 + k * 0.25
                fan.append((hx + math.cos(a) * R * 0.75, hy + math.sin(a) * R * 0.75))
            ink_polygon(surf, (255, 250, 236), fan, max(1, w - 1))
            for p in fan[1::2]:
                pygame.draw.line(surf, (190, 160, 120), (int(hx), int(hy)), (int(p[0]), int(p[1])), 1)
            pygame.draw.arc(surf, col["ring"], (int(hx - R * 0.6), int(hy - R * 0.6), int(R * 1.2), int(R * 1.2)), -ang - 0.75, -ang + 0.75, max(2, int(R * 0.08)))
        elif kind == PAWN:
            ang = -math.pi / 2 + swing * (math.pi / 2 - 0.05)
            hx, hy = hand_front
            back = rot(-R * 0.9, 0, ang)
            front = rot(R * 1.5, 0, ang)
            a = (hx + back[0], hy + back[1])
            b = (hx + front[0], hy + front[1])
            ink_line(surf, WOOD, a, b, R * 0.09, max(1, w - 1))
            tip_dir = rot(R * 0.35, 0, ang)
            side_dir = rot(0, R * 0.12, ang)
            ink_polygon(surf, STEEL, [(b[0] + side_dir[0], b[1] + side_dir[1]), (b[0] + tip_dir[0], b[1] + tip_dir[1]), (b[0] - side_dir[0], b[1] - side_dir[1])], max(1, w - 1))
            pygame.draw.circle(surf, col["ring"], (int(b[0] - tip_dir[0] * 0.2), int(b[1] - tip_dir[1] * 0.2)), max(2, int(R * 0.1)))
    for hand in (hand_back, hand_front):
        ink_circle(surf, col["face"], hand, R * 0.17, max(1, w - 1))
    return surf, spec


def blit_actor(
    dest: pygame.Surface,
    surf: pygame.Surface,
    spec: ActorSpec,
    pos: tuple[float, float],
    facing: int = 1,
    squash: float = 0.0,
    tilt: float = 0.0,
    alpha: int = 255,
    flash: float = 0.0,
    pivot: str = "feet",
) -> None:
    """Place an actor so its feet land on ``pos``. ``tilt`` is in radians (positive = lean forward)."""
    ax, ay = spec.ground
    px, py = (ax, ay) if pivot == "feet" else (spec.cx, spec.cy)
    gx, gy = spec.glyph_center
    if facing < 0:
        surf = pygame.transform.flip(surf, True, False)
        ax = spec.W - ax
        px = spec.W - px
        gx = spec.W - gx
        tilt = -tilt
    if spec.glyph is not None:
        surf.blit(spec.glyph, spec.glyph.get_rect(center=(int(gx), int(gy))))
    sx, sy = 1.0 + squash, 1.0 - squash * 0.8
    if abs(squash) > 1e-3:
        w, h = max(1, int(spec.W * sx)), max(1, int(spec.H * sy))
        surf = pygame.transform.smoothscale(surf, (w, h))
        ax, px = ax * sx, px * sx
        ay, py = h - (spec.H - ay) * sy, h - (spec.H - py) * sy
    if flash > 0:
        surf = surf.copy()
        v = int(255 * min(1.0, flash))
        surf.fill((v, v, v, 0), special_flags=pygame.BLEND_RGBA_ADD)
    world_pivot = (pos[0] + (px - ax), pos[1] + (py - ay))
    if abs(tilt) > 1e-3:
        cw, ch = surf.get_size()
        deg = -math.degrees(tilt)
        rotated = pygame.transform.rotozoom(surf, deg, 1.0)
        th = math.radians(deg)
        vx, vy = px - cw / 2, py - ch / 2
        rx = vx * math.cos(th) + vy * math.sin(th)
        ry = -vx * math.sin(th) + vy * math.cos(th)
        nw, nh = rotated.get_size()
        top_left = (world_pivot[0] - (nw / 2 + rx), world_pivot[1] - (nh / 2 + ry))
        surf = rotated
    else:
        top_left = (world_pivot[0] - px, world_pivot[1] - py)
    if alpha < 255:
        surf = surf.copy() if surf.get_flags() & pygame.SRCALPHA else surf
        surf.set_alpha(max(0, alpha))
    dest.blit(surf, (int(top_left[0]), int(top_left[1])))


def draw_shadow(dest: pygame.Surface, x: float, y: float, R: float, alpha: int = 90, lift: float = 0.0) -> None:
    scale = max(0.35, 1.0 - lift / (R * 4))
    w, h = int(R * 2.2 * scale), int(R * 0.55 * scale)
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (10, 4, 20, int(alpha * scale)), shadow.get_rect())
    dest.blit(shadow, (int(x - w / 2), int(y - h / 2)))


def actor(
    dest: pygame.Surface, kind: str, side: str, pos: tuple[float, float], R: float, *,
    facing: int = 1, expr: str = "idle", t: float = 0.0, walk: float = 0.0, swing: float = 0.0,
    squash: float = 0.0, tilt: float = 0.0, alpha: int = 255, flash: float = 0.0,
    props: Optional[set[str]] = None, pivot: str = "feet", shadow: bool = True, lift: float = 0.0,
) -> None:
    """Render + place in one call. ``lift`` raises the body above its shadow (jumps)."""
    if shadow and alpha > 40:
        draw_shadow(dest, pos[0], pos[1], R, int(90 * alpha / 255), lift)
    surf, spec = render_actor(kind, side, R, expr, t, walk, swing, props)
    blit_actor(dest, surf, spec, (pos[0], pos[1] - lift), facing, squash, tilt, alpha, flash, pivot)
