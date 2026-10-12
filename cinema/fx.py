"""Stateless, time-driven drawing effects for the cartoon cutscenes.

Every effect is a pure function of its progress value so scenes can be scrubbed,
skipped or rendered offline frame-by-frame without keeping particle state.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Iterable, Optional

import pygame

from cinema.actors import INK
from cinema.fonts import font


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def seg(t: float, a: float, b: float) -> float:
    """Normalised progress of ``t`` within ``[a, b]``."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp((t - a) / (b - a))


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def lerp2(a, b, u: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u


def ease_out(u: float) -> float:
    return 1 - (1 - u) ** 3


def ease_in(u: float) -> float:
    return u ** 3


def ease_in_out(u: float) -> float:
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_back(u: float) -> float:
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2


def ease_bounce(u: float) -> float:
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375


def hop(u: float) -> float:
    """0 → 1 → 0 arc."""
    return math.sin(clamp(u) * math.pi)


def wobble(t: float, speed: float = 1.0, amount: float = 1.0) -> float:
    return math.sin(t * 0.012 * speed) * amount


_LAYERS: dict[tuple[int, int], pygame.Surface] = {}


def _alpha_layer(surf: pygame.Surface) -> pygame.Surface:
    """A cleared full-size scratch layer; callers must blit it before requesting another."""
    size = surf.get_size()
    layer = _LAYERS.get(size)
    if layer is None:
        layer = _LAYERS[size] = pygame.Surface(size, pygame.SRCALPHA)
    else:
        layer.fill((0, 0, 0, 0))
    return layer


def with_alpha(color, a: float) -> tuple[int, int, int, int]:
    return int(color[0]), int(color[1]), int(color[2]), int(clamp(a / 255.0) * 255)


# --- text --------------------------------------------------------------------

def outlined_text(
    surf: pygame.Surface, text: str, size: float, color, center, outline=INK, ow: int = 3,
    scale: float = 1.0, angle: float = 0.0, alpha: int = 255, bold: bool = True,
) -> pygame.Rect:
    f = font(size * scale, bold)
    base = f.render(text, True, color)
    edge = f.render(text, True, outline)
    w, h = base.get_width() + ow * 2 + 2, base.get_height() + ow * 2 + 2
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    for dx in range(-ow, ow + 1, max(1, ow // 2)):
        for dy in range(-ow, ow + 1, max(1, ow // 2)):
            if dx * dx + dy * dy <= ow * ow + 1:
                layer.blit(edge, (ow + 1 + dx, ow + 1 + dy))
    layer.blit(base, (ow + 1, ow + 1))
    if angle:
        layer = pygame.transform.rotozoom(layer, angle, 1.0)
    if alpha < 255:
        layer.set_alpha(alpha)
    rect = layer.get_rect(center=(int(center[0]), int(center[1])))
    surf.blit(layer, rect)
    return rect


def onomatopoeia(surf: pygame.Surface, text: str, center, size: float, color, u: float, angle: float = -8.0) -> None:
    """Comic sound word that pops in, holds, then fades (``u`` 0..1)."""
    if u <= 0 or u >= 1:
        return
    pop = ease_back(seg(u, 0, 0.25))
    fade = 1 - seg(u, 0.75, 1.0)
    shake = math.sin(u * 60) * 3 * (1 - seg(u, 0, 0.4))
    outlined_text(surf, text, size, color, (center[0] + shake, center[1] - u * 20), ow=max(3, int(size / 12)),
                  scale=max(0.05, pop), angle=angle, alpha=int(255 * fade))


def wrap_cjk(text: str, f: pygame.font.Font, max_width: int) -> list[str]:
    lines, line = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(line)
            line = ""
            continue
        if f.size(line + ch)[0] > max_width and line:
            lines.append(line)
            line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    return lines


def speech_bubble(
    surf: pygame.Surface, text: str, anchor, u: float, size: int = 26, max_width: int = 340,
    tail: str = "down-left", fill=(255, 255, 255), color=INK, shout: bool = False,
) -> None:
    """Rounded speech bubble whose tail points at ``anchor``; ``u`` is 0..1 lifetime."""
    if u <= 0 or u >= 1 or not text:
        return
    appear = ease_back(seg(u, 0, 0.12))
    fade = 1 - seg(u, 0.88, 1.0)
    f = font(size, True)
    shown = text[: max(1, int(len(text) * seg(u, 0.02, 0.35)))]
    lines = wrap_cjk(text, f, max_width)
    shown_lines = wrap_cjk(shown, f, max_width)
    lh = f.get_linesize()
    tw = max(f.size(line)[0] for line in lines)
    pad = int(size * 0.7)
    bw, bh = tw + pad * 2, lh * len(lines) + pad * 2 - int(size * 0.2)
    layer = pygame.Surface((bw + 40, bh + 60), pygame.SRCALPHA)
    box = pygame.Rect(20, 10, bw, bh)
    right = "right" in tail
    up = "up" in tail
    if up:
        box.y = 50
    tx = box.right - 40 if right else box.x + 40
    if up:
        tri = [(tx - 14, box.y + 4), (tx + (18 if right else -18), box.y - 36), (tx + 16, box.y + 4)]
    else:
        tri = [(tx - 14, box.bottom - 4), (tx + (18 if right else -18), box.bottom + 36), (tx + 16, box.bottom - 4)]
    if shout:
        pts = []
        cxb, cyb = box.center
        for k in range(28):
            a = k / 28 * math.tau
            rr = 1.0 if k % 2 == 0 else 0.86
            pts.append((cxb + math.cos(a) * box.w * 0.6 * rr, cyb + math.sin(a) * box.h * 0.75 * rr))
        pygame.draw.polygon(layer, INK, [(x, y) for x, y in pts])
        inner = [(cxb + (x - cxb) * 0.95, cyb + (y - cyb) * 0.93) for x, y in pts]
        pygame.draw.polygon(layer, fill, inner)
    else:
        pygame.draw.polygon(layer, INK, [(x, y + (0 if up else 2)) for x, y in tri])
        pygame.draw.rect(layer, INK, box.inflate(8, 8), border_radius=int(size * 0.9))
        pygame.draw.rect(layer, fill, box, border_radius=int(size * 0.8))
        inner_tri = [(tri[0][0] + 4, tri[0][1] + (2 if up else -2)), (tri[1][0] * 0.92 + tx * 0.08, tri[1][1] + (8 if up else -8)), (tri[2][0] - 4, tri[2][1] + (2 if up else -2))]
        pygame.draw.polygon(layer, fill, inner_tri)
    for i, line in enumerate(shown_lines):
        img = f.render(line, True, color)
        layer.blit(img, (box.x + pad, box.y + pad - int(size * 0.1) + i * lh))
    if appear != 1.0:
        layer = pygame.transform.smoothscale(layer, (max(1, int(layer.get_width() * appear)), max(1, int(layer.get_height() * appear))))
    layer.set_alpha(int(255 * fade))
    lw, lh2 = layer.get_size()
    tip_x = (tx + (18 if right else -18)) * appear
    tip_y = (box.y - 36 if up else box.bottom + 36) * appear
    surf.blit(layer, (int(anchor[0] - tip_x), int(anchor[1] - tip_y)))


# --- impacts & particles -------------------------------------------------------

def star_points(cx: float, cy: float, r_out: float, r_in: float, spikes: int, rot: float = 0.0, seed: int = 0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    pts = []
    for k in range(spikes * 2):
        a = rot + k * math.pi / spikes
        r = (r_out * (0.8 + 0.4 * rng.random())) if k % 2 == 0 else r_in
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return pts


def impact_star(surf: pygame.Surface, center, u: float, size: float, color=(255, 236, 120), seed: int = 1) -> None:
    if u <= 0 or u >= 1:
        return
    grow = ease_out(seg(u, 0, 0.3))
    fade = 1 - seg(u, 0.4, 1.0)
    layer = _alpha_layer(surf)
    r = size * (0.4 + 0.6 * grow)
    pygame.draw.polygon(layer, with_alpha(INK, 255 * fade), star_points(center[0], center[1], r * 1.08, r * 0.5, 9, u, seed))
    pygame.draw.polygon(layer, with_alpha(color, 255 * fade), star_points(center[0], center[1], r, r * 0.45, 9, u, seed))
    pygame.draw.polygon(layer, with_alpha((255, 255, 255), 255 * fade), star_points(center[0], center[1], r * 0.55, r * 0.25, 9, u + 0.3, seed + 1))
    surf.blit(layer, (0, 0))


def shockwave(surf: pygame.Surface, center, u: float, radius: float, color=(255, 255, 255), width: int = 6, flat: float = 1.0) -> None:
    if u <= 0 or u >= 1:
        return
    r = radius * ease_out(u)
    a = 255 * (1 - u)
    layer = _alpha_layer(surf)
    rect = pygame.Rect(0, 0, int(r * 2), int(r * 2 * flat))
    rect.center = (int(center[0]), int(center[1]))
    pygame.draw.ellipse(layer, with_alpha(color, a), rect, max(1, int(width * (1 - u) + 1)))
    surf.blit(layer, (0, 0))


def dust(surf: pygame.Surface, center, u: float, size: float, color=(230, 215, 190), seed: int = 2, count: int = 7) -> None:
    if u <= 0 or u >= 1:
        return
    rng = random.Random(seed)
    layer = _alpha_layer(surf)
    for _ in range(count):
        ang = rng.uniform(math.pi * 1.05, math.pi * 1.95) if rng.random() < 0.8 else rng.uniform(0, math.pi)
        dist = size * (0.4 + rng.random()) * ease_out(u)
        x = center[0] + math.cos(ang) * dist * 1.4
        y = center[1] + math.sin(ang) * dist * 0.5 - u * size * 0.3
        r = size * (0.18 + rng.random() * 0.2) * (0.6 + u)
        pygame.draw.circle(layer, with_alpha(INK, 120 * (1 - u)), (int(x), int(y)), int(r) + 2)
        pygame.draw.circle(layer, with_alpha(color, 230 * (1 - u)), (int(x), int(y)), int(r))
    surf.blit(layer, (0, 0))


def explosion(surf: pygame.Surface, center, u: float, size: float, seed: int = 3) -> None:
    if u <= 0 or u >= 1:
        return
    rng = random.Random(seed)
    layer = _alpha_layer(surf)
    cx, cy = center
    for k in range(10):
        ang = rng.uniform(0, math.tau)
        d = size * (0.3 + rng.random() * 0.8) * ease_out(u)
        r = size * (0.25 + rng.random() * 0.25) * (1 - u * 0.6)
        x, y = cx + math.cos(ang) * d, cy + math.sin(ang) * d * 0.8 - u * size * 0.4
        smoke = 140 + int(60 * rng.random())
        pygame.draw.circle(layer, with_alpha((smoke, smoke - 10, smoke - 20), 220 * (1 - u)), (int(x), int(y)), int(r))
    fire = 1 - seg(u, 0.1, 0.55)
    if fire > 0:
        r = size * (0.5 + 0.6 * ease_out(seg(u, 0, 0.3)))
        pygame.draw.polygon(layer, with_alpha((255, 120, 40), 255 * fire), star_points(cx, cy, r, r * 0.6, 10, u * 2, seed))
        pygame.draw.polygon(layer, with_alpha((255, 220, 90), 255 * fire), star_points(cx, cy, r * 0.7, r * 0.4, 10, u * 2 + 0.2, seed + 1))
        pygame.draw.circle(layer, with_alpha((255, 255, 230), 255 * fire), (int(cx), int(cy)), int(r * 0.3))
    for k in range(14):
        ang = rng.uniform(0, math.tau)
        d = size * (0.6 + rng.random() * 1.2) * ease_out(u)
        x, y = cx + math.cos(ang) * d, cy + math.sin(ang) * d + (u ** 2) * size * 0.8
        pygame.draw.circle(layer, with_alpha((255, 230, 120), 255 * (1 - u)), (int(x), int(y)), max(1, int(size * 0.05)))
    surf.blit(layer, (0, 0))


def sparkles(surf: pygame.Surface, center, t: float, radius: float, color=(255, 240, 160), count: int = 8, seed: int = 4, alpha: float = 255) -> None:
    rng = random.Random(seed)
    layer = _alpha_layer(surf)
    for k in range(count):
        phase = rng.random()
        u = (t / 900.0 + phase) % 1.0
        ang = rng.uniform(0, math.tau)
        d = radius * (0.3 + 0.7 * rng.random())
        x = center[0] + math.cos(ang) * d
        y = center[1] + math.sin(ang) * d - u * radius * 0.5
        s = radius * 0.12 * hop(u)
        a = alpha * hop(u)
        pts = [(x, y - s * 2), (x + s * 0.5, y - s * 0.5), (x + s * 2, y), (x + s * 0.5, y + s * 0.5),
               (x, y + s * 2), (x - s * 0.5, y + s * 0.5), (x - s * 2, y), (x - s * 0.5, y - s * 0.5)]
        if s > 0.5:
            pygame.draw.polygon(layer, with_alpha(color, a), pts)
    surf.blit(layer, (0, 0))


def dizzy_stars(surf: pygame.Surface, center, t: float, radius: float, alpha: float = 255) -> None:
    layer = _alpha_layer(surf)
    for k in range(3):
        a = t / 220.0 + k * math.tau / 3
        x = center[0] + math.cos(a) * radius
        y = center[1] + math.sin(a) * radius * 0.35
        s = radius * 0.28
        pygame.draw.polygon(layer, with_alpha(INK, alpha), star_points(x, y, s * 1.2, s * 0.55, 5, a))
        pygame.draw.polygon(layer, with_alpha((255, 225, 90), alpha), star_points(x, y, s, s * 0.45, 5, a))
    surf.blit(layer, (0, 0))


def speed_lines(surf: pygame.Surface, rect: pygame.Rect, t: float, direction: int = 1, color=(255, 255, 255), count: int = 16, alpha: float = 160, seed: int = 5) -> None:
    rng = random.Random(seed)
    layer = _alpha_layer(surf)
    for k in range(count):
        y = rect.y + rng.random() * rect.h
        length = rect.w * (0.15 + rng.random() * 0.3)
        speed = 1.2 + rng.random()
        x = rect.x + ((rng.random() * rect.w + t * speed * direction) % (rect.w + length)) - length
        if direction < 0:
            x = rect.right - (x - rect.x) - length
        pygame.draw.line(layer, with_alpha(color, alpha * (0.4 + 0.6 * rng.random())), (int(x), int(y)), (int(x + length), int(y)), 2 + k % 3)
    surf.blit(layer, (0, 0))


def radial_lines(surf: pygame.Surface, center, t: float, inner: float, outer: float, color=(255, 255, 255), count: int = 28, alpha: float = 150, seed: int = 6) -> None:
    rng = random.Random(seed)
    layer = _alpha_layer(surf)
    for k in range(count):
        a = rng.uniform(0, math.tau) + t * 0.0004
        r0 = inner * (0.8 + 0.5 * rng.random())
        w = 2 + int(rng.random() * 6)
        p0 = (center[0] + math.cos(a) * r0, center[1] + math.sin(a) * r0)
        p1 = (center[0] + math.cos(a) * outer, center[1] + math.sin(a) * outer)
        pa = (center[0] + math.cos(a + 0.012 * w) * outer, center[1] + math.sin(a + 0.012 * w) * outer)
        pygame.draw.polygon(layer, with_alpha(color, alpha), [p0, p1, pa])
    surf.blit(layer, (0, 0))


def confetti(surf: pygame.Surface, rect: pygame.Rect, t: float, count: int = 60, seed: int = 7, alpha: float = 255) -> None:
    rng = random.Random(seed)
    colors = [(255, 110, 140), (255, 210, 90), (120, 220, 255), (150, 240, 170), (220, 160, 255)]
    layer = _alpha_layer(surf)
    for k in range(count):
        x0 = rect.x + rng.random() * rect.w
        speed = 0.08 + rng.random() * 0.12
        y = rect.y + ((t * speed + rng.random() * rect.h) % (rect.h + 40)) - 20
        x = x0 + math.sin(t / 300.0 + k) * 20
        a = t / 200.0 + k
        w, h = 10 * abs(math.cos(a)) + 2, 6
        pygame.draw.rect(layer, with_alpha(colors[k % len(colors)], alpha), (int(x), int(y), int(w), h))
    surf.blit(layer, (0, 0))


def ghost(surf: pygame.Surface, center, u: float, size: float, color=(240, 245, 255)) -> None:
    """A little soul drifting up from a defeated piece."""
    if u <= 0 or u >= 1:
        return
    a = 200 * hop(u)
    x = center[0] + math.sin(u * 9) * size * 0.3
    y = center[1] - u * size * 3
    layer = _alpha_layer(surf)
    body = [(x - size * 0.4, y), (x - size * 0.4, y + size * 0.5)]
    for k in range(5):
        body.append((x - size * 0.4 + k * size * 0.2, y + size * (0.5 if k % 2 == 0 else 0.35) + math.sin(u * 20 + k) * size * 0.05))
    body += [(x + size * 0.4, y + size * 0.5), (x + size * 0.4, y)]
    pygame.draw.circle(layer, with_alpha(color, a), (int(x), int(y)), int(size * 0.4))
    pygame.draw.polygon(layer, with_alpha(color, a), body)
    for sgn in (-1, 1):
        pygame.draw.circle(layer, with_alpha(INK, a), (int(x + sgn * size * 0.14), int(y - size * 0.02)), max(1, int(size * 0.06)))
    pygame.draw.circle(layer, with_alpha((255, 230, 120), a), (int(x), int(y - size * 0.62)), int(size * 0.25), max(1, int(size * 0.06)))
    surf.blit(layer, (0, 0))


def sweat_drop(surf: pygame.Surface, pos, size: float, t: float = 0.0) -> None:
    x, y = pos[0], pos[1] + (t * 0.02) % (size * 0.6)
    pts = [(x, y - size), (x + size * 0.55, y + size * 0.15), (x, y + size * 0.6), (x - size * 0.55, y + size * 0.15)]
    pygame.draw.polygon(surf, INK, [(px, py) for px, py in pts])
    inner = [(x, y - size * 0.75), (x + size * 0.4, y + size * 0.15), (x, y + size * 0.45), (x - size * 0.4, y + size * 0.15)]
    pygame.draw.polygon(surf, (150, 215, 255), inner)


def alert_mark(surf: pygame.Surface, pos, size: float, color=(255, 90, 110)) -> None:
    outlined_text(surf, "!", size, color, pos, ow=max(2, int(size / 10)))


def arrow_projectile(surf: pygame.Surface, a, b, u: float, color=(70, 74, 92), size: float = 14.0, arc: float = 0.0) -> Optional[tuple[float, float]]:
    """A cannonball flying from ``a`` to ``b`` over an arc; returns its position."""
    if u <= 0 or u >= 1:
        return None
    x, y = lerp2(a, b, u)
    y -= hop(u) * arc
    for k in range(1, 6):
        uu = max(0.0, u - k * 0.03)
        tx, ty = lerp2(a, b, uu)
        ty -= hop(uu) * arc
        pygame.draw.circle(surf, (255, 200 - k * 20, 120 - k * 15), (int(tx), int(ty)), max(1, int(size * (1 - k * 0.15))))
    pygame.draw.circle(surf, INK, (int(x), int(y)), int(size) + 2)
    pygame.draw.circle(surf, color, (int(x), int(y)), int(size))
    pygame.draw.circle(surf, (200, 205, 220), (int(x - size * 0.3), int(y - size * 0.3)), max(1, int(size * 0.3)))
    return x, y


def slash_arc(surf: pygame.Surface, center, radius: float, u: float, start: float, sweep: float, color=(255, 240, 170), width: float = 18) -> None:
    if u <= 0 or u >= 1:
        return
    layer = _alpha_layer(surf)
    head = start + sweep * ease_out(seg(u, 0, 0.5))
    tail = start + sweep * ease_in(seg(u, 0.1, 0.9))
    pts_out, pts_in = [], []
    steps = 18
    for k in range(steps + 1):
        a = tail + (head - tail) * k / steps
        wk = width * math.sin(k / steps * math.pi)
        pts_out.append((center[0] + math.cos(a) * (radius + wk / 2), center[1] + math.sin(a) * (radius + wk / 2)))
        pts_in.append((center[0] + math.cos(a) * (radius - wk / 2), center[1] + math.sin(a) * (radius - wk / 2)))
    poly = pts_out + pts_in[::-1]
    if len(poly) >= 3:
        pygame.draw.polygon(layer, with_alpha(color, 255 * (1 - seg(u, 0.6, 1))), poly)
        pygame.draw.lines(layer, with_alpha((255, 255, 255), 255 * (1 - seg(u, 0.5, 1))), False, pts_out, 3)
    surf.blit(layer, (0, 0))


# --- scenery -------------------------------------------------------------------

@lru_cache(maxsize=64)
def gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> pygame.Surface:
    w, h = size
    surf = pygame.Surface((w, h))
    for y in range(h):
        u = y / max(1, h - 1)
        pygame.draw.line(surf, (int(lerp(top[0], bottom[0], u)), int(lerp(top[1], bottom[1], u)), int(lerp(top[2], bottom[2], u))), (0, y), (w, y))
    return surf


def cloud(surf: pygame.Surface, x: float, y: float, scale: float, color=(255, 255, 255), alpha: float = 230, outline: bool = True) -> None:
    blobs = [(-0.9, 0.15, 0.5), (-0.35, -0.2, 0.65), (0.3, -0.25, 0.6), (0.85, 0.1, 0.5), (0.0, 0.2, 0.6)]
    layer = pygame.Surface((int(scale * 4), int(scale * 2.4)), pygame.SRCALPHA)
    ox, oy = layer.get_width() / 2, layer.get_height() / 2
    if outline:
        for bx, by, br in blobs:
            pygame.draw.circle(layer, with_alpha(INK, alpha), (int(ox + bx * scale), int(oy + by * scale)), int(br * scale) + 3)
    for bx, by, br in blobs:
        pygame.draw.circle(layer, with_alpha(color, alpha), (int(ox + bx * scale), int(oy + by * scale)), int(br * scale))
    pygame.draw.circle(layer, with_alpha((255, 255, 255), alpha * 0.6), (int(ox - 0.35 * scale), int(oy - 0.4 * scale)), int(0.25 * scale))
    surf.blit(layer, (int(x - ox), int(y - oy)))


def mountains(surf: pygame.Surface, rect: pygame.Rect, color, seed: int = 0, peaks: int = 6, height: float = 0.5, outline: bool = True) -> None:
    rng = random.Random(seed)
    pts = [(rect.x, rect.bottom)]
    for k in range(peaks * 2 + 1):
        x = rect.x + rect.w * k / (peaks * 2)
        hgt = rect.h * height * (0.5 + 0.5 * rng.random()) if k % 2 else rect.h * height * 0.25 * rng.random()
        pts.append((x, rect.bottom - hgt))
    pts.append((rect.right, rect.bottom))
    pygame.draw.polygon(surf, color, pts)
    if outline:
        pygame.draw.lines(surf, INK, False, pts[1:-1], 3)


def waves(surf: pygame.Surface, rect: pygame.Rect, t: float, base=(70, 150, 210), crest=(200, 235, 255), rows: int = 4) -> None:
    pygame.draw.rect(surf, base, rect)
    for r in range(rows):
        y = rect.y + 10 + r * (rect.h / rows)
        pts = []
        for k in range(0, rect.w + 40, 20):
            x = rect.x + k - (t * (0.04 + r * 0.015)) % 40
            pts.append((x, y + math.sin((k + r * 30) / 40.0 + t / 400.0) * 5))
        if len(pts) > 1:
            pygame.draw.lines(surf, crest, False, pts, 3)


def grass(surf: pygame.Surface, rect: pygame.Rect, t: float, color=(90, 170, 90), seed: int = 8, count: int = 40) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        x = rect.x + rng.random() * rect.w
        y = rect.y + rng.random() * rect.h
        h = 10 + rng.random() * 18
        sway = math.sin(t / 500.0 + x * 0.05) * 4
        for dx in (-5, 0, 5):
            pygame.draw.line(surf, color, (int(x + dx * 0.4), int(y)), (int(x + dx + sway), int(y - h)), 3)


def palace(surf: pygame.Surface, cx: float, ground: float, scale: float, wall=(196, 64, 64), roof=(50, 60, 90), gold=(250, 198, 92)) -> pygame.Rect:
    """A stylised palace gate with upturned eaves; returns the door rect."""
    w = 360 * scale
    body = pygame.Rect(int(cx - w / 2), int(ground - 160 * scale), int(w), int(160 * scale))
    pygame.draw.rect(surf, INK, body.inflate(8, 8))
    pygame.draw.rect(surf, wall, body)
    for k in range(5):
        x = body.x + (k + 0.5) * body.w / 5
        pygame.draw.rect(surf, (150, 40, 40), (int(x - 9 * scale), body.y, int(18 * scale), body.h))
    door = pygame.Rect(0, 0, int(110 * scale), int(120 * scale))
    door.midbottom = (int(cx), int(ground))
    pygame.draw.rect(surf, INK, door.inflate(8, 8), border_top_left_radius=int(55 * scale), border_top_right_radius=int(55 * scale))
    pygame.draw.rect(surf, (120, 30, 30), door, border_top_left_radius=int(55 * scale), border_top_right_radius=int(55 * scale))
    for i in range(3):
        for j in range(4):
            pygame.draw.circle(surf, gold, (int(door.x + door.w * (0.2 + 0.25 * i) - 4), int(door.y + door.h * (0.3 + 0.17 * j))), max(2, int(5 * scale)))
    roof_pts = [(cx - w * 0.68, body.y - 4), (cx - w * 0.6, body.y - 22 * scale), (cx - w * 0.42, body.y - 62 * scale),
                (cx + w * 0.42, body.y - 62 * scale), (cx + w * 0.6, body.y - 22 * scale), (cx + w * 0.68, body.y - 4),
                (cx + w * 0.5, body.y + 8 * scale), (cx - w * 0.5, body.y + 8 * scale)]
    pygame.draw.polygon(surf, roof, roof_pts)
    pygame.draw.polygon(surf, INK, roof_pts, 4)
    pygame.draw.line(surf, gold, (int(cx - w * 0.42), int(body.y - 62 * scale)), (int(cx + w * 0.42), int(body.y - 62 * scale)), max(3, int(8 * scale)))
    for sgn in (-1, 1):
        tip = (cx + sgn * w * 0.7, body.y - 30 * scale)
        pygame.draw.circle(surf, gold, (int(tip[0]), int(tip[1])), max(3, int(8 * scale)))
    return door


def bamboo(surf: pygame.Surface, x: float, ground: float, height: float, t: float, color=(90, 160, 90)) -> None:
    sway = math.sin(t / 700.0 + x) * 8
    segs = 6
    prev = (x, ground)
    for k in range(1, segs + 1):
        u = k / segs
        cur = (x + sway * u * u, ground - height * u)
        pygame.draw.line(surf, INK, prev, cur, 14)
        pygame.draw.line(surf, color, prev, cur, 9)
        pygame.draw.line(surf, INK, (cur[0] - 7, cur[1]), (cur[0] + 7, cur[1]), 3)
        if k % 2 == 0:
            leaf = [(cur[0], cur[1]), (cur[0] + 40, cur[1] - 12 + sway * 0.3), (cur[0] + 8, cur[1] + 6)]
            pygame.draw.polygon(surf, (120, 190, 110), leaf)
            pygame.draw.polygon(surf, INK, leaf, 2)
        prev = cur


def moon(surf: pygame.Surface, center, radius: float, color=(255, 246, 210), glow: bool = True) -> None:
    if glow:
        layer = _alpha_layer(surf)
        for k in range(5, 0, -1):
            pygame.draw.circle(layer, with_alpha(color, 18), (int(center[0]), int(center[1])), int(radius * (1 + k * 0.25)))
        surf.blit(layer, (0, 0))
    pygame.draw.circle(surf, INK, (int(center[0]), int(center[1])), int(radius) + 3)
    pygame.draw.circle(surf, color, (int(center[0]), int(center[1])), int(radius))
    for dx, dy, r in ((-0.3, -0.2, 0.18), (0.25, 0.15, 0.12), (0.05, 0.4, 0.1)):
        pygame.draw.circle(surf, (235, 222, 180), (int(center[0] + dx * radius), int(center[1] + dy * radius)), int(r * radius))


def sun(surf: pygame.Surface, center, radius: float, t: float) -> None:
    radial_lines(surf, center, t, radius * 1.2, radius * 2.4, (255, 236, 160), count=16, alpha=110)
    pygame.draw.circle(surf, INK, (int(center[0]), int(center[1])), int(radius) + 3)
    pygame.draw.circle(surf, (255, 214, 100), (int(center[0]), int(center[1])), int(radius))


def banner(surf: pygame.Surface, x: float, ground: float, height: float, color, t: float, text: str = "") -> None:
    pygame.draw.line(surf, INK, (int(x), int(ground)), (int(x), int(ground - height)), 8)
    pygame.draw.line(surf, (150, 100, 60), (int(x), int(ground)), (int(x), int(ground - height)), 4)
    top = ground - height
    pts = [(x, top + 6)]
    for k in range(7):
        u = k / 6
        pts.append((x + 90 * u, top + 6 + math.sin(t / 200.0 + u * 4) * 6 * u))
    for k in range(6, -1, -1):
        u = k / 6
        pts.append((x + 90 * u, top + 70 + math.sin(t / 200.0 + u * 4) * 6 * u))
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, INK, pts, 3)
    if text:
        outlined_text(surf, text, 34, (255, 255, 255), (x + 45, top + 40), ow=2)


def vignette(surf: pygame.Surface, strength: int = 120) -> None:
    w, h = surf.get_size()
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    for k in range(12):
        pygame.draw.rect(layer, (0, 0, 0, strength // 12), (k * 10, k * 8, w - k * 20, h - k * 16), 14, border_radius=60)
    surf.blit(layer, (0, 0))


def polyline_points(points: Iterable[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in points]
