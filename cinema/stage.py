"""Letterboxed cutscene stage: backdrops, camera, captions and transitions."""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import pygame

from game_engine import RED
from cinema import fx
from cinema.actors import INK, TEAM
from cinema.detect import StoryEvent
from cinema.fonts import font

SCREEN_W, SCREEN_H = 1440, 900
BAR = 84
STAGE_W, STAGE_H = SCREEN_W, SCREEN_H - BAR * 2
GROUND = 560

CATEGORY = {
    "opening": ("开局", "OPENING"),
    "capture": ("吃子", "CAPTURE"),
    "sacrifice": ("妙手", "SACRIFICE"),
    "check": ("将军", "CHECK"),
    "mate": ("绝杀", "CHECKMATE"),
    "stalemate": ("困毙", "STALEMATE"),
}

_backdrops: dict[str, pygame.Surface] = {}


def _ground(surf: pygame.Surface, top: int, color, edge) -> None:
    pygame.draw.rect(surf, color, (0, top, STAGE_W, STAGE_H - top))
    pygame.draw.line(surf, INK, (0, top), (STAGE_W, top), 4)
    rng = random.Random(top)
    for _ in range(40):
        x, y = rng.random() * STAGE_W, top + 12 + rng.random() * (STAGE_H - top - 20)
        pygame.draw.ellipse(surf, edge, (int(x), int(y), 18 + int(rng.random() * 30), 5))


def _paint_backdrop(name: str) -> pygame.Surface:
    surf = pygame.Surface((STAGE_W, STAGE_H)).convert()
    size = (STAGE_W, STAGE_H)
    if name == "plains":
        surf.blit(fx.gradient(size, (255, 170, 110), (255, 226, 170)), (0, 0))
        fx.sun(surf, (1120, 190), 70, 0)
        fx.mountains(surf, pygame.Rect(0, 250, STAGE_W, 300), (196, 120, 110), seed=1, peaks=5, height=0.8)
        fx.mountains(surf, pygame.Rect(0, 340, STAGE_W, 220), (160, 92, 96), seed=2, peaks=7, height=0.7)
        _ground(surf, GROUND, (214, 170, 110), (190, 146, 90))
    elif name == "grassland":
        surf.blit(fx.gradient(size, (110, 190, 255), (210, 240, 255)), (0, 0))
        fx.mountains(surf, pygame.Rect(0, 300, STAGE_W, 260), (130, 190, 160), seed=3, peaks=4, height=0.7)
        _ground(surf, GROUND, (120, 196, 100), (96, 170, 84))
    elif name == "fortress":
        surf.blit(fx.gradient(size, (120, 110, 140), (220, 190, 170)), (0, 0))
        for k in range(14):
            x = k * 110
            pygame.draw.rect(surf, (120, 96, 90), (x, 330, 104, 230))
            pygame.draw.rect(surf, INK, (x, 330, 104, 230), 3)
            pygame.draw.rect(surf, (120, 96, 90), (x + 12, 296, 36, 40))
            pygame.draw.rect(surf, INK, (x + 12, 296, 36, 40), 3)
            for r in range(4):
                pygame.draw.line(surf, (96, 76, 72), (x, 360 + r * 50), (x + 104, 360 + r * 50), 2)
        _ground(surf, GROUND, (150, 132, 116), (126, 110, 98))
    elif name == "jungle":
        surf.blit(fx.gradient(size, (150, 220, 170), (230, 250, 210)), (0, 0))
        fx.mountains(surf, pygame.Rect(0, 280, STAGE_W, 280), (100, 170, 120), seed=4, peaks=6, height=0.8)
        _ground(surf, GROUND, (126, 170, 90), (106, 146, 76))
    elif name == "courtyard":
        surf.blit(fx.gradient(size, (255, 214, 170), (255, 240, 220)), (0, 0))
        fx.palace(surf, STAGE_W / 2, GROUND, 1.6)
        _ground(surf, GROUND, (200, 180, 160), (176, 156, 138))
        for x in range(0, STAGE_W, 90):
            pygame.draw.line(surf, (176, 156, 138), (x, GROUND), (x - 120, STAGE_H), 2)
    elif name == "throne":
        surf.blit(fx.gradient(size, (120, 30, 50), (200, 70, 70)), (0, 0))
        for x in (140, 420, 1020, 1300):
            pygame.draw.rect(surf, INK, (x - 34, 0, 68, GROUND))
            pygame.draw.rect(surf, (220, 60, 60), (x - 30, 0, 60, GROUND))
            pygame.draw.rect(surf, (250, 198, 92), (x - 30, 60, 60, 16))
            pygame.draw.rect(surf, (250, 198, 92), (x - 30, GROUND - 70, 60, 16))
        pygame.draw.rect(surf, (250, 198, 92), (560, 220, 320, 340), border_radius=30)
        pygame.draw.rect(surf, INK, (560, 220, 320, 340), 4, border_radius=30)
        _ground(surf, GROUND, (150, 40, 50), (120, 30, 40))
        pygame.draw.rect(surf, (200, 50, 60), (420, GROUND, 600, STAGE_H - GROUND))
    elif name == "riverbank":
        surf.blit(fx.gradient(size, (150, 210, 255), (230, 245, 255)), (0, 0))
        fx.mountains(surf, pygame.Rect(0, 300, STAGE_W, 200), (150, 180, 210), seed=5, peaks=6, height=0.7)
        pygame.draw.rect(surf, (90, 160, 220), (0, 430, STAGE_W, 110))
        _ground(surf, GROUND, (210, 190, 140), (186, 166, 120))
        pygame.draw.rect(surf, (190, 170, 120), (0, 530, STAGE_W, 30))
        pygame.draw.line(surf, INK, (0, 530), (STAGE_W, 530), 3)
    elif name == "night":
        surf.blit(fx.gradient(size, (20, 24, 60), (60, 60, 120)), (0, 0))
        rng = random.Random(9)
        for _ in range(90):
            pygame.draw.circle(surf, (255, 255, 230), (int(rng.random() * STAGE_W), int(rng.random() * 420)), 1 + int(rng.random() * 2))
        fx.mountains(surf, pygame.Rect(0, 330, STAGE_W, 230), (40, 44, 90), seed=6, peaks=6, height=0.7)
        _ground(surf, GROUND, (50, 56, 100), (40, 46, 86))
    elif name == "sea":
        surf.blit(fx.gradient(size, (16, 20, 56), (40, 60, 120)), (0, 0))
        rng = random.Random(10)
        for _ in range(70):
            pygame.draw.circle(surf, (255, 255, 230), (int(rng.random() * STAGE_W), int(rng.random() * 300)), 1 + int(rng.random() * 2))
        pygame.draw.rect(surf, (20, 60, 120), (0, 330, STAGE_W, STAGE_H - 330))
        surf.blit(fx.gradient((STAGE_W, STAGE_H - 330), (30, 90, 160), (8, 20, 60)), (0, 330))
    elif name == "sky":
        surf.blit(fx.gradient(size, (120, 180, 255), (255, 230, 240)), (0, 0))
    elif name == "palace_night":
        surf.blit(fx.gradient(size, (40, 20, 70), (120, 60, 110)), (0, 0))
        fx.moon(surf, (1180, 140), 60)
        fx.palace(surf, STAGE_W / 2, GROUND, 1.7, wall=(150, 50, 70), roof=(30, 30, 60))
        _ground(surf, GROUND, (90, 70, 100), (76, 58, 86))
    elif name == "ink":
        surf.blit(fx.gradient(size, (246, 238, 222), (226, 214, 190)), (0, 0))
        fx.mountains(surf, pygame.Rect(0, 260, STAGE_W, 300), (190, 190, 186), seed=7, peaks=5, height=0.9, outline=False)
        fx.mountains(surf, pygame.Rect(0, 360, STAGE_W, 200), (150, 150, 148), seed=8, peaks=7, height=0.8, outline=False)
        _ground(surf, GROUND, (214, 200, 172), (196, 182, 156))
    else:
        surf.blit(fx.gradient(size, (60, 40, 90), (140, 90, 150)), (0, 0))
        _ground(surf, GROUND, (90, 70, 110), (76, 58, 96))
    return surf


def backdrop(name: str) -> pygame.Surface:
    if name not in _backdrops:
        _backdrops[name] = _paint_backdrop(name)
    return _backdrops[name]


def brush_stroke(surf: pygame.Surface, rect: pygame.Rect, color, u: float, seed: int = 11) -> None:
    """Ink-brush swash revealed left to right."""
    if u <= 0:
        return
    rng = random.Random(seed)
    right = rect.x + rect.w * fx.ease_out(min(1.0, u))
    top, bottom = [], []
    steps = 24
    for k in range(steps + 1):
        x = rect.x + (right - rect.x) * k / steps
        taper = math.sin(min(1.0, k / steps * 1.15) * math.pi * 0.5 + 0.2)
        top.append((x, rect.centery - rect.h * 0.5 * taper + rng.uniform(-6, 6)))
        bottom.append((x, rect.centery + rect.h * 0.5 * taper + rng.uniform(-6, 6)))
    poly = top + bottom[::-1]
    pygame.draw.polygon(surf, color, poly)
    for k in range(5):
        y = rect.y + rect.h * (0.2 + 0.15 * k)
        pygame.draw.line(surf, tuple(max(0, c - 40) for c in color), (rect.x + 30, int(y)), (int(right - 20), int(y + rng.uniform(-4, 4))), 2)


def title_card(surf: pygame.Surface, title: str, subtitle: str, center, u: float, color=(255, 236, 160), stroke=(200, 40, 60), size: int = 92) -> None:
    """Big brushed title that slams in (u 0..1 over its lifetime)."""
    if u <= 0 or u >= 1:
        return
    slam = fx.seg(u, 0.0, 0.12)
    fade = 1 - fx.seg(u, 0.85, 1.0)
    rect = pygame.Rect(0, 0, int(size * (len(title.replace(" ", "")) * 1.05 + 1.6)), int(size * 1.25))
    rect.center = (int(center[0]), int(center[1]))
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    brush_stroke(layer, rect, stroke, fx.seg(u, 0.04, 0.2), seed=len(title))
    scale = 1 + 1.6 * (1 - fx.ease_out(slam))
    fx.outlined_text(layer, title, size, color, center, ow=6, scale=scale)
    if subtitle:
        fx.outlined_text(layer, subtitle, size * 0.3, (255, 255, 255), (center[0], center[1] + size * 0.86), ow=3, alpha=int(255 * fx.seg(u, 0.15, 0.3)))
    layer.set_alpha(int(255 * fade))
    surf.blit(layer, (0, 0))


class Cutscene:
    """Base class: subclasses implement :meth:`paint` on a stage-sized surface."""

    duration = 3000
    backdrop_name = "ink"

    def __init__(self, event: StoryEvent):
        self.event = event
        self.side = event.actor.side
        self.color = TEAM[self.side]["ring"]
        self.enemy = "black" if self.side == RED else "red"
        self._stage = pygame.Surface((STAGE_W, STAGE_H)).convert()
        self.shakes: list[tuple[float, float, float]] = []

    # --- hooks -----------------------------------------------------------
    def paint(self, stage: pygame.Surface, t: float) -> None:
        raise NotImplementedError

    def camera(self, t: float) -> tuple[float, float, float]:
        """(zoom, dx, dy)."""
        dx = dy = 0.0
        for at, length, power in self.shakes:
            if at <= t < at + length:
                k = 1 - (t - at) / length
                dx += math.sin(t * 0.21) * power * k
                dy += math.cos(t * 0.17) * power * 0.6 * k
        return 1.0, dx, dy

    def shake(self, at: float, length: float = 300, power: float = 14) -> None:
        self.shakes.append((at, length, power))

    # --- rendering -----------------------------------------------------
    def draw(self, canvas: pygame.Surface, t: float) -> None:
        stage = self._stage
        stage.blit(backdrop(self.backdrop_name), (0, 0))
        self.paint(stage, t)
        zoom, dx, dy = self.camera(t)
        enter = fx.ease_out(fx.seg(t, 0, 260))
        leave = fx.ease_in(fx.seg(t, self.duration - 260, self.duration))
        shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        shade.fill((6, 4, 14, int(200 * enter * (1 - leave))))
        canvas.blit(shade, (0, 0))
        open_frac = enter * (1 - leave)
        if open_frac > 0.001:
            view = stage
            if abs(zoom - 1.0) > 1e-3:
                zw, zh = int(STAGE_W * zoom), int(STAGE_H * zoom)
                view = pygame.transform.smoothscale(stage, (zw, zh))
                cut = pygame.Rect((zw - STAGE_W) // 2 - int(dx), (zh - STAGE_H) // 2 - int(dy), STAGE_W, STAGE_H)
                cut.clamp_ip(view.get_rect())
                view = view.subsurface(cut)
                dx = dy = 0
            h = int(STAGE_H * open_frac)
            src = pygame.Rect(0, (STAGE_H - h) // 2, STAGE_W, h)
            canvas.blit(view, (int(dx), BAR + (STAGE_H - h) // 2 + int(dy)), src)
        bar_h = int(BAR * enter * (1 - leave)) + int(STAGE_H / 2 * (1 - open_frac)) if open_frac < 1 else BAR
        pygame.draw.rect(canvas, (8, 6, 14), (0, 0, SCREEN_W, max(0, min(BAR, bar_h))))
        pygame.draw.rect(canvas, (8, 6, 14), (0, SCREEN_H - max(0, min(BAR, bar_h)), SCREEN_W, max(0, min(BAR, bar_h))))
        if enter > 0.9 and leave < 0.1:
            self.draw_captions(canvas, t)

    def draw_captions(self, canvas: pygame.Surface, t: float) -> None:
        cat, en = CATEGORY.get(self.event.kind, ("演出", "SCENE"))
        alpha = int(255 * fx.seg(t, 260, 520) * (1 - fx.seg(t, self.duration - 400, self.duration - 260)))
        layer = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        accent = self.color
        pygame.draw.line(layer, (*accent, 200), (60, BAR // 2), (520, BAR // 2), 2)
        pygame.draw.line(layer, (*accent, 200), (920, BAR // 2), (1380, BAR // 2), 2)
        top = font(26, True).render(f"{cat}  ·  {en}", True, (255, 240, 220))
        layer.blit(top, top.get_rect(center=(SCREEN_W // 2, BAR // 2)))
        seal = pygame.Rect(54, SCREEN_H - BAR + 14, 56, 56)
        pygame.draw.rect(layer, accent, seal, border_radius=10)
        pygame.draw.rect(layer, (255, 240, 220), seal, 2, border_radius=10)
        mark = font(30, True).render("红" if self.side == RED else "黑", True, (255, 255, 255))
        layer.blit(mark, mark.get_rect(center=seal.center))
        title = font(40, True).render(self.event.title, True, (255, 226, 150))
        layer.blit(title, (128, SCREEN_H - BAR + 12))
        sub = font(19, False).render(self.event.subtitle, True, (230, 220, 240))
        layer.blit(sub, (132 + title.get_width() + 24, SCREEN_H - BAR + 32))
        hint = font(16, False).render("空格 / 点击 跳过", True, (160, 150, 180))
        layer.blit(hint, hint.get_rect(right=SCREEN_W - 50, centery=SCREEN_H - BAR // 2))
        progress = t / self.duration
        pygame.draw.rect(layer, (*accent, 220), (0, SCREEN_H - 4, int(SCREEN_W * progress), 4))
        layer.set_alpha(alpha)
        canvas.blit(layer, (0, 0))


class LambdaCutscene(Cutscene):
    """Cutscene whose painter is a plain function ``painter(scene, stage, t)``."""

    def __init__(self, event: StoryEvent, painter: Callable, duration: int, backdrop_name: str, setup: Optional[Callable] = None):
        super().__init__(event)
        self.painter = painter
        self.duration = duration
        self.backdrop_name = backdrop_name
        if setup:
            setup(self)

    def paint(self, stage: pygame.Surface, t: float) -> None:
        self.painter(self, stage, t)
