"""Capture duels: attacker style × victim reaction, one close-up per pair (车吃马, 炮打象 …)."""

from __future__ import annotations

import math

import pygame

from game_engine import ADVISOR, CANNON, ELEPHANT, HORSE, KING, PAWN, ROOK
from cinema import fx
from cinema.actors import GOLD, INK, STRAW, WOOD, actor, ink_circle
from cinema.stage import GROUND, Cutscene, title_card

R_ACT = 92
AX, VX = 330, 1090

BACKDROP = {ROOK: "plains", HORSE: "grassland", CANNON: "fortress", ELEPHANT: "jungle",
            ADVISOR: "courtyard", KING: "throne", PAWN: "riverbank"}
SOUND = {ROOK: "轰隆!", HORSE: "咚咚!", CANNON: "砰!!", ELEPHANT: "咚——!", ADVISOR: "唰!", KING: "斩!", PAWN: "嘿哈!"}
SOUND_COLOR = {ROOK: (255, 150, 80), HORSE: (255, 220, 100), CANNON: (255, 110, 70), ELEPHANT: (180, 230, 120),
               ADVISOR: (150, 230, 255), KING: (255, 230, 120), PAWN: (255, 170, 200)}
IMPACT = {ROOK: 1450, HORSE: 1640, CANNON: 1660, ELEPHANT: 1720, ADVISOR: 1640, KING: 1590, PAWN: 1570}


def flying_wheel(surf: pygame.Surface, pos, r: float, angle: float) -> None:
    ink_circle(surf, WOOD, pos, r, 4)
    for k in range(4):
        a = angle + k * math.pi / 4
        dx, dy = math.cos(a) * r * 0.85, math.sin(a) * r * 0.85
        pygame.draw.line(surf, INK, (pos[0] - dx, pos[1] - dy), (pos[0] + dx, pos[1] + dy), 4)
    pygame.draw.circle(surf, GOLD, (int(pos[0]), int(pos[1])), int(r * 0.25))


def flying_prop(surf: pygame.Surface, kind: str, pos, size: float, angle: float, side_color) -> None:
    layer = pygame.Surface((int(size * 3), int(size * 3)), pygame.SRCALPHA)
    c = size * 1.5
    if kind == KING:
        pts = [(c - size * 0.6, c + size * 0.3), (c - size * 0.65, c - size * 0.4), (c - size * 0.3, c - size * 0.1),
               (c, c - size * 0.6), (c + size * 0.3, c - size * 0.1), (c + size * 0.65, c - size * 0.4), (c + size * 0.6, c + size * 0.3)]
        color = GOLD
    elif kind == ADVISOR:
        pts = [(c - size * 0.45, c + size * 0.3), (c - size * 0.4, c - size * 0.3), (c + size * 0.4, c - size * 0.3), (c + size * 0.45, c + size * 0.3)]
        pygame.draw.ellipse(layer, INK, (c - size * 1.05, c - size * 0.2, size * 2.1, size * 0.3))
        color = (40, 36, 54)
    elif kind == PAWN:
        pts = [(c - size, c + size * 0.3), (c, c - size * 0.5), (c + size, c + size * 0.3)]
        color = STRAW
    elif kind == ELEPHANT:
        pts = [(c - size * 0.5, c), (c + size * 0.5, c - size * 0.2), (c + size * 0.2, c + size * 0.15)]
        color = (255, 252, 240)
    else:
        pts = [(c - size * 0.6, c - size * 0.2), (c + size * 0.8, c - size * 0.25), (c + size * 0.8, c + size * 0.25), (c - size * 0.6, c + size * 0.2)]
        color = (70, 74, 92)
    pygame.draw.polygon(layer, color, pts)
    pygame.draw.polygon(layer, INK, pts, 3)
    if kind == PAWN:
        pygame.draw.line(layer, side_color, (c - size * 0.55, c), (c + size * 0.55, c), 5)
    layer = pygame.transform.rotozoom(layer, math.degrees(angle), 1.0)
    surf.blit(layer, layer.get_rect(center=(int(pos[0]), int(pos[1]))))


class CaptureCutscene(Cutscene):
    duration = 3100

    def __init__(self, event):
        super().__init__(event)
        self.a = event.actor
        self.v = event.victim
        self.backdrop_name = BACKDROP[self.a.kind]
        self.ti = IMPACT[self.a.kind]
        self.duel = self.a.kind == self.v.kind
        self.shake(self.ti, 380, 18)
        if self.a.kind == ELEPHANT:
            self.shake(1250, 160, 8)
            self.shake(1450, 160, 8)

    def camera(self, t: float):
        zoom, dx, dy = super().camera(t)
        punch = fx.hop(fx.seg(t, self.ti - 40, self.ti + 260)) * 0.07
        return zoom + punch, dx, dy

    # --- attacker choreography ---------------------------------------------
    def attacker_state(self, t: float) -> dict:
        k = self.a.kind
        ti = self.ti
        st = {"x": AX, "lift": 0.0, "expr": "fierce", "walk": 0.0, "swing": 0.0, "squash": 0.0, "tilt": 0.0}
        enter = fx.ease_out(fx.seg(t, 0, 600))
        st["x"] = fx.lerp(-200, AX, enter)
        if t < 600:
            st["walk"] = t / 70.0
            st["expr"] = "idle"
        if k == ROOK:
            u = fx.seg(t, 1080, ti)
            if t < 1080:
                st["squash"] = 0.18 * fx.seg(t, 800, 1080)
                st["tilt"] = -0.1 * fx.seg(t, 800, 1080)
            elif t < ti:
                st["x"] = fx.lerp(AX, VX - 190, fx.ease_in(u))
                st["walk"] = t / 25.0
                st["tilt"] = 0.15
                st["squash"] = -0.1
                st["swing"] = 0.5 * fx.seg(t, ti - 150, ti)
            else:
                back = fx.ease_out(fx.seg(t, ti, ti + 300))
                st["x"] = VX - 190 - 70 * back
                st["swing"] = 0.5 + 0.5 * fx.seg(t, ti, ti + 200)
                st["walk"] = (ti + 300 * back) / 25.0
        elif k == HORSE:
            u = fx.seg(t, 1000, 1450)
            if 1000 <= t < 1450:
                st["x"] = fx.lerp(AX, VX - 230, u)
                st["lift"] = fx.hop((u * 3) % 1.0) * 70
                st["walk"] = t / 40.0
                st["tilt"] = 0.1
            elif t >= 1450:
                st["x"] = VX - 230
                rear = fx.seg(t, 1450, 1560)
                kick = fx.seg(t, 1560, 1720)
                st["tilt"] = -0.45 * fx.hop(rear * 0.5 + kick * 0.5) if kick == 0 else -0.45 * (1 - kick) + 0.2 * fx.hop(kick)
                st["swing"] = kick
        elif k == CANNON:
            if 950 <= t < 1300:
                st["squash"] = 0.15 * fx.seg(t, 950, 1300)
            elif t >= 1300:
                rec = fx.seg(t, 1300, 1560)
                st["swing"] = rec
                st["x"] = AX - 40 * fx.hop(rec)
                st["squash"] = -0.15 * fx.hop(rec)
        elif k == ELEPHANT:
            if 1000 <= t < 1500:
                u = fx.seg(t, 1000, 1500)
                steps = u * 2
                st["x"] = fx.lerp(AX, VX - 240, fx.ease_in_out(u))
                st["walk"] = steps * math.pi
                st["lift"] = fx.hop(steps % 1.0) * 26
                st["squash"] = 0.12 * (1 - fx.hop(steps % 1.0))
            elif t >= 1500:
                st["x"] = VX - 240
                rise = fx.seg(t, 1500, 1640)
                stomp = fx.seg(t, 1640, ti)
                st["lift"] = 70 * fx.ease_out(rise) * (1 - fx.ease_in(stomp))
                st["squash"] = -0.25 * rise * (1 - stomp) + 0.35 * fx.hop(fx.seg(t, ti, ti + 200))
                st["swing"] = rise
        elif k == ADVISOR:
            if 1000 <= t < 1350:
                u = fx.ease_in_out(fx.seg(t, 1000, 1350))
                st["x"] = fx.lerp(AX, VX - 330, u)
                st["lift"] = 24 * fx.hop(u)
            elif t >= 1350:
                st["x"] = VX - 330
                st["swing"] = fx.ease_out(fx.seg(t, 1350, 1520))
                st["tilt"] = 0.15 * fx.hop(fx.seg(t, 1350, 1600))
            st["expr"] = "smug" if t > 600 else "idle"
        elif k == KING:
            if 1000 <= t < 1400:
                u = fx.seg(t, 1000, 1400)
                st["x"] = fx.lerp(AX, VX - 250, u)
                st["walk"] = t / 90.0
            elif t >= 1400:
                st["x"] = VX - 250
                st["swing"] = fx.ease_out(fx.seg(t, 1490, ti))
                st["tilt"] = 0.12 * fx.hop(fx.seg(t, 1480, ti + 200))
        elif k == PAWN:
            if 1000 <= t < 1450:
                u = fx.seg(t, 1000, 1450)
                st["x"] = fx.lerp(AX, VX - 290, u)
                st["walk"] = t / 45.0
                st["lift"] = fx.hop((u * 4) % 1.0) * 18
            elif t >= 1450:
                lunge = fx.seg(t, 1450, ti)
                st["x"] = VX - 290 + 70 * fx.ease_out(lunge) - 30 * fx.seg(t, ti + 100, ti + 400)
                st["swing"] = fx.ease_out(fx.seg(t, 1430, 1520))
                st["tilt"] = 0.18 * lunge * (1 - fx.seg(t, ti + 100, ti + 400))
        if t > ti + 450:
            cheer = fx.seg(t, ti + 450, ti + 700)
            st["expr"] = "smug" if k in (ADVISOR, KING, CANNON) else "cheer"
            if k in (PAWN, HORSE, ROOK):
                st["lift"] = fx.hop(((t - ti - 450) / 380.0) % 1.0) * 30 * cheer
            st["swing"] = st["swing"] * (1 - cheer)
            st["tilt"] = st["tilt"] * (1 - cheer)
        return st

    def attack_effects(self, stage: pygame.Surface, t: float, st: dict) -> None:
        k = self.a.kind
        ti = self.ti
        gy = GROUND
        if k == ROOK and 1080 <= t < ti + 100:
            fx.speed_lines(stage, pygame.Rect(0, gy - 260, int(st["x"]), 240), t, 1, count=14, alpha=200)
            for n in range(4):
                fx.dust(stage, (st["x"] - 80 - n * 70, gy - 10), fx.seg(t, 1100 + n * 70, 1500 + n * 70), 50, seed=n)
        if k == ROOK and t >= ti:
            fx.dust(stage, (VX - 180, gy - 10), fx.seg(t, ti, ti + 600), 80, seed=21)
        if k == HORSE and 1000 <= t < 1500:
            for n in range(3):
                land = 1000 + (n + 1) * 150
                fx.dust(stage, (fx.lerp(AX, VX - 230, (n + 1) / 3), gy - 6), fx.seg(t, land, land + 420), 46, seed=n + 4)
        if k == HORSE and 1450 <= t < 1600:
            fx.speech_bubble(stage, "嘶——！", (VX - 200, gy - 250), fx.seg(t, 1440, 1720), size=30, tail="down-left", shout=True)
        if k == CANNON:
            screen_x = (AX + VX) / 2
            ducking = fx.hop(fx.seg(t, 1300, 1700))
            actor(stage, PAWN, self.side, (screen_x, gy), R_ACT * 0.55, facing=1, expr="scared" if ducking > 0.05 else "idle",
                  t=t, squash=0.35 * ducking)
            fx.outlined_text(stage, "炮架", 22, (255, 255, 255), (screen_x, gy + 32), ow=2)
            if t < 1300:
                aim = fx.seg(t, 900, 1300)
                if aim > 0:
                    pygame.draw.circle(stage, (255, 80, 80), (VX, gy - 100), int(70 - 30 * aim), 4)
                    pygame.draw.line(stage, (255, 80, 80), (VX - 90, gy - 100), (VX + 90, gy - 100), 3)
                    pygame.draw.line(stage, (255, 80, 80), (VX, gy - 190), (VX, gy - 10), 3)
            muzzle = (st["x"] + R_ACT * 1.1, gy - R_ACT * 2.5)
            fx.explosion(stage, muzzle, fx.seg(t, 1300, 1600), 50, seed=31)
            fx.arrow_projectile(stage, muzzle, (VX, gy - 110), fx.seg(t, 1310, ti), size=20, arc=230)
            if t >= ti:
                fx.explosion(stage, (VX, gy - 110), fx.seg(t, ti, ti + 900), 170, seed=32)
        if k == ELEPHANT:
            for n, land in enumerate((1250, 1500)):
                fx.dust(stage, (fx.lerp(AX, VX - 240, 0.5 * (n + 1)), gy - 6), fx.seg(t, land, land + 500), 60, seed=n + 7)
            fx.shockwave(stage, (VX - 200, gy), fx.seg(t, ti, ti + 600), 420, (255, 255, 230), 10, flat=0.25)
            fx.shockwave(stage, (VX - 200, gy), fx.seg(t, ti + 80, ti + 700), 300, (255, 255, 230), 6, flat=0.25)
        if k == ADVISOR:
            fx.sparkles(stage, (st["x"], gy - 120), t, 140, count=10)
            for n in range(3):
                u = fx.seg(t, 1400 + n * 60, ti + n * 60)
                if 0 < u < 1:
                    x = fx.lerp(VX - 240, VX, u)
                    y = gy - 150 + (n - 1) * 60
                    pygame.draw.arc(stage, INK, (int(x - 40), int(y - 50), 80, 100), -1.2, 1.2, 9)
                    pygame.draw.arc(stage, (200, 245, 255), (int(x - 40), int(y - 50), 80, 100), -1.2, 1.2, 5)
        if k == KING:
            fx.slash_arc(stage, (VX - 40, gy - 130), 150, fx.seg(t, 1500, 1820), -2.2, 2.6, (255, 236, 150), 34)
            if t > 1000:
                fx.sparkles(stage, (st["x"], gy - 220), t, 90, (255, 220, 120), count=5)
        if k == PAWN and 1430 <= t < ti + 200:
            fx.speed_lines(stage, pygame.Rect(int(st["x"]) - 200, gy - 200, 260, 120), t, 1, count=8, alpha=180, seed=40)

    # --- victim reactions ----------------------------------------------------
    def victim_state(self, t: float) -> dict:
        k = self.v.kind
        ti = self.ti
        st = {"x": VX, "y": GROUND, "expr": "idle", "tilt": 0.0, "squash": 0.0, "props": set(), "alpha": 255,
              "flash": 0.0, "lift": 0.0, "walk": 0.0, "swing": 0.0, "pivot": "feet", "R": R_ACT}
        if t > 500:
            st["expr"] = "surprised"
        if t > 900:
            st["expr"] = "scared"
            st["x"] = VX + math.sin(t * 0.08) * 3
        if self.duel and 1000 <= t < ti:
            u = fx.seg(t, 1000, ti - 80)
            st["expr"] = "fierce"
            st["x"] = VX - 60 * fx.ease_out(u)
            st["swing"] = fx.seg(t, ti - 250, ti - 80)
        if t < ti:
            return st
        r = t - ti
        st["flash"] = 1 - fx.seg(t, ti, ti + 160)
        st["expr"] = "hurt"
        if k == ROOK:
            u = fx.seg(r, 0, 900)
            st["props"] = {"no_wheels", "no_weapon"}
            st["x"] = VX + 520 * fx.ease_out(u)
            st["lift"] = fx.hop(u) * 260
            st["tilt"] = -u * math.tau * 1.5
            st["pivot"] = "center"
            st["expr"] = "ko"
        elif k == HORSE:
            fall = fx.ease_bounce(fx.seg(r, 250, 700))
            st["tilt"] = -1.45 * fall
            st["x"] = VX + 60 * fall
            st["expr"] = "ko" if r > 250 else "hurt"
        elif k == CANNON:
            st["props"] = {"soot"} if r > 120 else set()
            fall = fx.ease_bounce(fx.seg(r, 500, 1000))
            st["tilt"] = 1.4 * fall
            st["expr"] = "ko" if r > 500 else "hurt"
        elif k == ELEPHANT:
            flat = fx.ease_out(fx.seg(r, 0, 180))
            st["squash"] = 0.75 * flat - 0.1 * fx.hop(fx.seg(r, 180, 500))
            st["props"] = {"no_tusks"}
            st["expr"] = "ko"
        elif k == ADVISOR:
            st["props"] = {"no_hat"}
            spin = fx.seg(r, 0, 600)
            st["tilt"] = math.sin(spin * math.pi * 3) * 0.5 * (1 - spin) + 1.45 * fx.ease_bounce(fx.seg(r, 500, 900))
            st["expr"] = "ko" if r > 500 else "hurt"
        elif k == PAWN:
            u = fx.seg(r, 0, 1000)
            st["props"] = {"no_hat"}
            st["x"] = VX + 450 * fx.ease_out(u)
            st["lift"] = 520 * fx.ease_out(u)
            st["R"] = R_ACT * (1 - 0.8 * u)
            st["tilt"] = -u * math.tau * 2
            st["pivot"] = "center"
            st["alpha"] = int(255 * (1 - fx.seg(u, 0.8, 1.0)))
        elif k == KING:
            st["props"] = {"no_hat"}
            fall = fx.ease_bounce(fx.seg(r, 200, 700))
            st["tilt"] = -1.4 * fall
            st["expr"] = "ko"
        return st

    def victim_effects(self, stage: pygame.Surface, t: float, st: dict) -> None:
        k = self.v.kind
        ti = self.ti
        if t < ti:
            return
        r = t - ti
        if k == ROOK:
            for n, (dx, spin) in enumerate(((-1, 1), (1, -1))):
                u = fx.seg(r, 0, 1100)
                pos = (VX + dx * 300 * u + 80 * u, GROUND - 40 - fx.hop(u) * (200 + n * 90) + 40 * u)
                flying_wheel(stage, pos, 36, spin * u * 14)
        elif k == HORSE:
            if r > 350:
                fx.dizzy_stars(stage, (st["x"] + 70, GROUND - 120), t, 60)
        elif k == CANNON:
            fx.explosion(stage, (VX + 30, GROUND - 230), fx.seg(r, 80, 650), 70, seed=55)
            fx.speech_bubble(stage, "咳咳……", (VX + 40, GROUND - 250), fx.seg(r, 600, 1300), size=26, tail="down-right")
        elif k == ELEPHANT:
            for n, dx in enumerate((-1, 1)):
                u = fx.seg(r, 0, 800)
                pos = (VX + dx * 260 * u, GROUND - 80 - fx.hop(u) * 220)
                flying_prop(stage, ELEPHANT, pos, 36, u * 12 * dx, self.color)
        elif k == ADVISOR:
            u = fx.seg(r, 0, 1100)
            flying_prop(stage, ADVISOR, (VX + 160 * u, GROUND - 230 - 300 * fx.ease_out(u) + 200 * u * u), 50, u * 15, self.color)
        elif k == PAWN:
            u = fx.seg(r, 0, 900)
            flying_prop(stage, PAWN, (VX - 120 * u, GROUND - 240 - fx.hop(u) * 200 + 220 * u), 70, -u * 10, self.color)
            tw = fx.seg(r, 950, 1350)
            if 0 < tw < 1:
                star = (VX + 450, GROUND - 520 - 92 + 20)
                s = 30 * fx.hop(tw)
                pygame.draw.polygon(stage, (255, 255, 200), fx.star_points(star[0], star[1], s, s * 0.3, 4, 0.0))
        elif k == KING:
            u = fx.seg(r, 0, 700)
            flying_prop(stage, KING, (VX + 120 * u, GROUND - 260 + fx.ease_bounce(u) * 230), 46, u * 4, self.color)
            if r > 500:
                fx.dizzy_stars(stage, (VX - 80, GROUND - 120), t, 60)
        if k in (HORSE, CANNON, ELEPHANT, ADVISOR, KING):
            fx.ghost(stage, (st["x"], GROUND - 160), fx.seg(r, 700, 1450), 46)

    def paint(self, stage: pygame.Surface, t: float) -> None:
        ti = self.ti
        if self.backdrop_name in ("grassland", "riverbank", "jungle"):
            for n in range(3):
                fx.cloud(stage, (n * 520 + t * 0.03) % 1700 - 150, 110 + n * 40, 50 + n * 10, alpha=220)
        if self.backdrop_name == "jungle":
            for x in (60, 180, 1300, 1400):
                fx.bamboo(stage, x, GROUND, 420, t)
        if self.backdrop_name == "grassland":
            fx.grass(stage, pygame.Rect(0, GROUND + 20, 1440, 140), t)
        if self.backdrop_name == "riverbank":
            fx.waves(stage, pygame.Rect(0, 440, 1440, 88), t, (90, 160, 220), (200, 235, 255), rows=3)
            fx.outlined_text(stage, "楚 河", 40, (255, 255, 255), (380, 480), ow=3)
            fx.outlined_text(stage, "汉 界", 40, (255, 255, 255), (1060, 480), ow=3)
        if self.backdrop_name == "courtyard":
            for x in (220, 1220):
                pygame.draw.line(stage, INK, (x, 120), (x, 200), 3)
                pygame.draw.ellipse(stage, INK, (x - 34, 196, 68, 80))
                pygame.draw.ellipse(stage, (255, 90, 80), (x - 30, 200, 60, 72))
                pygame.draw.line(stage, (255, 220, 120), (x, 272), (x, 296), 3)

        vs = self.victim_state(t)
        st = self.attacker_state(t)
        if self.duel and ti - 120 <= t < ti + 200:
            mid = ((st["x"] + vs["x"]) / 2, GROUND - 120)
            fx.radial_lines(stage, mid, t, 40, 260, (255, 250, 210), count=20, alpha=200)
            fx.onomatopoeia(stage, "铛!", (mid[0], mid[1] - 140), 70, (255, 255, 255), fx.seg(t, ti - 120, ti + 500))
        actor(stage, self.v.kind, self.v.side, (vs["x"], vs["y"]), vs["R"], facing=-1, expr=vs["expr"], t=t,
              tilt=vs["tilt"], squash=vs["squash"], props=vs["props"], alpha=vs["alpha"], flash=vs["flash"],
              lift=vs["lift"], pivot=vs["pivot"], walk=vs["walk"], swing=vs["swing"])
        self.victim_effects(stage, t, vs)
        self.attack_effects(stage, t, st)
        actor(stage, self.a.kind, self.a.side, (st["x"], GROUND), R_ACT, facing=1, expr=st["expr"], t=t,
              walk=st["walk"], swing=st["swing"], squash=st["squash"], tilt=st["tilt"], lift=st["lift"])

        fx.impact_star(stage, (VX - 60, GROUND - 130), fx.seg(t, ti, ti + 420), 150, seed=hash(self.event.key) % 97)
        fx.onomatopoeia(stage, SOUND[self.a.kind], (VX - 150, GROUND - 300), 92, SOUND_COLOR[self.a.kind], fx.seg(t, ti, ti + 700))
        fx.speech_bubble(stage, self.event.attacker_line, (st["x"] + 40, GROUND - 2.6 * R_ACT), fx.seg(t, 350, 1350),
                         size=28, tail="down-left")
        fx.speech_bubble(stage, self.event.victim_line, (VX - 30, GROUND - 2.6 * R_ACT), fx.seg(t, 820, 1500),
                         size=26, tail="down-right", fill=(255, 246, 236))
        if t > ti + 450:
            fx.sparkles(stage, (st["x"], GROUND - 150), t, 150, count=10, alpha=255 * fx.seg(t, ti + 450, ti + 700))
        title_u = fx.seg(t, ti + 520, self.duration - 150)
        if title_u > 0:
            title_card(stage, self.event.title, self.event.subtitle, (720, 120), title_u, size=78)
