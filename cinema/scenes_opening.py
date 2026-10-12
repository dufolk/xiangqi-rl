"""One illustrated skit per opening (当头炮、过宫炮、仙人指路、飞相局 …)."""

from __future__ import annotations

import math

import pygame

from game_engine import ADVISOR, CANNON, ELEPHANT, HORSE, KING, PAWN, ROOK
from cinema import fx
from cinema.actors import GOLD, INK, WOOD, ink_circle
from cinema.kit import (
    boom, crosshair, drifting_clouds, fishing_rod, other, put, question_marks, say, title, walk, wings,
)
from cinema.stage import GROUND, LambdaCutscene

DURATION = 3600


def _center_cannon(sc, st, t):
    me, foe = sc.side, other(sc.side)
    x, ph = walk(t, 0, 800, -150, 400)
    plant = fx.seg(t, 800, 1000)
    pawn_scared = t > 1200
    fx.radial_lines(st, (1040, GROUND - 120), t, 140, 900, (255, 255, 255), count=22, alpha=60 * fx.seg(t, 1000, 1400))
    put(st, PAWN, foe, 1040, t, facing=-1, expr="scared" if pawn_scared else "idle")
    put(st, CANNON, me, x, t, walk=ph, expr="fierce", squash=0.2 * fx.hop(plant), swing=0.15 * fx.seg(t, 2400, 2600))
    if t > 1000:
        layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
        pygame.draw.line(layer, (255, 60, 60, 160), (x + 120, GROUND - 235), (1040, GROUND - 110), 4)
        st.blit(layer, (0, 0))
        crosshair(st, (1040, GROUND - 110), fx.seg(t, 1000, 1700))
    say(st, "中路，归我了！", (x + 30, GROUND - 260), t, 900, 2200, size=30)
    say(st, "瞄、瞄准我了？！", (1010, GROUND - 240), t, 1500, 2700, size=26, tail="down-right")
    boom(st, "锁定!", (1040, GROUND - 330), t, 1700, 800, 80, (255, 90, 90))
    title(st, sc, t, 2100)


def _cross_palace(sc, st, t):
    me = sc.side
    put(st, ADVISOR, me, 520, t, R=62, expr="cheer" if 900 < t < 1700 else "idle", tilt=0.25 * fx.hop(fx.seg(t, 900, 1500)))
    x, ph = walk(t, 0, 2600, 120, 1240, speed=60, ease=lambda u: u)
    inside = 640 < x < 800
    if not inside:
        put(st, CANNON, me, x, t, walk=ph, expr="cheer" if x > 800 else "fierce")
    else:
        fx.outlined_text(st, "…", 60, (255, 255, 255), (720, GROUND - 160), ow=3)
    say(st, "借道九宫～", (x + 20, GROUND - 260), t, 300, 1300, size=28)
    say(st, "请——", (520, GROUND - 170), t, 900, 1600, size=24)
    if x > 820:
        boom(st, "到!", (x + 90, GROUND - 300), t, 2400, 700, 70)
    fx.sparkles(st, (x, GROUND - 120), t, 120, count=6)
    title(st, sc, t, 2000)


def _advisor_cannon(sc, st, t):
    me = sc.side
    x, ph = walk(t, 0, 900, -120, 560)
    nod = fx.hop(fx.seg(t, 1100, 1450)) * 0.2
    put(st, ADVISOR, me, 860, t, facing=-1, expr="smug", tilt=nod, swing=fx.hop(fx.seg(t, 1500, 1900)))
    put(st, CANNON, me, x, t, walk=ph, expr="cheer" if t > 1800 else "idle", tilt=nod, squash=0.25 * fx.seg(t, 2100, 2300))
    say(st, "有劳炮兄守角。", (840, GROUND - 230), t, 900, 1900, size=26, tail="down-right")
    say(st, "这角，我包了！", (560, GROUND - 260), t, 1700, 2800, size=28)
    if t > 2100:
        u = fx.ease_back(fx.seg(t, 2100, 2400))
        shield = [(710, GROUND - 330 - 80 * u), (790, GROUND - 300 - 80 * u), (780, GROUND - 200 - 80 * u), (710, GROUND - 150 - 80 * u),
                  (640, GROUND - 200 - 80 * u), (630, GROUND - 300 - 80 * u)]
        pygame.draw.polygon(st, GOLD, shield)
        pygame.draw.polygon(st, INK, shield, 5)
        fx.outlined_text(st, "稳", 54, (255, 255, 255), (710, GROUND - 250 - 80 * u), ow=3)
    title(st, sc, t, 2300)


def _pawn_base(sc, st, t):
    me = sc.side
    put(st, PAWN, me, 760, t, expr="smug" if t > 1300 else "idle", facing=1)
    x, ph = walk(t, 200, 1300, -100, 680, speed=40)
    hidden = t > 1300
    if not hidden:
        put(st, CANNON, me, x, t, walk=ph, expr="fierce", squash=0.3)
    else:
        peek = fx.hop(fx.seg(t, 1800, 2700))
        layer = pygame.Surface((300, 200), pygame.SRCALPHA)
        pygame.draw.rect(layer, INK, (30, 60, 200, 54), border_radius=24)
        pygame.draw.rect(layer, (70, 74, 92), (36, 66, 188, 42), border_radius=20)
        pygame.draw.circle(layer, INK, (226, 87), 30)
        pygame.draw.circle(layer, (20, 20, 30), (226, 87), 16)
        layer = pygame.transform.rotozoom(layer, 8 + 10 * math.sin(t / 240.0), 1.0)
        st.blit(layer, (600, GROUND - 330 - 70 * peek))
    say(st, "嘘……躲好别出声。", (x + 10, GROUND - 230), t, 500, 1300, size=24)
    say(st, "这里没有炮～", (790, GROUND - 260), t, 1600, 2900, size=28)
    title(st, sc, t, 2200)


def _retreat_cannon(sc, st, t):
    me = sc.side
    sink = fx.ease_in_out(fx.seg(t, 900, 1700))
    for k in range(9):
        bx = 520 + k * 52
        pygame.draw.ellipse(st, INK, (bx - 4, GROUND - 46, 60, 48))
        pygame.draw.ellipse(st, (200, 180, 130), (bx, GROUND - 42, 52, 40))
    put(st, CANNON, me, 720, t, y=GROUND + 120 * sink, expr="smug" if t > 1700 else "idle", walk=t / 80 if t < 900 else 0)
    for k in range(9):
        bx = 520 + k * 52
        pygame.draw.ellipse(st, INK, (bx - 4, GROUND - 10, 60, 48))
        pygame.draw.ellipse(st, (214, 194, 140), (bx, GROUND - 6, 52, 40))
    if t > 1900:
        u = fx.ease_out(fx.seg(t, 1900, 2300))
        tx, ty = 770, GROUND - 120 - 30 * u
        pygame.draw.line(st, INK, (tx, ty), (tx + 130 * u, ty - 40 * u), 22)
        pygame.draw.line(st, GOLD, (tx, ty), (tx + 130 * u, ty - 40 * u), 14)
        ink_circle(st, (180, 220, 255), (tx + 130 * u, ty - 40 * u), 14, 4)
    say(st, "退一步，海阔天空。", (720, GROUND - 260), t, 300, 1500, size=26)
    boom(st, "藏锋", (720, GROUND - 330), t, 2200, 900, 72, (200, 230, 255), 0)
    title(st, sc, t, 2300)


def _edge_cannon(sc, st, t):
    me = sc.side
    cliff = [(1150, GROUND), (1440, GROUND), (1440, 732), (1110, 732)]
    pygame.draw.polygon(st, (110, 160, 210), cliff)
    fx.waves(st, pygame.Rect(1160, GROUND + 40, 280, 132), t)
    x, ph = walk(t, 0, 1300, 300, 1060)
    wobble = math.sin(t / 60.0) * 0.08 if 1300 < t < 1900 else 0
    put(st, CANNON, me, x, t, walk=ph, expr="scared" if 1300 < t < 1900 else "fierce", tilt=wobble)
    rx, rph = walk(t, 1600, 3000, -200, 720, speed=40)
    hx, hph = walk(t, 1800, 3200, -200, 520, speed=40)
    put(st, ROOK, me, rx, t, R=70, walk=rph, expr="cheer")
    put(st, HORSE, me, hx, t, R=70, walk=hph, expr="cheer")
    say(st, "哇，悬崖边……", (x, GROUND - 260), t, 1300, 2000, size=24)
    say(st, "多谢让路！", (rx + 20, GROUND - 200), t, 2200, 3200, size=26)
    title(st, sc, t, 2000)


def _river_cannon(sc, st, t):
    me = sc.side
    fx.waves(st, pygame.Rect(0, 440, 1440, 88), t)
    x = fx.lerp(-100, 1500, fx.seg(t, 0, DURATION))
    bob = math.sin(t / 200.0) * 6
    boat = [(x - 150, 470 + bob), (x + 150, 470 + bob), (x + 110, 520 + bob), (x - 110, 520 + bob)]
    put(st, CANNON, me, x, t, R=72, y=478 + bob, expr="smug", shadow=False)
    pygame.draw.polygon(st, WOOD, boat)
    pygame.draw.polygon(st, INK, boat, 5)
    tx, ty = x + 60, 478 + bob - 140
    pygame.draw.line(st, INK, (tx, ty), (tx + 110, ty - 26), 20)
    pygame.draw.line(st, GOLD, (tx, ty), (tx + 110, ty - 26), 12)
    say(st, "巡河中，一切正常！", (x + 20, 300 + bob), t, 600, 2200, size=26)
    fx.outlined_text(st, "楚 河 · 汉 界", 36, (255, 255, 255), (720, 600), ow=3)
    title(st, sc, t, 2000)


def _hook_cannon(sc, st, t):
    me = sc.side
    fx.waves(st, pygame.Rect(0, 440, 1440, 88), t)
    put(st, CANNON, me, 420, t, expr="sleepy" if t < 1900 else "surprised", y=GROUND - 10)
    bob = math.sin(t / 250.0) * 8 + (40 if 1900 < t < 2300 else 0)
    hand, tip, hook = (480, GROUND - 120), (860, GROUND - 330), (900, 470 + bob)
    fishing_rod(st, hand, tip, hook)
    if t > 1900:
        u = fx.seg(t, 1900, 2700)
        fx_x, fx_y = 900 + 80 * u, 470 - fx.hop(u) * 260
        fish = [(fx_x - 40, fx_y), (fx_x, fx_y - 22), (fx_x + 40, fx_y), (fx_x, fx_y + 22)]
        pygame.draw.polygon(st, GOLD, fish)
        pygame.draw.polygon(st, INK, fish, 4)
        tail = [(fx_x - 40, fx_y), (fx_x - 70, fx_y - 22), (fx_x - 70, fx_y + 22)]
        pygame.draw.polygon(st, GOLD, tail)
        pygame.draw.polygon(st, INK, tail, 4)
        fx.sparkles(st, (fx_x, fx_y), t, 60, count=6)
    say(st, "金钩垂下，愿者上钩～", (430, GROUND - 260), t, 300, 1800, size=26)
    boom(st, "上钩!", (1000, 300), t, 2000, 800, 76)
    title(st, sc, t, 2300)


def _cross_river_cannon(sc, st, t):
    me = sc.side
    fx.waves(st, pygame.Rect(0, 440, 1440, 88), t)
    u = fx.seg(t, 900, 1900)
    x = fx.lerp(300, 1120, fx.ease_in_out(u))
    lift = fx.hop(u) * 320
    put(st, CANNON, me, x, t, lift=lift, expr="fierce" if u < 1 else "cheer", squash=0.25 * (fx.hop(fx.seg(t, 700, 900)) + fx.hop(fx.seg(t, 1900, 2100))),
        tilt=0.3 * math.sin(u * math.pi))
    fx.dust(st, (1120, GROUND - 6), fx.seg(t, 1900, 2500), 80, seed=3)
    for k, sx in enumerate((540, 720, 900)):
        fx.shockwave(st, (sx, 484), fx.seg(t, 1100 + k * 150, 1700 + k * 150), 90, (230, 245, 255), 5, flat=0.3)
    say(st, "先发制人！", (1120, GROUND - 260), t, 2000, 3200, size=28, tail="down-right")
    title(st, sc, t, 2100)


def _immortal_pawn(sc, st, t):
    me = sc.side
    drifting_clouds(st, t, 3, 120)
    path = fx.seg(t, 900, 1800)
    if path > 0:
        layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
        end_x = 300 + 1000 * fx.ease_out(path)
        pygame.draw.polygon(layer, (255, 236, 150, 150), [(300, GROUND + 60), (end_x, GROUND - 20), (end_x, GROUND + 10), (300, GROUND + 150)])
        st.blit(layer, (0, 0))
        fx.sparkles(st, (end_x, GROUND), t, 80, count=6)
    hand_u = fx.ease_out(fx.seg(t, 200, 900))
    hx, hy = 1180, -80 + 280 * hand_u
    skin = (255, 226, 196)
    sleeve = [(hx + 10, hy - 140), (hx + 150, hy - 140), (hx + 70, hy + 20), (hx - 40, hy + 10)]
    pygame.draw.polygon(st, (250, 250, 255), sleeve)
    pygame.draw.polygon(st, INK, sleeve, 5)
    pygame.draw.line(st, (230, 200, 120), (hx - 30, hy + 4), (hx + 66, hy + 14), 8)
    palm = pygame.Rect(int(hx - 70), int(hy - 10), 90, 70)
    pygame.draw.ellipse(st, INK, palm.inflate(10, 10))
    pygame.draw.ellipse(st, skin, palm)
    for k in range(3):
        pygame.draw.ellipse(st, INK, (int(hx - 60 + k * 22), int(hy + 40), 26, 34))
        pygame.draw.ellipse(st, skin, (int(hx - 57 + k * 22), int(hy + 43), 20, 26))
    tip = (hx - 170, hy + 110)
    pygame.draw.line(st, INK, (hx - 50, hy + 30), tip, 30)
    pygame.draw.line(st, skin, (hx - 50, hy + 30), tip, 20)
    pygame.draw.circle(st, skin, (int(tip[0]), int(tip[1])), 10)
    fx.cloud(st, hx + 110, hy - 120, 100)
    glow = fx.seg(t, 700, 1100)
    if glow > 0:
        fx.sparkles(st, tip, t, 70, (255, 240, 150), count=8)
        layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
        pygame.draw.line(layer, (255, 236, 150, int(170 * glow)), tip, (320, GROUND + 60), 10)
        st.blit(layer, (0, 0))
    x, ph = walk(t, 1600, 3200, 300, 900, speed=55)
    put(st, PAWN, me, x, t, walk=ph, expr="surprised" if t < 1600 else "cheer")
    say(st, "此路可行。", (hx - 140, hy + 140), t, 900, 1900, size=26, tail="up-right")
    say(st, "前方，有路！", (x + 20, GROUND - 240), t, 1800, 3000, size=28)
    title(st, sc, t, 2200)


def _edge_pawn(sc, st, t):
    me, foe = sc.side, other(sc.side)
    put(st, KING, foe, 1150, t, R=70, facing=-1, expr="surprised" if t > 1500 else "sleepy")
    x, ph = walk(t, 0, 2600, -100, 760, speed=60, ease=lambda u: u)
    put(st, PAWN, me, x, t, walk=ph, squash=0.25, expr="smug")
    for k in range(16):
        rx = 40 + k * 60
        sway = math.sin(t / 400.0 + k) * 8
        pygame.draw.line(st, INK, (rx, GROUND + 30), (rx + sway, GROUND - 120 - (k % 3) * 20), 9)
        pygame.draw.line(st, (150, 190, 100), (rx, GROUND + 30), (rx + sway, GROUND - 120 - (k % 3) * 20), 5)
        pygame.draw.ellipse(st, (160, 120, 80), (rx + sway - 8, GROUND - 150 - (k % 3) * 20, 16, 40))
    say(st, "嘘——走边路。", (x + 10, GROUND - 210), t, 400, 1600, size=26)
    if t > 1500:
        question_marks(st, (1150, GROUND - 250), t)
    title(st, sc, t, 2000)


def _center_pawn(sc, st, t):
    me = sc.side
    x, ph = walk(t, 300, 2400, 100, 900, speed=30, ease=fx.ease_in)
    fx.speed_lines(st, pygame.Rect(0, GROUND - 260, int(x), 240), t, 1, count=12, alpha=180 * fx.seg(t, 600, 900))
    fx.banner(st, x - 140, GROUND - 40, 260, sc.color, t, "冲")
    put(st, PAWN, me, x, t, walk=ph, expr="fierce", swing=0.9, tilt=0.15)
    say(st, "中路冲锋，冲啊！", (x + 20, GROUND - 240), t, 600, 2200, size=28, shout=True)
    title(st, sc, t, 2000)


def _flying_elephant(sc, st, t):
    me = sc.side
    drifting_clouds(st, t, 4, 140)
    fx.palace(st, 720, GROUND + 40, 1.0)
    u = fx.seg(t, 300, 2100)
    x = fx.lerp(150, 720, fx.ease_in_out(u))
    lift = fx.hop(u) * 280 * (1 if u < 1 else 0)
    if 0 < u < 1:
        wings(st, (x - 20, GROUND - lift - 150), 150, t)
    put(st, ELEPHANT, me, x, t, lift=lift, expr="cheer" if u < 1 else "smug", tilt=-0.1 * math.sin(u * math.pi * 2))
    fx.dust(st, (720, GROUND - 6), fx.seg(t, 2100, 2700), 110, seed=5)
    fx.shockwave(st, (720, GROUND), fx.seg(t, 2100, 2700), 320, (255, 255, 255), 8, flat=0.25)
    say(st, "飞相镇中！", (x + 40, GROUND - 260 - lift), t, 2000, 3300, size=30)
    title(st, sc, t, 2200)


def _edge_elephant(sc, st, t):
    me = sc.side
    for bx in (80, 200, 1320):
        fx.bamboo(st, bx, GROUND, 420, t)
    ex, eph = walk(t, 400, 1400, 700, 1150, speed=90)
    put(st, ELEPHANT, me, ex, t, walk=eph, expr="cheer" if t > 1400 else "idle", facing=1 if t < 1400 else -1)
    hx, hph = walk(t, 1300, 2800, -150, 760, speed=36)
    put(st, HORSE, me, hx, t, walk=hph, expr="cheer", lift=fx.hop((t / 300.0) % 1.0) * 30 if 1300 < t < 2800 else 0)
    say(st, "马兄，请！", (ex - 20, GROUND - 250), t, 1300, 2300, size=26, tail="down-right")
    say(st, "多谢让路！", (hx + 20, GROUND - 230), t, 2200, 3300, size=26)
    title(st, sc, t, 2200)


def _horse_opening(sc, st, t):
    me = sc.side
    fx.sun(st, (720, 260), 90, t)
    door = pygame.Rect(540, GROUND - 330, 360, 330)
    open_u = fx.ease_out(fx.seg(t, 500, 900))
    pygame.draw.rect(st, INK, door.inflate(16, 16))
    pygame.draw.rect(st, (60, 40, 30), door)
    for sgn, base in ((-1, door.x), (1, door.right)):
        w = int(door.w / 2 * (1 - open_u))
        rect = pygame.Rect(base if sgn < 0 else base - w, door.y, w, door.h)
        pygame.draw.rect(st, WOOD, rect)
        pygame.draw.rect(st, INK, rect, 4)
    rear = fx.seg(t, 1100, 1600)
    x, ph = walk(t, 1700, 2600, 720, 1000, speed=36)
    if t > 700:
        put(st, HORSE, me, x, t, walk=ph, expr="fierce", tilt=-0.5 * fx.hop(rear), swing=fx.seg(t, 1200, 1500))
    fx.radial_lines(st, (720, GROUND - 160), t, 160, 800, (255, 240, 200), count=24, alpha=150 * fx.hop(fx.seg(t, 1000, 2000)))
    boom(st, "嘶——!", (900, GROUND - 330), t, 1200, 900, 92)
    fx.dust(st, (x, GROUND - 6), fx.seg(t, 1600, 2200), 80, seed=9)
    title(st, sc, t, 2100)


def _edge_horse(sc, st, t):
    me = sc.side
    for bx in (420, 560, 880, 1020, 1160):
        fx.bamboo(st, bx, GROUND, 460, t)
    x, ph = walk(t, 0, 3000, -150, 1500, speed=36, ease=lambda u: u)
    put(st, HORSE, me, x, t, R=75, y=GROUND + 90, walk=ph, expr="smug", lift=fx.hop((t / 280.0) % 1.0) * 22)
    fx.grass(st, pygame.Rect(0, GROUND + 100, 1440, 60), t, count=60)
    say(st, "绕个远路，出其不意。", (min(x, 1100) + 20, GROUND - 140), t, 700, 2400, size=26)
    title(st, sc, t, 2000)


def _screen_horses(sc, st, t):
    me = sc.side
    unfold = fx.ease_out(fx.seg(t, 1300, 2000))
    for k in range(6 if unfold > 0.02 else 0):
        w = 120 * unfold
        x = 720 - 360 * unfold + k * w
        panel = pygame.Rect(int(x), GROUND - 330, max(1, int(w)), 300)
        pygame.draw.rect(st, (250, 240, 220), panel)
        pygame.draw.rect(st, INK, panel, 4)
        if w > 30:
            fx.mountains(st, pygame.Rect(panel.x + 6, panel.y + 120, panel.w - 12, 150), (150, 170, 160), seed=k, peaks=2, height=0.8, outline=False)
    lx, lph = walk(t, 0, 1200, -150, 560, speed=36)
    rx, rph = walk(t, 0, 1200, 1590, 880, speed=36)
    put(st, HORSE, me, lx, t, walk=lph, expr="fierce" if t < 1200 else "smug")
    put(st, HORSE, me, rx, t, walk=rph, facing=-1, expr="fierce" if t < 1200 else "smug")
    say(st, "双马屏风，", (lx, GROUND - 250), t, 1300, 2600, size=26)
    say(st, "固若金汤！", (rx, GROUND - 250), t, 1600, 2900, size=26, tail="down-right")
    title(st, sc, t, 2200)


def _cannon_duel(sc, st, t, crossed: bool):
    me, foe = sc.side, other(sc.side)
    left_side, right_side = (foe, me)
    put(st, CANNON, left_side, 360, t, expr="fierce", swing=fx.seg(t, 1200, 1400))
    put(st, CANNON, right_side, 1080, t, facing=-1, expr="fierce", swing=fx.seg(t, 1250, 1450))
    if not crossed:
        fx.radial_lines(st, (720, GROUND - 230), t, 30, 300, (255, 250, 220), count=18, alpha=200 * fx.hop(fx.seg(t, 900, 1900)))
        fx.impact_star(st, (720, GROUND - 230), fx.seg(t, 1350, 1900), 110)
        boom(st, "针锋相对!", (720, GROUND - 350), t, 1100, 1100, 72)
        say(st, "来呀！", (360, GROUND - 260), t, 300, 1200, size=28)
        say(st, "谁怕谁！", (1080, GROUND - 260), t, 500, 1400, size=28, tail="down-right")
    else:
        fx.arrow_projectile(st, (470, GROUND - 240), (720, GROUND - 330), fx.seg(t, 1300, 1700), size=18, arc=120)
        fx.arrow_projectile(st, (970, GROUND - 240), (720, GROUND - 330), fx.seg(t, 1350, 1700), size=18, arc=120)
        fx.explosion(st, (720, GROUND - 330), fx.seg(t, 1700, 2500), 150, seed=13)
        if t > 1900:
            fx.confetti(st, pygame.Rect(400, 0, 640, 500), t - 1900, 50)
        say(st, "开炮！", (360, GROUND - 260), t, 400, 1300, size=28)
        say(st, "开炮！", (1080, GROUND - 260), t, 500, 1400, size=28, tail="down-right")
    title(st, sc, t, 2000)


def _advisor_opening(sc, st, t):
    me = sc.side
    u = fx.seg(t, 300, 1200)
    x = fx.lerp(500, 720, fx.ease_in_out(u))
    put(st, ADVISOR, me, x, t, y=GROUND - 60 * (1 - u) + 40 * u, expr="smug", tilt=0.3 * fx.hop(fx.seg(t, 1300, 1800)))
    say(st, "臣，前来护驾。", (x + 20, GROUND - 240), t, 1200, 2400, size=26)
    title(st, sc, t, 2000)


def _rook_opening(sc, st, t):
    me = sc.side
    rev = fx.seg(t, 300, 1200)
    u = fx.seg(t, 1200, 2000)
    x = fx.lerp(260, 1700, fx.ease_in(u))
    put(st, ROOK, me, x, t, walk=t / (40 - 30 * rev), expr="fierce", squash=0.15 * rev * (1 - u) + math.sin(t / 30.0) * 0.03 * rev, tilt=-0.1 * rev + 0.2 * u)
    for k in range(5):
        fx.dust(st, (x - 120 - k * 60, GROUND - 6), fx.seg(t, 400 + k * 150, 1000 + k * 150), 40, seed=k)
    if u > 0:
        fx.speed_lines(st, pygame.Rect(0, GROUND - 280, 1440, 260), t, 1, count=18, alpha=200)
    boom(st, "出车!", (720, GROUND - 330), t, 1300, 900, 92)
    title(st, sc, t, 2000)


def _king_opening(sc, st, t):
    me = sc.side
    x, ph = walk(t, 300, 1800, 720, 720, speed=90)
    put(st, KING, me, 720, t, walk=t / 90 if 300 < t < 1800 else 0, expr="smug" if t > 1800 else "fierce", swing=fx.seg(t, 1900, 2200))
    fx.banner(st, 330, GROUND, 330, sc.color, t, "帅")
    fx.banner(st, 1020, GROUND, 330, sc.color, t, "帅")
    if t > 1900:
        fx.confetti(st, pygame.Rect(0, 0, 1440, 560), t - 1900, 70)
    say(st, "本帅亲自出征！", (740, GROUND - 270), t, 1000, 2400, size=28)
    title(st, sc, t, 2100)


def _generic(sc, st, t):
    put(st, sc.event.actor.kind, sc.side, 720, t, expr="cheer")
    title(st, sc, t, 800)


PAINTERS = {
    "center_cannon": (_center_cannon, "fortress"),
    "cross_palace_cannon": (_cross_palace, "courtyard"),
    "advisor_cannon": (_advisor_cannon, "courtyard"),
    "pawn_base_cannon": (_pawn_base, "riverbank"),
    "retreat_cannon": (_retreat_cannon, "fortress"),
    "edge_cannon": (_edge_cannon, "plains"),
    "river_cannon": (_river_cannon, "riverbank"),
    "hook_cannon": (_hook_cannon, "riverbank"),
    "cross_river_cannon": (_cross_river_cannon, "riverbank"),
    "immortal_pawn": (_immortal_pawn, "ink"),
    "edge_pawn": (_edge_pawn, "riverbank"),
    "center_pawn": (_center_pawn, "plains"),
    "flying_elephant": (_flying_elephant, "sky"),
    "edge_elephant": (_edge_elephant, "jungle"),
    "horse_opening": (_horse_opening, "grassland"),
    "edge_horse": (_edge_horse, "jungle"),
    "screen_horses": (_screen_horses, "grassland"),
    "same_cannons": (lambda sc, st, t: _cannon_duel(sc, st, t, False), "fortress"),
    "crossed_cannons": (lambda sc, st, t: _cannon_duel(sc, st, t, True), "fortress"),
    "advisor_opening": (_advisor_opening, "courtyard"),
    "rook_opening": (_rook_opening, "plains"),
    "king_opening": (_king_opening, "throne"),
}


def opening_scene(event):
    painter, backdrop = PAINTERS.get(event.key, (_generic, "ink"))
    return LambdaCutscene(event, painter, DURATION, backdrop)
