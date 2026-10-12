"""Finale skits for named mates (双车错、卧槽马、马后炮、海底捞月 …).

The same painters double as shorter "check" scenes: when ``scene.finale`` is false the
king survives the final blow and scrambles away instead of being knocked out.
"""

from __future__ import annotations

import math

import pygame

from game_engine import ADVISOR, CANNON, ELEPHANT, HORSE, KING, PAWN, ROOK
from cinema import fx
from cinema.actors import INK, STEEL, actor
from cinema.kit import boom, drifting_clouds, fishing_rod, iron_bars, other, put, question_marks, say
from cinema.scenes_capture import flying_prop
from cinema.stage import GROUND, LambdaCutscene, title_card

TIMELINE = 4400
CHECK_DURATION = 2900


def mtitle(st, sc, tt: float, start: float, center=(720, 118), size: int = 86) -> None:
    title_card(st, sc.event.title, sc.event.subtitle, center, fx.seg(tt, start, TIMELINE - 100), size=size)


def seal_stamp(st, text: str, center, tt: float, t0: float, color=(220, 40, 60)) -> None:
    u = fx.seg(tt, t0, t0 + 260)
    if u <= 0:
        return
    scale = 1 + 2.5 * (1 - fx.ease_out(u))
    size = int(150 * scale)
    layer = pygame.Surface((size + 20, size + 20), pygame.SRCALPHA)
    rect = pygame.Rect(10, 10, size, size)
    pygame.draw.rect(layer, (*color, 230), rect, border_radius=int(size * 0.12))
    pygame.draw.rect(layer, (255, 236, 220, 255), rect.inflate(-int(size * 0.12), -int(size * 0.12)), max(2, int(size * 0.03)), border_radius=int(size * 0.1))
    f = fx.font(size * 0.36, True)
    for i, ch in enumerate(text[:2]):
        img = f.render(ch, True, (255, 240, 230))
        layer.blit(img, img.get_rect(center=(rect.centerx, rect.y + rect.h * (0.3 + 0.4 * i))))
    layer = pygame.transform.rotozoom(layer, 12, 1.0)
    st.blit(layer, layer.get_rect(center=(int(center[0]), int(center[1]))))


def king_fate(sc, st, x: float, tt: float, hit: float, y: float = GROUND, R: float = 90, facing: int = -1,
              lift: float = 0.0, calm_until: float = 600) -> None:
    """Draw the defending king: nervous before ``hit``; KO (finale) or narrow escape (check) after."""
    foe = other(sc.side)
    if tt < hit:
        expr = "idle" if tt < calm_until else "scared"
        jitter = math.sin(tt * 0.09) * 3 if tt > calm_until else 0
        put(st, KING, foe, x + jitter, tt, R=R, y=y, facing=facing, expr=expr, lift=lift)
        return
    r = tt - hit
    flash = 1 - fx.seg(r, 0, 160)
    fx.impact_star(st, (x, y - R * 1.2 - lift), fx.seg(r, 0, 420), R * 1.7, seed=77)
    if sc.finale:
        fall = fx.ease_bounce(fx.seg(r, 120, 700))
        put(st, KING, foe, x + 40 * fall * -facing, tt, R=R, y=y, facing=facing, expr="ko", tilt=-1.45 * fall,
            props={"no_hat", "no_weapon"}, flash=flash, lift=lift * (1 - fx.seg(r, 0, 400)))
        u = fx.seg(r, 0, 800)
        flying_prop(st, KING, (x - facing * 160 * u, y - 2.4 * R - fx.hop(u) * 140 + 200 * u), R * 0.5, u * 6, sc.color)
        if r > 600:
            fx.dizzy_stars(st, (x - facing * 60, y - R * 1.2), tt, R * 0.7)
        seal_stamp(st, "将死", (x + 40, y - 3.2 * R), tt, hit + 500)
        if r > 700:
            fx.confetti(st, pygame.Rect(0, 0, 1440, 560), r - 700, 60)
    else:
        back = fx.ease_out(fx.seg(r, 100, 600))
        put(st, KING, foe, x - facing * 140 * back, tt, R=R, y=y, facing=facing, expr="hurt" if r < 400 else "scared",
            flash=flash, lift=fx.hop(fx.seg(r, 100, 600)) * 80 + lift * (1 - back))
        say(st, "好险！", (x - facing * 140 * back, y - 2.6 * R), tt, hit + 350, TIMELINE, size=30, tail="down-right")


# --- individual skits -----------------------------------------------------------

def _double_rook(sc, st, tt):
    me = sc.side
    hit = 2700
    bx = fx.lerp(1640, 1060, fx.ease_out(fx.seg(tt, 300, 1100)))
    put(st, ROOK, me, bx, tt, R=74, y=GROUND - 70, facing=-1, walk=tt / 30 if tt < 1100 else 0, expr="fierce",
        swing=fx.seg(tt, hit - 200, hit))
    king_fate(sc, st, 760, tt, hit, calm_until=500)
    ax = fx.lerp(-200, 470, fx.ease_out(fx.seg(tt, 200, 1000)))
    put(st, ROOK, me, ax, tt, walk=tt / 30 if tt < 1000 else 0, expr="fierce", swing=fx.seg(tt, hit - 200, hit))
    layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
    for k, (y, x0, x1, t0) in enumerate(((GROUND - 140, 560, 1440, 1300), (GROUND - 230, 0, 980, 1650))):
        u = fx.seg(tt, t0, t0 + 250)
        if u > 0:
            pulse = 0.6 + 0.4 * math.sin(tt / 60.0 + k)
            xe = x0 + (x1 - x0) * u if k == 0 else x1 - (x1 - x0) * u
            a, b = (x0, y), (xe, y)
            if k == 1:
                a, b = (x1, y), (xe, y)
            pygame.draw.line(layer, (255, 80, 90, int(200 * pulse)), a, b, 16)
            pygame.draw.line(layer, (255, 230, 230, 255), a, b, 5)
    st.blit(layer, (0, 0))
    boom(st, "错!", (620, GROUND - 330), tt, 1350, 600, 90, (255, 120, 120))
    boom(st, "错!", (900, GROUND - 380), tt, 1700, 600, 90, (255, 120, 120), 8)
    say(st, "长车封线，", (ax + 20, GROUND - 260), tt, 900, 2000, size=26)
    say(st, "短车锁喉！", (bx - 20, GROUND - 330), tt, 1300, 2400, size=26, tail="down-right")
    mtitle(st, sc, tt, 2900)


def _wocao_horse(sc, st, tt):
    me, foe = sc.side, other(sc.side)
    hit = 2600
    put(st, ELEPHANT, foe, 1290, tt, R=62, facing=-1, expr="sleepy" if tt < 1300 else "surprised")
    if tt < 1300:
        fx.outlined_text(st, "Zz", 34, (255, 255, 255), (1330, GROUND - 190 - (tt / 20) % 30), ow=2)
    put(st, ADVISOR, foe, 720, tt, R=62, facing=-1, expr="scared" if tt > 1500 else "idle")
    king_fate(sc, st, 860, tt, hit, R=80)
    put(st, ADVISOR, foe, 1000, tt, R=62, facing=-1, expr="scared" if tt > 1500 else "idle")
    u = fx.seg(tt, 400, 1400)
    hx = fx.lerp(-150, 1150, fx.ease_in_out(u))
    crouch = fx.seg(tt, 1400, 1600)
    put(st, HORSE, me, hx, tt, facing=-1 if u >= 1 else 1, lift=fx.hop(u) * 330, expr="fierce",
        squash=0.3 * crouch * (1 - fx.seg(tt, hit - 250, hit)), tilt=-0.35 * fx.hop(fx.seg(tt, hit - 300, hit)), swing=fx.seg(tt, hit - 150, hit + 100))
    fx.dust(st, (1150, GROUND - 6), fx.seg(tt, 1400, 2000), 70, seed=3)
    if 1700 < tt < hit:
        dash = fx.seg(tt, 1700, 2100)
        pts = [(1150, GROUND - 120), (1150, GROUND - 250), (900, GROUND - 250)]
        for i in range(int(dash * 20)):
            u2 = i / 20
            p = fx.lerp2(pts[0], pts[1], u2 * 2) if u2 < 0.5 else fx.lerp2(pts[1], pts[2], u2 * 2 - 1)
            pygame.draw.circle(st, (255, 220, 90), (int(p[0]), int(p[1])), 7)
    say(st, "卧槽在此！", (1150, GROUND - 260), tt, 1500, 2500, size=28, tail="down-right")
    say(st, "让开，让我出去！", (860, GROUND - 240), tt, 1900, 2700, size=22, tail="down-left")
    mtitle(st, sc, tt, 2900)


def _corner_horse(sc, st, tt):
    me = sc.side
    hit = 2400
    corner = (1148, 236)
    king_fate(sc, st, 720, tt, hit, R=80)
    if tt < 1700:
        swing = math.sin(tt / 260.0) * 0.4
        hx = corner[0] + math.sin(swing) * 70
        hy = corner[1] + 40 + math.cos(swing) * 70
        pygame.draw.line(st, INK, corner, (hx, hy - 60), 5)
        actor(st, HORSE, me, (hx, hy + 80), 70, t=tt, facing=-1, expr="smug", tilt=swing, pivot="center", shadow=False)
    else:
        u = fx.seg(tt, 1700, hit)
        x = fx.lerp(corner[0], 800, fx.ease_in(u))
        y = fx.lerp(corner[1] + 190, GROUND, fx.ease_in(u))
        actor(st, HORSE, me, (x, y), 80, t=tt, facing=-1, expr="fierce", tilt=0.4 * (1 - u), swing=u, shadow=u > 0.5)
    say(st, "挂角一刺！", (corner[0] - 40, corner[1] + 20), tt, 600, 1700, size=26, tail="down-right")
    mtitle(st, sc, tt, 2800)


def _fishing_horse(sc, st, tt):
    me = sc.side
    fx.waves(st, pygame.Rect(0, 440, 1440, 88), tt)
    hit = 2500
    put(st, HORSE, me, 300, tt, expr="smug", y=GROUND - 10)
    hook_y = fx.lerp(260, GROUND - 260, fx.ease_out(fx.seg(tt, 600, 1300)))
    catch = fx.seg(tt, 1400, 2300)
    lift = 260 * fx.ease_out(catch) if catch > 0 else 0
    hook = (1000, hook_y - lift)
    fishing_rod(st, (380, GROUND - 160), (900, 160 - 40 * catch), hook)
    king_fate(sc, st, 1000, tt, hit, lift=lift if tt < hit + 300 or not sc.finale else lift)
    say(st, "钓鱼，要有耐心。", (330, GROUND - 260), tt, 300, 1400, size=24)
    boom(st, "上钩了!", (1000, 220), tt, 1500, 900, 80)
    mtitle(st, sc, tt, 2800)


def _horse_cannon(sc, st, tt):
    me = sc.side
    hit = 2000
    king_fate(sc, st, 1100, tt, hit)
    duck = fx.hop(fx.seg(tt, 1350, 1900))
    put(st, HORSE, me, 860, tt, R=80, expr="smug" if duck < 0.1 else "scared", squash=0.4 * duck)
    put(st, CANNON, me, 360, tt, expr="fierce", swing=fx.seg(tt, 1300, 1550))
    fx.explosion(st, (480, GROUND - 240), fx.seg(tt, 1300, 1700), 60, seed=4)
    fx.arrow_projectile(st, (480, GROUND - 240), (1100, GROUND - 110), fx.seg(tt, 1320, hit), size=20, arc=220)
    fx.explosion(st, (1100, GROUND - 110), fx.seg(tt, hit, hit + 900), 160, seed=5)
    say(st, "我在前头堵门，", (860, GROUND - 240), tt, 300, 1300, size=24)
    say(st, "我在后头开炮！", (360, GROUND - 260), tt, 800, 1800, size=24)
    mtitle(st, sc, tt, 2700)


def _double_cannon(sc, st, tt):
    me = sc.side
    hit = 1900
    king_fate(sc, st, 1100, tt, hit)
    put(st, CANNON, me, 380, tt, R=80, expr="fierce", swing=fx.seg(tt, 1450, 1650))
    actor(st, CANNON, me, (390, GROUND - 205), 80, t=tt, expr="cheer", swing=fx.seg(tt, 1250, 1450), shadow=False)
    for k, (y, t0) in enumerate(((GROUND - 400, 1250), (GROUND - 200, 1450))):
        fx.arrow_projectile(st, (500, y), (1100, GROUND - 120), fx.seg(tt, t0, t0 + 450), size=20)
        fx.explosion(st, (1100 - k * 30, GROUND - 130 - k * 30), fx.seg(tt, t0 + 450, t0 + 1250), 150, seed=k)
    boom(st, "重炮!", (720, GROUND - 400), tt, 1300, 900, 96, (255, 140, 90))
    say(st, "叠罗汉，", (380, GROUND - 440), tt, 300, 1200, size=24)
    mtitle(st, sc, tt, 2700)


def _heaven_earth(sc, st, tt):
    me = sc.side
    hit = 2100
    drifting_clouds(st, tt, 2, 300)
    pygame.draw.rect(st, (200, 190, 220), (0, GROUND, 1440, 200))
    pygame.draw.line(st, INK, (0, GROUND), (1440, GROUND), 4)
    king_fate(sc, st, 1000, tt, hit)
    fx.cloud(st, 340, 330, 90)
    actor(st, CANNON, me, (340, 300), 70, t=tt, expr="fierce", tilt=0.5, swing=fx.seg(tt, 1500, 1700), shadow=False)
    put(st, CANNON, me, 330, tt, R=70, expr="fierce", squash=0.2, swing=fx.seg(tt, 1600, 1800))
    fx.outlined_text(st, "天", 80, (255, 236, 150), (200, 200), ow=5)
    fx.outlined_text(st, "地", 80, (255, 236, 150), (180, GROUND - 120), ow=5)
    layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
    for (a, t0) in (((420, 280), 1500), ((420, GROUND - 160), 1600)):
        u = fx.seg(tt, t0, t0 + 400)
        if u > 0:
            b = fx.lerp2(a, (1000, GROUND - 120), u)
            pygame.draw.line(layer, (255, 200, 90, 200), a, b, 26)
            pygame.draw.line(layer, (255, 255, 230, 255), a, b, 8)
    st.blit(layer, (0, 0))
    fx.explosion(st, (1000, GROUND - 120), fx.seg(tt, hit, hit + 900), 190, seed=8)
    say(st, "天上一炮，", (360, 190), tt, 400, 1400, size=24)
    say(st, "地下一炮！", (360, GROUND - 230), tt, 800, 1800, size=24)
    mtitle(st, sc, tt, 2800)


def _iron_bolt(sc, st, tt):
    me = sc.side
    hit = 2600
    door = pygame.Rect(0, 0, 187, 204)
    door.midbottom = (720, GROUND)
    banging = 1600 < tt < hit
    king_fate(sc, st, 720 + (math.sin(tt / 40.0) * 6 if banging else 0), tt, hit, R=70)
    bolt_u = fx.ease_out(fx.seg(tt, 1100, 1450))
    bx = fx.lerp(200, door.x - 20, bolt_u)
    pygame.draw.rect(st, INK, (int(bx) - 4, GROUND - 130, door.w + 48, 40), border_radius=10)
    pygame.draw.rect(st, STEEL, (int(bx), GROUND - 126, door.w + 40, 32), border_radius=8)
    for k in range(3):
        pygame.draw.circle(st, INK, (int(bx + 30 + k * 70), GROUND - 110), 6)
    boom(st, "咔嚓!", (720, GROUND - 230), tt, 1450, 700, 80, (220, 230, 255))
    actor(st, CANNON, me, (720, 190), 60, t=tt, expr="fierce", facing=1, tilt=1.3, swing=fx.seg(tt, hit - 250, hit), shadow=False)
    put(st, ROOK, me, 1040, tt, R=80, facing=-1, expr="smug")
    if banging:
        say(st, "开门！开门！", (720, GROUND - 220), tt, 1600, hit, size=26, tail="down-right", shout=True)
    fx.arrow_projectile(st, (720, 260), (720, GROUND - 100), fx.seg(tt, hit - 200, hit), size=18)
    say(st, "门，栓上了。", (1040, GROUND - 260), tt, 500, 1400, size=24, tail="down-right")
    mtitle(st, sc, tt, 2900)


def _heart_strike(sc, st, tt):
    me, foe = sc.side, other(sc.side)
    sacrifice = sc.event.kind == "sacrifice"
    hit = 2200
    king_x = 1180
    if sacrifice:
        put(st, KING, foe, king_x, tt, facing=-1, expr="scared" if tt > 1400 else "idle")
    else:
        king_fate(sc, st, king_x, tt, hit)
    adv_hit = 1400
    if tt < adv_hit:
        put(st, ADVISOR, foe, 920, tt, facing=-1, expr="fierce" if tt > 800 else "idle")
    else:
        u = fx.seg(tt, adv_hit, adv_hit + 900)
        actor(st, ADVISOR, foe, (920 + 300 * u, GROUND - fx.hop(u) * 400 - 200 * u), 90 * (1 - 0.5 * u), t=tt, facing=-1,
              expr="ko", tilt=u * 9, pivot="center", alpha=int(255 * (1 - fx.seg(u, 0.8, 1))))
    x = fx.lerp(150, 840, fx.ease_in(fx.seg(tt, 900, adv_hit)))
    if tt > adv_hit and not sacrifice:
        x = fx.lerp(840, king_x - 220, fx.ease_in(fx.seg(tt, adv_hit + 300, hit)))
    put(st, ROOK, me, x, tt, walk=tt / 25 if 900 < tt < hit else 0, expr="fierce", swing=fx.seg(tt, adv_hit - 150, adv_hit + 100), tilt=0.12)
    if 900 < tt < hit:
        fx.speed_lines(st, pygame.Rect(0, GROUND - 260, int(x), 240), tt, 1, count=14, alpha=200)
    hu = fx.seg(tt, adv_hit, adv_hit + 900)
    if 0 < hu < 1:
        cx, cy = 920, GROUND - 280
        s = 70 * fx.ease_back(min(1.0, hu * 3))
        heart = [(cx, cy + s * 0.9)] + [(cx + math.sin(a) ** 3 * s, cy - (13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a)) * s / 16)
                                       for a in [k * math.tau / 40 for k in range(40)]]
        pygame.draw.polygon(st, (255, 70, 100), heart)
        pygame.draw.polygon(st, INK, heart, 5)
        crack = fx.seg(hu, 0.2, 0.5)
        pts = [(cx, cy - s * 0.6), (cx - 14, cy - s * 0.2), (cx + 12, cy + s * 0.1), (cx - 6, cy + s * 0.5)]
        pygame.draw.lines(st, INK, False, pts[: 2 + int(crack * 2)], 6)
    boom(st, "穿心!", (900, GROUND - 400), tt, adv_hit, 800, 96, (255, 110, 130))
    say(st, "大胆穿心，弃车也值！", (x + 20, GROUND - 260), tt, 200, 1100, size=24)
    mtitle(st, sc, tt, 2600)


def _facing_kings(sc, st, tt):
    me = sc.side
    hit = 2600
    pygame.draw.rect(st, (120, 180, 220), (660, GROUND, 120, 172))
    fx.outlined_text(st, "楚河汉界", 22, (255, 255, 255), (720, GROUND + 60), ow=2)
    sight = fx.seg(tt, 800, 1200)
    if sight > 0:
        layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
        for k in range(int(sight * 24)):
            x = 400 + k * 30
            pygame.draw.circle(layer, (255, 240, 160, 220), (x, GROUND - 160), 6)
        st.blit(layer, (0, 0))
    laugh = tt > 1200
    put(st, KING, me, 300, tt, expr="cheer" if laugh else "smug", tilt=-0.1 * math.sin(tt / 70.0) if laugh else 0)
    put(st, ROOK, me, 1330, tt, R=70, facing=-1, expr="fierce")
    king_fate(sc, st, 1080, tt, hit)
    pale = fx.seg(tt, 1500, 2000)
    if pale > 0 and tt < hit:
        layer = pygame.Surface((180, 120), pygame.SRCALPHA)
        pygame.draw.ellipse(layer, (255, 255, 255, int(200 * pale)), layer.get_rect())
        st.blit(layer, (1080 - 90, GROUND - 1.18 * 90 - 150))
    if laugh:
        for k in range(4):
            u = ((tt - 1200) / 700.0 + k * 0.25) % 1.0
            fx.outlined_text(st, "哈", 40 + 30 * u, (255, 236, 150), (400 + u * 560, GROUND - 200 - math.sin(u * 6) * 30), ow=3,
                             alpha=int(255 * (1 - u)))
    say(st, "王不见王——哈哈哈！", (300, GROUND - 270), tt, 1200, 2400, size=24)
    say(st, "我、我脸都白了……", (1080, GROUND - 260), tt, 1900, hit, size=22, tail="down-right")
    mtitle(st, sc, tt, 2900)


def _sea_moon(sc, st, tt):
    me = sc.side
    hit = 2500
    fx.moon(st, (720, 150), 70)
    fx.waves(st, pygame.Rect(0, 330, 1440, 60), tt, (30, 90, 160), (120, 180, 230), rows=2)
    shimmer = pygame.Surface((220, 60), pygame.SRCALPHA)
    pygame.draw.ellipse(shimmer, (255, 246, 210, 150), shimmer.get_rect())
    st.blit(shimmer, (int(720 - 110 + math.sin(tt / 300.0) * 6), 470))
    rock = [(820, 470), (1000, 470), (1040, 520), (780, 520)]
    pygame.draw.polygon(st, (90, 90, 110), rock)
    pygame.draw.polygon(st, INK, rock, 4)
    king_fate(sc, st, 910, tt, hit, y=472, R=70)
    dive = fx.seg(tt, 400, 1100)
    if tt < 1100:
        x = fx.lerp(200, 520, dive)
        actor(st, ROOK, me, (x, 330 + fx.ease_in(dive) * 260 - fx.hop(dive) * 160), 70, t=tt, tilt=dive * 2.0, pivot="center", expr="fierce", shadow=False)
    fx.shockwave(st, (560, 360), fx.seg(tt, 1050, 1600), 160, (230, 245, 255), 6, flat=0.3)
    boom(st, "扑通!", (560, 280), tt, 1050, 700, 66, (200, 230, 255))
    rise = fx.seg(tt, 1700, 2200)
    if rise > 0:
        x = 1120
        y = 640 - 200 * fx.ease_out(rise)
        net_u = fx.seg(tt, 2200, hit)
        actor(st, ROOK, me, (x, y), 70, t=tt, facing=-1, expr="cheer" if net_u > 0 else "fierce", swing=net_u, shadow=False)
        cx, cy = x - 140 - 60 * net_u, y - 160 - 80 * net_u
        pygame.draw.line(st, INK, (x - 60, y - 90), (cx, cy), 8)
        pygame.draw.arc(st, INK, (int(cx - 90), int(cy - 60), 180, 140), math.pi * 0.9, math.pi * 2.1, 6)
        for k in range(5):
            pygame.draw.line(st, (230, 230, 240), (cx - 80 + k * 40, cy), (cx - 60 + k * 30, cy + 60), 2)
    say(st, "海底捞月，背后见！", (1120, 300), tt, 2000, 3000, size=24, tail="down-right")
    mtitle(st, sc, tt, 2800)


def _two_ghosts(sc, st, tt):
    me = sc.side
    hit = 2500
    door = pygame.Rect(0, 0, 187, 204)
    door.midbottom = (720, GROUND)
    shake = math.sin(tt / 30.0) * 4 if 900 < tt < hit else 0
    if tt > hit and sc.finale:
        pygame.draw.rect(st, (20, 10, 20), door, border_top_left_radius=90, border_top_right_radius=90)
        king_fate(sc, st, 720, tt, hit, R=60)
    else:
        pygame.draw.rect(st, (130, 36, 40), door.move(int(shake), 0), border_top_left_radius=90, border_top_right_radius=90)
        pygame.draw.line(st, INK, (720 + shake, door.y + 20), (720 + shake, GROUND), 4)
        if 1500 < tt:
            for sgn in (-1, 1):
                pygame.draw.circle(st, (255, 255, 255), (int(720 + sgn * 16 + shake), door.y + 90), 9)
                pygame.draw.circle(st, INK, (int(720 + sgn * 16 + shake), door.y + 92), 4)
        if tt > hit:
            say(st, "顶住门！", (720, door.y - 10), tt, hit, TIMELINE, size=26, tail="down-right")
    for k, x in enumerate((540, 900)):
        knock = ((tt - 900 - k * 250) % 500) / 500 if 900 < tt < hit else 0
        float_y = math.sin(tt / 200.0 + k) * 14
        layer = pygame.Surface((240, 140), pygame.SRCALPHA)
        pts = [(20, 0), (220, 0)] + [(220 - i * 25, 70 + (14 if i % 2 else 0) + math.sin(tt / 120.0 + i) * 5) for i in range(9)]
        pygame.draw.polygon(layer, (240, 245, 255, 150), pts)
        st.blit(layer, (x - 120, GROUND - 90 + float_y))
        actor(st, PAWN, me, (x, GROUND - 40 + float_y), 72, t=tt, facing=1 if k == 0 else -1, expr="smug",
              swing=fx.hop(knock), alpha=235, shadow=False)
    for k in range(4):
        boom(st, "咚!", (720 + (-120 if k % 2 == 0 else 120), door.y - 30), tt, 900 + k * 400, 380, 64, (255, 240, 200))
    mtitle(st, sc, tt, 2800)


def _generic(sc, st, tt):
    me = sc.side
    kind = sc.event.actor.kind
    hit = 2200
    layer = pygame.Surface(st.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(layer, (255, 250, 220, 60), [(1000, 0), (1120, 0), (1300, GROUND + 40), (820, GROUND + 40)])
    st.blit(layer, (0, 0))
    king_fate(sc, st, 1060, tt, hit)
    x, _ = fx.lerp(-150, 760, fx.ease_in_out(fx.seg(tt, 300, 1700))), 0
    put(st, kind, me, x, tt, walk=tt / 60 if 300 < tt < 1700 else 0, expr="fierce", swing=fx.seg(tt, hit - 200, hit + 100))
    question_marks(st, (1060, GROUND - 260), tt, 255 * (1 - fx.seg(tt, hit - 200, hit)))
    say(st, "无路可走了。", (x + 20, GROUND - 260), tt, 1500, 2400, size=26)
    mtitle(st, sc, tt, 2600)


def _stalemate(sc, st, tt):
    foe = other(sc.side)
    close = fx.ease_out(fx.seg(tt, 400, 1600))
    put(st, KING, foe, 720, tt, expr="sad" if tt > 1600 else "scared", squash=0.2 * fx.seg(tt, 2000, 2400))
    for sgn in (-1, 1):
        x = 720 + sgn * (720 - 480 * close)
        wall = pygame.Rect(0, 80, 400, GROUND - 80)
        if sgn < 0:
            wall.right = int(x - 120)
        else:
            wall.x = int(x + 120)
        pygame.draw.rect(st, (150, 110, 100), wall)
        for r in range(10):
            pygame.draw.line(st, INK, (wall.x, wall.y + r * 48), (wall.right, wall.y + r * 48), 3)
        pygame.draw.rect(st, INK, wall, 4)
    iron_bars(st, pygame.Rect(560, 80, 320, 200), fx.seg(tt, 1200, 2000))
    question_marks(st, (720, GROUND - 260), tt)
    say(st, "没被将军……却一步也走不了……", (720, GROUND - 230), tt, 1800, 3400, size=24, tail="down-right")
    mtitle(st, sc, tt, 2800)


PAINTERS = {
    "double_rook": (_double_rook, "night"),
    "wocao_horse": (_wocao_horse, "palace_night"),
    "corner_horse": (_corner_horse, "palace_night"),
    "fishing_horse": (_fishing_horse, "riverbank"),
    "horse_cannon": (_horse_cannon, "fortress"),
    "double_cannon": (_double_cannon, "fortress"),
    "heaven_earth": (_heaven_earth, "sky"),
    "iron_bolt": (_iron_bolt, "palace_night"),
    "heart_strike": (_heart_strike, "throne"),
    "facing_kings": (_facing_kings, "ink"),
    "sea_moon": (_sea_moon, "sea"),
    "two_ghosts": (_two_ghosts, "palace_night"),
    "generic": (_generic, "night"),
    "stalemate": (_stalemate, "ink"),
}


def mate_scene(event):
    painter, backdrop = PAINTERS.get(event.key, PAINTERS["generic"])
    finale = event.kind in ("mate", "stalemate")
    duration = TIMELINE if finale else CHECK_DURATION
    if event.kind == "sacrifice":
        duration = 3000
    scale = TIMELINE / duration

    def paint(scene, stage, t):
        painter(scene, stage, t * scale)

    def setup(scene):
        scene.finale = finale

    return LambdaCutscene(event, paint, duration, backdrop, setup)
