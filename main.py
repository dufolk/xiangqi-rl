import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# 让 macOS Retina 屏幕使用原生像素密度，避免 pygame 界面被系统放大后发虚。
os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")

import pygame

from ai_player import GenerativeAIPlayer
from network import LobbyConfig, create_lobby_gateway

pygame.init()

WIDTH, HEIGHT = 1540, 960
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
pygame.display.set_caption("璀璨宝石 · 四人标准局")
clock = pygame.time.Clock()

BG = (18, 25, 39)
PANEL = (28, 39, 57)
PANEL_2 = (35, 49, 70)
TEXT = (238, 242, 250)
MUTED = (153, 170, 193)
GOLD = (244, 197, 66)
ACCENT = (116, 185, 255)
RED = (218, 84, 92)
GREEN = (94, 203, 145)
COLOR_NAMES = ["钻石", "蓝宝石", "祖母绿", "红宝石", "黑玛瑙"]
COLORS = [(232, 236, 244), (71, 142, 232), (61, 177, 129), (215, 78, 89), (58, 66, 83)]
GEM_KEYS = ["white", "blue", "green", "red", "black"]
ALL_GEMS = GEM_KEYS + ["gold"]
GEM_COLOR = dict(zip(GEM_KEYS, COLORS))
GEM_COLOR["gold"] = GOLD
TIER_COLORS = [(112, 182, 236), (160, 117, 229), (241, 174, 80)]

def _make_font(size, bold=False):
    # 直接加载系统中的中文字体文件，避免 SysFont 找不到 PingFang 时
    # 回退到不支持中文的位图字体导致乱码。
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            f = pygame.font.Font(path, size)
            f.set_bold(bold)
            return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size)


FONT = _make_font(20)
SMALL = _make_font(16)
TINY = _make_font(13)
TITLE = _make_font(32, bold=True)
CARD_FONT = _make_font(17, bold=True)

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_FILES = {
    "gems": "Game_UI_asset_sheet__exactly_s_2026-08-25T08-16-27.png",
    "nobles": "Premium_Renaissance_fantasy_bo_2026-08-25T08-16-28.png",
    "tier1": "Tier_one_Renaissance_gem_tradi_2026-08-25T08-16-26.png",
    "tier2": "Tier_two_Renaissance_gem_tradi_2026-08-25T08-16-30.png",
    "tier3": "Tier_three_Renaissance_gem_tra_2026-08-25T08-16-26.png",
}
ASSETS: Dict[str, pygame.Surface] = {}
for key, filename in ASSET_FILES.items():
    path = os.path.join(ASSET_DIR, filename)
    if os.path.exists(path):
        ASSETS[key] = pygame.image.load(path).convert()

GEM_CELLS = {
    "white": (0, 0), "blue": (1, 0), "green": (2, 0),
    "red": (0, 1), "black": (1, 1), "gold": (2, 1),
}


def sheet_cell(sheet_key, col, row, columns, rows, size, fit="cover"):
    sheet = ASSETS.get(sheet_key)
    if not sheet:
        return None
    source_w, source_h = sheet.get_size()
    cell_w, cell_h = source_w // columns, source_h // rows
    inset = 5
    source = pygame.Rect(col * cell_w + inset, row * cell_h + inset, cell_w - inset * 2, cell_h - inset * 2)
    target_w, target_h = size

    if fit == "contain":
        scale = min(target_w / source.width, target_h / source.height)
        fitted_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        fitted = pygame.transform.smoothscale(sheet.subsurface(source), fitted_size)
        result = pygame.Surface(size)
        result.fill((38, 29, 49))
        result.blit(fitted, fitted.get_rect(center=(target_w // 2, target_h // 2)))
        return result

    # 场景插画采用等比居中裁切，禁止直接拉伸。
    target_ratio = target_w / target_h
    source_ratio = source.width / source.height
    if source_ratio < target_ratio:
        crop_h = round(source.width / target_ratio)
        source.y += (source.height - crop_h) // 2
        source.height = crop_h
    elif source_ratio > target_ratio:
        crop_w = round(source.height * target_ratio)
        source.x += (source.width - crop_w) // 2
        source.width = crop_w
    return pygame.transform.smoothscale(sheet.subsurface(source), size)


def card_art(card, size):
    index = (card.art_index + GEM_KEYS.index(card.bonus) * 2) % 12
    return sheet_cell(f"tier{card.tier}", index % 4, index // 4, 4, 3, size)


def gem_art(gem, size):
    col, row = GEM_CELLS[gem]
    art = sheet_cell("gems", col, row, 3, 2, size)
    if not art:
        return None

    # 原始 AI 图集带有不透明背景；用圆形 Alpha 蒙版只保留宝石徽章。
    art = art.convert_alpha()
    mask = pygame.Surface(size, pygame.SRCALPHA)
    center = (size[0] // 2, size[1] // 2)
    pygame.draw.circle(mask, (255, 255, 255, 255), center, min(size) // 2 - 1)
    art.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return art


def noble_art(index, size):
    return sheet_cell("nobles", index % 4, (index // 4) % 2, 4, 2, size, fit="contain")


def text(s, pos, font=FONT, color=TEXT, anchor="topleft"):
    img = font.render(str(s), True, color)
    rect = img.get_rect()
    setattr(rect, anchor, pos)
    screen.blit(img, rect)
    return rect


def rounded(rect, color, radius=12, border=None, width=2):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(screen, border, rect, width, border_radius=radius)


@dataclass
class Card:
    tier: int
    bonus: str
    points: int
    cost: Dict[str, int]
    art_index: int = 0


@dataclass
class Player:
    name: str
    color: Tuple[int, int, int]
    gems: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in ALL_GEMS})
    cards: List[Card] = field(default_factory=list)
    reserved: List[Card] = field(default_factory=list)
    nobles: List[Dict[str, int]] = field(default_factory=list)

    def discount(self, gem):
        return sum(c.bonus == gem for c in self.cards)

    @property
    def points(self):
        return sum(c.points for c in self.cards) + 3 * len(self.nobles)

    @property
    def gem_count(self):
        return sum(self.gems.values())


class Game:
    def __init__(self, ai_count=3, human_seat=0, target_score=15, player_name="你"):
        self.ai_count = max(0, min(3, ai_count))
        self.human_seat = max(0, min(3, human_seat))
        self.target_score = max(5, min(30, target_score))
        self.player_name = player_name.strip() or "你"
        self.restart()

    def restart(self):
        self.bank = {k: 7 for k in GEM_KEYS} | {"gold": 5}
        self.decks = {tier: self.make_deck(tier) for tier in (1, 2, 3)}
        self.table = {tier: [] for tier in (1, 2, 3)}
        for tier in self.table:
            for _ in range(4):
                self.table[tier].append(self.decks[tier].pop())
        noble_templates = [
            {"white": 4, "blue": 4, "green": 4}, {"blue": 4, "green": 4, "red": 4},
            {"green": 4, "red": 4, "black": 4}, {"red": 4, "black": 4, "white": 4},
            {"black": 4, "white": 4, "blue": 4}, {"white": 3, "blue": 3, "green": 3},
            {"blue": 3, "green": 3, "red": 3}, {"green": 3, "red": 3, "black": 3},
            {"red": 3, "black": 3, "white": 3}, {"black": 3, "white": 3, "blue": 3},
        ]
        self.nobles = random.sample(noble_templates, 5)
        colors = [(113, 184, 255), (239, 129, 139), (108, 210, 157), (226, 176, 88)]
        ai_seats = [i for i in range(4) if i != self.human_seat][:self.ai_count]
        self.ai_seats = set(ai_seats)
        self.players = [
            Player(self.player_name if i == self.human_seat else (f"AI {i + 1}" if i in self.ai_seats else f"玩家 {i + 1}"), colors[i])
            for i in range(4)
        ]
        self.turn = 0
        self.selected_card: Optional[Tuple[str, int, int]] = None
        self.selected_gems: Dict[str, int] = {}
        self.phase = "action"
        self.noble_choices: List[Dict[str, int]] = []
        self.round_end = False
        self.winner: Optional[Player] = None
        self.message = "轮到玩家 1：选择宝石，或选中一张发展卡。"

    def make_deck(self, tier):
        cards = []
        templates = {
            1: [([0, 0, 0, 0, 3], 0), ([0, 0, 0, 1, 2], 0), ([0, 0, 1, 1, 1], 0),
                ([0, 0, 0, 2, 2], 0), ([0, 0, 0, 0, 4], 1), ([0, 0, 0, 1, 3], 0),
                ([0, 0, 2, 0, 2], 0), ([0, 1, 1, 1, 1], 1)],
            2: [([0, 0, 2, 3, 0], 1), ([0, 0, 3, 2, 2], 2), ([0, 1, 0, 4, 2], 2),
                ([0, 0, 5, 0, 0], 2), ([0, 2, 2, 3, 0], 3), ([0, 0, 4, 1, 2], 2),
                ([0, 3, 0, 3, 2], 1), ([0, 2, 3, 0, 3], 2)],
            3: [([0, 3, 3, 3, 5], 3), ([0, 0, 0, 7, 0], 4), ([0, 3, 0, 6, 3], 4),
                ([0, 0, 3, 3, 6], 4), ([0, 3, 5, 3, 3], 5), ([0, 0, 0, 3, 7], 5),
                ([0, 3, 3, 5, 3], 4), ([0, 5, 3, 3, 3], 4)],
        }[tier]
        for bonus_idx, bonus in enumerate(GEM_KEYS):
            for template, points in templates:
                rotated = template[-bonus_idx:] + template[:-bonus_idx] if bonus_idx else template[:]
                cards.append(Card(tier, bonus, points, dict(zip(GEM_KEYS, rotated)), len(cards) % 12))
        random.shuffle(cards)
        return cards

    @property
    def player(self):
        return self.players[self.turn]

    def card_from_ref(self, ref):
        where, tier, index = ref
        return self.table[tier][index] if where == "table" else self.player.reserved[index]

    def can_afford(self, card):
        missing = sum(max(0, card.cost[g] - self.player.discount(g) - self.player.gems[g]) for g in GEM_KEYS)
        return missing <= self.player.gems["gold"]

    def take_gem(self, gem):
        if self.phase == "return":
            if self.player.gems[gem] <= 0:
                return
            self.player.gems[gem] -= 1
            self.bank[gem] += 1
            self.message = f"请归还宝石，手中还有 {self.player.gem_count} 枚（上限 10）。"
            if self.player.gem_count <= 10:
                self.finish_turn()
            return
        if self.phase != "action" or gem == "gold" or self.bank[gem] <= 0:
            return
        if gem in self.selected_gems:
            if self.selected_gems[gem] == 1 and len(self.selected_gems) == 1 and self.bank[gem] >= 4:
                self.selected_gems[gem] = 2
            else:
                self.selected_gems.pop(gem)
        elif len(self.selected_gems) < 3:
            self.selected_gems[gem] = 1
        self.message = "已选 " + ("、".join(f"{COLOR_NAMES[GEM_KEYS.index(k)]} ×{v}" for k, v in self.selected_gems.items()) or "0 枚")

    def confirm_gems(self):
        if self.phase != "action" or not self.selected_gems:
            self.message = "请先选择宝石。"
            return
        total = sum(self.selected_gems.values())
        same_two = total == 2 and len(self.selected_gems) == 1
        if total not in (1, 2, 3) or (total == 2 and not same_two):
            self.message = "规则：拿 3 枚不同宝石，或在库存不少于 4 时拿 2 枚同色宝石。"
            return
        if total == 1:
            self.message = "请拿满 3 枚不同宝石，或拿 2 枚同色宝石。"
            return
        if any(self.bank[g] < amount for g, amount in self.selected_gems.items()):
            self.message = "宝石库存不足。"
            return
        for g, amount in self.selected_gems.items():
            self.bank[g] -= amount
            self.player.gems[g] += amount
        self.selected_gems = {}
        self.after_action()

    def reserve_selected(self):
        if self.phase != "action" or not self.selected_card:
            self.message = "请先选择桌面上的发展卡。"
            return
        if len(self.player.reserved) >= 3:
            self.message = "每位玩家最多保留 3 张发展卡。"
            return
        where, tier, index = self.selected_card
        if where != "table":
            self.message = "保留只适用于桌面发展卡。"
            return
        card = self.table[tier].pop(index)
        self.player.reserved.append(card)
        if self.decks[tier]:
            self.table[tier].insert(index, self.decks[tier].pop())
        if self.bank["gold"]:
            self.bank["gold"] -= 1
            self.player.gems["gold"] += 1
        self.selected_card = None
        self.after_action()

    def buy_selected(self):
        if self.phase != "action" or not self.selected_card:
            self.message = "请先选中一张桌面或保留的发展卡。"
            return
        card = self.card_from_ref(self.selected_card)
        if not self.can_afford(card):
            self.message = "宝石和折扣不足，暂时买不起这张卡。"
            return
        for g in GEM_KEYS:
            need = max(0, card.cost[g] - self.player.discount(g))
            paid = min(need, self.player.gems[g])
            self.player.gems[g] -= paid
            self.bank[g] += paid
            need -= paid
            if need:
                self.player.gems["gold"] -= need
                self.bank["gold"] += need
        where, tier, index = self.selected_card
        if where == "table":
            self.table[tier].pop(index)
            if self.decks[tier]:
                self.table[tier].insert(index, self.decks[tier].pop())
        else:
            self.player.reserved.pop(index)
        self.player.cards.append(card)
        self.selected_card = None
        self.check_nobles()

    def check_nobles(self):
        eligible = [n for n in self.nobles if all(self.player.discount(g) >= n.get(g, 0) for g in GEM_KEYS)]
        if eligible:
            self.phase = "noble"
            self.noble_choices = eligible
            self.message = "你满足了贵族条件：请选择一位贵族拜访。"
        else:
            self.after_action()

    def choose_noble(self, noble):
        if self.phase != "noble":
            return
        self.player.nobles.append(noble)
        self.nobles.remove(noble)
        self.noble_choices = []
        self.after_action()

    def after_action(self):
        if self.player.gem_count > 10:
            self.phase = "return"
            self.message = f"手中有 {self.player.gem_count} 枚宝石，请点击宝石归还至 10 枚。"
        else:
            self.finish_turn()

    def finish_turn(self):
        if self.player.points >= self.target_score:
            self.round_end = True
        self.turn = (self.turn + 1) % len(self.players)
        if self.turn == 0 and self.round_end:
            best = max(p.points for p in self.players)
            candidates = [p for p in self.players if p.points == best]
            self.winner = min(candidates, key=lambda p: len(p.cards))
            self.phase = "over"
            self.message = f"游戏结束：{self.winner.name} 获胜！"
            return
        self.phase = "action"
        self.selected_gems = {}
        self.selected_card = None
        self.message = f"轮到 {self.player.name}：选择宝石，或选中一张发展卡。"


game = Game()
ai_player = GenerativeAIPlayer()
lobby_gateway = create_lobby_gateway()
scene = "setup"
setup_options = {"player_name": "玩家", "ai_count": 3, "human_seat": 0, "target_score": 15}
name_focused = False
lobby_info = None
ai_due_at = 0
ai_executor = ThreadPoolExecutor(max_workers=1)
ai_future: Optional[Future] = None
ai_request_turn = -1


def gem_circle(center, gem, count, selected=False, small=False, radius=None, label_font=None):
    radius = radius if radius is not None else (20 if small else 27)
    art = gem_art(gem, (radius * 2 + 6, radius * 2 + 6))
    if art:
        screen.blit(art, art.get_rect(center=center))
    else:
        pygame.draw.circle(screen, (6, 10, 18), center, radius + 3)
        pygame.draw.circle(screen, GEM_COLOR[gem], center, radius)
        pygame.draw.circle(screen, (255, 255, 255), center, radius, 2)
    if selected:
        pygame.draw.circle(screen, ACCENT, center, radius + 7, 3)
    if count is not None:
        badge_center = (center[0] + radius - 2, center[1] + radius - 2)
        badge_radius = max(9, radius // 2 + 3)
        # 所有宝石使用统一的深色数字底座；数量在贴图之后绘制，保证位于最上层。
        pygame.draw.circle(screen, (10, 15, 24), badge_center, badge_radius)
        pygame.draw.circle(screen, (7, 11, 19), badge_center, badge_radius, 2)
        text(str(count), badge_center, label_font or TINY, TEXT, "center")


def noble_rect(index):
    return pygame.Rect(500 + index * 112, 651, 98, 128)


def draw_cost(card, x, y, compact=False):
    shown = [(g, n) for g, n in card.cost.items() if n]
    spacing = 28 if compact else 31
    for i, (g, n) in enumerate(shown):
        cx = x + i * spacing
        pygame.draw.circle(screen, GEM_COLOR[g], (cx, y), 10 if compact else 12)
        text(n, (cx, y), TINY if compact else SMALL, (18, 25, 39) if g == "white" else TEXT, "center")


def draw_card(card, rect, selected=False, inactive=False, label=""):
    x, y, w, h = rect
    base = (42, 56, 78) if not inactive else (31, 40, 55)
    rounded(rect, base, 12, ACCENT if selected else TIER_COLORS[card.tier - 1], 3 if selected else 1)
    pygame.draw.rect(screen, TIER_COLORS[card.tier - 1], (x, y, 8, h), border_top_left_radius=12, border_bottom_left_radius=12)

    art_h = max(16, h - 55)
    art_w = max(28, w - 26)
    art = card_art(card, (art_w, art_h))
    if art:
        art_rect = pygame.Rect(x + 13, y + 37, art_w, art_h)
        screen.blit(art, art_rect)
        shade = pygame.Surface(art_rect.size, pygame.SRCALPHA)
        shade.fill((6, 10, 18, 38 if not inactive else 120))
        screen.blit(shade, art_rect)
        pygame.draw.rect(screen, TIER_COLORS[card.tier - 1], art_rect, 1, border_radius=5)

    token_radius = 13 if w < 145 else 17
    gem_circle((x + w - token_radius - 9, y + token_radius + 9), card.bonus, 0, radius=token_radius)
    text(card.points, (x + 17, y + 13), CARD_FONT, GOLD)
    text("点", (x + 39, y + 18), TINY, MUTED)
    draw_cost(card, x + 24, y + h - 20, compact=w < 145)
    if label:
        text(label, (x + w / 2, y + h / 2 - 5), SMALL, MUTED, "center")


def draw_noble(noble, rect, index=0, selectable=False):
    x, y, w, h = rect
    rounded(rect, (74, 52, 91), 13, GOLD if selectable else (147, 105, 177), 3 if selectable else 1)
    portrait = noble_art(index, (w - 8, h - 38))
    if portrait:
        screen.blit(portrait, (x + 4, y + 4))
        veil = pygame.Surface((w - 8, 24), pygame.SRCALPHA)
        veil.fill((24, 16, 35, 170))
        screen.blit(veil, (x + 4, y + h - 30))
    for i, (g, n) in enumerate(noble.items()):
        gem_circle((x + 17 + i * 32, y + h - 17), g, n, radius=11, label_font=TINY)


def draw_button(rect, label, active=True, color=ACCENT):
    rounded(rect, color if active else (58, 68, 83), 10)
    text(label, rect.center, SMALL, (15, 21, 33) if active else MUTED, "center")


def draw_player_panel(player, rect, active):
    x, y, w, h = rect
    rounded(rect, PANEL_2 if active else PANEL, 13, player.color if active else None, 3)
    pygame.draw.circle(screen, player.color, (x + 20, y + 20), 7)
    text(player.name, (x + 34, y + 10), SMALL, TEXT)
    text(f"{player.points} 分", (x + w - 15, y + 10), SMALL, GOLD, "topright")

    text("持有宝石", (x + 14, y + 34), TINY, MUTED)
    for i, gem in enumerate(ALL_GEMS):
        gem_circle((x + 91 + i * 38, y + 41), gem, player.gems[gem], radius=13, label_font=TINY)

    text(f"发展卡 {len(player.cards)}   保留 {len(player.reserved)}/3   贵族 {len(player.nobles)}", (x + 14, y + 62), TINY, MUTED)
    by_color = [(g, player.discount(g)) for g in GEM_KEYS if player.discount(g)]
    if by_color:
        text("折扣", (x + 14, y + 82), TINY, MUTED)
        for i, (g, amount) in enumerate(by_color):
            gem_circle((x + 62 + i * 31, y + 83), g, amount, radius=11, label_font=TINY)


def table_hitboxes():
    hits = []
    for tier, y in zip((3, 2, 1), (192, 342, 492)):
        for i in range(len(game.table[tier])):
            hits.append((pygame.Rect(405 + i * 170, y, 150, 125), ("table", tier, i)))
    return hits


def render_setup():
    screen.fill(BG)
    pygame.draw.circle(screen, (40, 65, 96), (1260, 100), 430)
    pygame.draw.circle(screen, (46, 39, 72), (180, 900), 400)
    text("SPLENDOR", (WIDTH // 2, 92), TITLE, GOLD, "center")
    text("璀璨宝石 · 新对局", (WIDTH // 2, 138), FONT, TEXT, "center")
    text("选择本局设置，然后与生成式 AI 或本地玩家同桌竞技", (WIDTH // 2, 178), SMALL, MUTED, "center")

    panel = pygame.Rect(WIDTH // 2 - 380, 225, 760, 520)
    rounded(panel, PANEL, 24, (69, 91, 122), 2)
    text("玩家名称", (430, 259), FONT, TEXT)
    name_rect = pygame.Rect(760, 247, 254, 48)
    rounded(name_rect, PANEL_2, 10, ACCENT if name_focused else (79, 98, 124), 2)
    text(setup_options["player_name"] or "输入名称", (778, 259), SMALL, TEXT if setup_options["player_name"] else MUTED)

    rows = [
        ("AI 玩家数量", "ai_count", 0, 3, "其余席位可供本地玩家热座"),
        ("我的位次", "human_seat", 0, 3, "决定你的行动顺序"),
        ("获胜分数", "target_score", 5, 30, "达到后完成本轮并结算"),
    ]
    for i, (label, key, minimum, maximum, hint) in enumerate(rows):
        y = 325 + i * 88
        text(label, (430, y), FONT, TEXT)
        text(hint, (430, y + 31), TINY, MUTED)
        draw_button(pygame.Rect(850, y - 4, 48, 44), "−", setup_options[key] > minimum, (103, 126, 158))
        shown = setup_options[key] + 1 if key == "human_seat" else setup_options[key]
        text(shown, (932, y + 17), FONT, GOLD, "center")
        draw_button(pygame.Rect(966, y - 4, 48, 44), "+", setup_options[key] < maximum, (103, 126, 158))

    model_state = "已启用外部生成式模型" if ai_player.model_enabled else "未配置外部模型，自动使用本地策略 AI"
    text(model_state, (WIDTH // 2, 610), SMALL, GREEN if ai_player.model_enabled else MUTED, "center")
    draw_button(pygame.Rect(510, 657, 250, 58), "开始本地对局", True, GREEN)
    draw_button(pygame.Rect(780, 657, 250, 58), "创建好友大厅", True, GOLD)
    if lobby_info:
        text(f"大厅邀请码：{lobby_info.invite_code} · {lobby_info.status}", (WIDTH // 2, 775), FONT, GOLD, "center")
        text("服务接口已预留；配置 SPLENDOR_LOBBY_ENDPOINT 后即可连接真实大厅服务。", (WIDTH // 2, 812), TINY, MUTED, "center")


def render():
    screen.fill(BG)
    pygame.draw.circle(screen, (29, 47, 72), (WIDTH - 100, -10), 270)
    pygame.draw.circle(screen, (37, 54, 78), (170, HEIGHT + 80), 330)
    text("SPLENDOR", (38, 27), TITLE, GOLD)
    text("璀璨宝石 · AI 对战", (38, 62), SMALL, MUTED)
    text(f"第 {sum(len(p.cards) + len(p.reserved) for p in game.players) + 1} 回合", (355, 31), SMALL, MUTED)
    text(f"达成 {game.target_score} 分后完成本轮，分高者胜；平局时发展卡少者胜。", (355, 59), TINY, MUTED)

    rounded((20, 110, 345, 560), PANEL, 18)
    text("宝石供应", (42, 132), FONT, TEXT)
    text("点击选择；同色再点一次可拿 2 枚（库存须 ≥4）", (42, 159), TINY, MUTED)
    for i, gem in enumerate(GEM_KEYS):
        y = 210 + i * 65
        gem_circle((68, y), gem, game.bank[gem], gem in game.selected_gems)
        text(COLOR_NAMES[i], (110, y - 11), SMALL)
        text("库存 ≥4 才可双拿" if game.bank[gem] >= 4 else "库存不足以双拿", (110, y + 10), TINY, MUTED)
    gem_circle((68, 540), "gold", game.bank["gold"])
    text("黄金（百搭，仅保留发展卡时获得）", (110, 529), SMALL)
    text("库存", (110, 550), TINY, MUTED)
    draw_button(pygame.Rect(42, 590, 145, 44), "确认拿取", game.phase == "action" and bool(game.selected_gems), GREEN)
    draw_button(pygame.Rect(200, 590, 145, 44), "清除选择", bool(game.selected_gems), (129, 148, 174))

    text("发展卡", (405, 132), FONT)
    for tier, y, name in zip((3, 2, 1), (192, 342, 492), ("III · 高阶", "II · 中阶", "I · 初阶")):
        text(name, (405, y - 25), TINY, TIER_COLORS[tier - 1])
        text(f"牌库 {len(game.decks[tier])}", (1032, y - 25), TINY, MUTED, "topright")
        for i, card in enumerate(game.table[tier]):
            rect = pygame.Rect(405 + i * 170, y, 150, 125)
            draw_card(card, rect, game.selected_card == ("table", tier, i))

    rounded((405, 645, 675, 140), PANEL, 14)
    text("贵族", (425, 662), SMALL, TEXT)
    for i, noble in enumerate(game.nobles):
        draw_noble(noble, noble_rect(i), i, game.phase == "noble" and noble in game.noble_choices)

    rounded((1100, 110, 420, 657), PANEL, 18)
    text("玩家状态", (1122, 132), FONT)
    for i, player in enumerate(game.players):
        draw_player_panel(player, pygame.Rect(1122, 168 + i * 122, 376, 100), i == game.turn)

    active = game.player
    text(f"{active.name} 的保留发展卡", (1122, 666), SMALL)
    if active.reserved:
        for i, card in enumerate(active.reserved):
            draw_card(card, pygame.Rect(1122 + i * 124, 692, 112, 62), game.selected_card == ("reserved", 0, i))
    else:
        text("尚未保留发展卡", (1122, 710), SMALL, MUTED)

    rounded((20, 800, 1500, 130), PANEL_2, 18)
    text("行动台", (42, 820), FONT)
    color = RED if game.phase == "return" else GOLD if game.phase == "noble" else TEXT
    text(game.message, (42, 850), SMALL, color)
    if game.phase == "return":
        text("归还模式：点击左侧任意持有的宝石，直到数量不超过 10。", (42, 880), SMALL, MUTED)
    elif game.phase == "noble":
        text("选择一张高亮的贵族卡；每张贵族卡价值 3 分。", (42, 880), SMALL, MUTED)
    elif game.phase == "over":
        text("按 R 键开始新的一局。", (42, 880), SMALL, MUTED)
    else:
        draw_button(pygame.Rect(1050, 830, 135, 52), "购买所选", bool(game.selected_card), GREEN)
        draw_button(pygame.Rect(1200, 830, 135, 52), "保留所选", bool(game.selected_card and game.selected_card[0] == "table"), GOLD)
        draw_button(pygame.Rect(1350, 830, 135, 52), "取消选牌", bool(game.selected_card), (129, 148, 174))

    text("Esc 返回对局设置", (1490, 78), TINY, MUTED, "topright")

    if game.phase == "over" and game.winner:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 8, 14, 170))
        screen.blit(overlay, (0, 0))
        rounded((WIDTH // 2 - 280, HEIGHT // 2 - 120, 560, 240), (43, 57, 79), 22, GOLD, 3)
        text("本局结束", (WIDTH // 2, HEIGHT // 2 - 75), TITLE, GOLD, "center")
        text(f"{game.winner.name} 获胜！", (WIDTH // 2, HEIGHT // 2 - 22), TITLE, TEXT, "center")
        text(f"得分 {game.winner.points} · 发展卡 {len(game.winner.cards)} · 贵族 {len(game.winner.nobles)}", (WIDTH // 2, HEIGHT // 2 + 22), SMALL, MUTED, "center")
        text("按 R 键开始新的一局", (WIDTH // 2, HEIGHT // 2 + 68), SMALL, ACCENT, "center")


def ai_legal_actions():
    player = game.player
    if game.phase == "noble":
        return [{"id": f"noble:{game.nobles.index(noble)}", "type": "noble"} for noble in game.noble_choices]
    if game.phase == "return":
        return [
            {"id": f"return:{gem}", "type": "return", "usefulness": player.discount(gem) if gem != "gold" else 5}
            for gem in ALL_GEMS if player.gems[gem] > 0
        ]
    actions = []
    for tier in (1, 2, 3):
        for index, card in enumerate(game.table[tier]):
            if game.can_afford(card):
                actions.append({"id": f"buy:table:{tier}:{index}", "type": "buy", "tier": tier, "points": card.points})
            if len(player.reserved) < 3:
                actions.append({"id": f"reserve:{tier}:{index}", "type": "reserve", "tier": tier, "points": card.points})
    for index, card in enumerate(player.reserved):
        if game.can_afford(card):
            actions.append({"id": f"buy:reserved:0:{index}", "type": "buy", "tier": card.tier, "points": card.points})
    available = [gem for gem in GEM_KEYS if game.bank[gem] > 0]
    wanted = {gem: 0 for gem in GEM_KEYS}
    for tier in game.table:
        for card in game.table[tier]:
            for gem in GEM_KEYS:
                wanted[gem] += max(0, card.cost[gem] - player.discount(gem) - player.gems[gem])
    for combo in combinations(available, min(3, len(available))):
        if len(combo) >= 2:
            actions.append({"id": "take:" + ",".join(combo), "type": "take", "usefulness": sum(wanted[g] for g in combo)})
    for gem in GEM_KEYS:
        if game.bank[gem] >= 4:
            actions.append({"id": f"take:{gem},{gem}", "type": "take", "usefulness": wanted[gem] * 2})
    return actions


def ai_snapshot():
    return {
        "target_score": game.target_score,
        "bank": game.bank,
        "current_player": {
            "seat": game.turn,
            "score": game.player.points,
            "gems": game.player.gems,
            "discounts": {g: game.player.discount(g) for g in GEM_KEYS},
            "reserved": len(game.player.reserved),
        },
        "opponents": [{"score": p.points, "cards": len(p.cards)} for p in game.players if p is not game.player],
    }


def apply_ai_action(action_id):
    parts = action_id.split(":")
    if parts[0] == "noble":
        index = int(parts[1])
        if 0 <= index < len(game.nobles) and game.nobles[index] in game.noble_choices:
            game.choose_noble(game.nobles[index])
    elif parts[0] == "return":
        game.take_gem(parts[1])
    elif parts[0] == "buy":
        game.selected_card = (parts[1], int(parts[2]), int(parts[3]))
        game.buy_selected()
    elif parts[0] == "reserve":
        game.selected_card = ("table", int(parts[1]), int(parts[2]))
        game.reserve_selected()
    elif parts[0] == "take":
        gems = parts[1].split(",")
        game.selected_gems = {gem: gems.count(gem) for gem in set(gems)}
        game.confirm_gems()


def update_ai():
    global ai_due_at, ai_future, ai_request_turn
    if scene != "game" or game.phase == "over" or game.turn not in game.ai_seats:
        return
    now = pygame.time.get_ticks()
    if ai_due_at == 0:
        ai_due_at = now + 650
        game.message = f"{game.player.name} 正在思考…"
        return
    if now < ai_due_at:
        return
    actions = ai_legal_actions()
    if not actions:
        game.finish_turn()
        ai_due_at = 0
        return
    if ai_future is None:
        ai_request_turn = game.turn
        ai_future = ai_executor.submit(ai_player.choose_action, ai_snapshot(), actions)
        return
    if not ai_future.done():
        return
    action_id = ai_future.result()
    ai_future = None
    if ai_request_turn == game.turn and action_id:
        apply_ai_action(action_id)
        if game.turn in game.ai_seats and game.phase != "over":
            game.message = f"{game.player.name} 正在思考…（{ai_player.last_mode}）"
    ai_due_at = 0


def click_setup(pos):
    global game, scene, lobby_info, ai_due_at, name_focused
    name_focused = pygame.Rect(760, 247, 254, 48).collidepoint(pos)
    if name_focused:
        return
    keys = [("ai_count", 0, 3), ("human_seat", 0, 3), ("target_score", 5, 30)]
    for i, (key, minimum, maximum) in enumerate(keys):
        y = 325 + i * 88
        if pygame.Rect(850, y - 4, 48, 44).collidepoint(pos):
            setup_options[key] = max(minimum, setup_options[key] - 1)
            return
        if pygame.Rect(966, y - 4, 48, 44).collidepoint(pos):
            setup_options[key] = min(maximum, setup_options[key] + 1)
            return
    config = LobbyConfig(setup_options["player_name"].strip() or "玩家", setup_options["ai_count"], setup_options["human_seat"], setup_options["target_score"])
    if pygame.Rect(510, 657, 250, 58).collidepoint(pos):
        game = Game(config.ai_count, config.human_seat, config.target_score, config.player_name)
        scene = "game"
        ai_due_at = 0
    elif pygame.Rect(780, 657, 250, 58).collidepoint(pos):
        lobby_info = lobby_gateway.create_lobby(config)


def click(pos):
    if scene == "setup":
        click_setup(pos)
        return
    if game.turn in game.ai_seats:
        return
    if game.phase == "over":
        return
    for i, gem in enumerate(GEM_KEYS):
        if (pygame.Vector2(pos) - pygame.Vector2(68, 210 + i * 65)).length() <= 34:
            game.take_gem(gem)
            return
    if (pygame.Vector2(pos) - pygame.Vector2(68, 540)).length() <= 34:
        game.take_gem("gold")
        return
    if game.phase == "noble":
        for i, noble in enumerate(game.nobles):
            if noble_rect(i).collidepoint(pos) and noble in game.noble_choices:
                game.choose_noble(noble)
                return
        return
    if game.phase != "action":
        return
    for rect, ref in table_hitboxes():
        if rect.collidepoint(pos):
            game.selected_card = ref if game.selected_card != ref else None
            game.selected_gems = {}
            game.message = "已选发展卡：可购买或保留。"
            return
    for i, card in enumerate(game.player.reserved):
        if pygame.Rect(1122 + i * 124, 692, 112, 62).collidepoint(pos):
            ref = ("reserved", 0, i)
            game.selected_card = ref if game.selected_card != ref else None
            game.selected_gems = {}
            game.message = "已选保留发展卡：可购买。"
            return
    if pygame.Rect(42, 590, 145, 44).collidepoint(pos):
        game.confirm_gems()
    elif pygame.Rect(200, 590, 145, 44).collidepoint(pos):
        game.selected_gems = {}
        game.message = "已清除宝石选择。"
    elif pygame.Rect(1050, 830, 135, 52).collidepoint(pos):
        game.buy_selected()
    elif pygame.Rect(1200, 830, 135, 52).collidepoint(pos):
        game.reserve_selected()
    elif pygame.Rect(1350, 830, 135, 52).collidepoint(pos):
        game.selected_card = None
        game.message = "已取消选牌。"


def main():
    global scene, ai_due_at, name_focused
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and scene == "setup" and name_focused:
                if event.key == pygame.K_BACKSPACE:
                    setup_options["player_name"] = setup_options["player_name"][:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    name_focused = False
                elif event.unicode and event.unicode.isprintable() and len(setup_options["player_name"]) < 12:
                    setup_options["player_name"] += event.unicode
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and scene == "game":
                scene = "setup"
                ai_due_at = 0
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and scene == "game":
                game.restart()
                ai_due_at = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click(event.pos)
        if scene == "setup":
            render_setup()
        else:
            update_ai()
            render()
        pygame.display.flip()
        clock.tick(60)
    ai_executor.shutdown(wait=False, cancel_futures=True)
    pygame.quit()


if __name__ == "__main__":
    main()
