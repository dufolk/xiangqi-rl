import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

pygame.init()

WIDTH, HEIGHT = 1540, 960
screen = pygame.display.set_mode((WIDTH, HEIGHT))
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

FONT = pygame.font.SysFont("PingFang SC", 18)
SMALL = pygame.font.SysFont("PingFang SC", 14)
TINY = pygame.font.SysFont("PingFang SC", 12)
TITLE = pygame.font.SysFont("PingFang SC", 29, bold=True)
CARD_FONT = pygame.font.SysFont("PingFang SC", 16, bold=True)


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
    def __init__(self):
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
        self.players = [
            Player("玩家 1", (113, 184, 255)), Player("玩家 2", (239, 129, 139)),
            Player("玩家 3", (108, 210, 157)), Player("玩家 4", (226, 176, 88)),
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
                cards.append(Card(tier, bonus, points, dict(zip(GEM_KEYS, rotated))))
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
        if self.player.points >= 15:
            self.round_end = True
        self.turn = (self.turn + 1) % 4
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


def gem_circle(center, gem, count, selected=False, small=False):
    radius = 20 if small else 27
    pygame.draw.circle(screen, (6, 10, 18), center, radius + 3)
    pygame.draw.circle(screen, GEM_COLOR[gem], center, radius)
    pygame.draw.circle(screen, (255, 255, 255), center, radius, 2)
    if selected:
        pygame.draw.circle(screen, ACCENT, center, radius + 7, 3)
    if count:
        label_color = (25, 32, 47) if gem in ("white", "gold") else TEXT
        text(str(count), center, FONT if not small else SMALL, label_color, "center")


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
    gem_circle((x + w - 25, y + 25), card.bonus, 0, small=True)
    text(card.points, (x + 17, y + 15), CARD_FONT, GOLD)
    text("点", (x + 35, y + 19), TINY, MUTED)
    draw_cost(card, x + 24, y + h - 26, compact=w < 145)
    if label:
        text(label, (x + w / 2, y + h / 2 - 5), SMALL, MUTED, "center")


def draw_noble(noble, rect, selectable=False):
    x, y, w, h = rect
    rounded(rect, (74, 52, 91), 13, GOLD if selectable else (147, 105, 177), 3 if selectable else 1)
    text("贵族", (x + w / 2, y + 18), SMALL, (247, 222, 154), "center")
    for i, (g, n) in enumerate(noble.items()):
        gem_circle((x + 24 + i * 34, y + h - 25), g, n, small=True)


def draw_button(rect, label, active=True, color=ACCENT):
    rounded(rect, color if active else (58, 68, 83), 10)
    text(label, rect.center, SMALL, (15, 21, 33) if active else MUTED, "center")


def draw_player_panel(player, rect, active):
    x, y, w, h = rect
    rounded(rect, PANEL_2 if active else PANEL, 13, player.color if active else None, 3)
    pygame.draw.circle(screen, player.color, (x + 21, y + 22), 7)
    text(player.name, (x + 35, y + 12), SMALL, TEXT)
    text(f"{player.points} 分", (x + w - 15, y + 12), SMALL, GOLD, "topright")
    labels = "".join(str(player.gems[g]) for g in GEM_KEYS) + f" +{player.gems['gold']}"
    text(f"宝石  {labels}", (x + 14, y + 41), TINY, MUTED)
    text(f"发展卡 {len(player.cards)}   保留 {len(player.reserved)}/3   贵族 {len(player.nobles)}", (x + 14, y + 59), TINY, MUTED)
    by_color = [(g, player.discount(g)) for g in GEM_KEYS if player.discount(g)]
    for i, (g, amount) in enumerate(by_color):
        gem_circle((x + 18 + i * 28, y + h - 14), g, amount, small=True)


def table_hitboxes():
    hits = []
    for tier, y in zip((3, 2, 1), (192, 342, 492)):
        for i in range(4):
            hits.append((pygame.Rect(405 + i * 170, y, 150, 125), ("table", tier, i)))
    return hits


def render():
    screen.fill(BG)
    pygame.draw.circle(screen, (29, 47, 72), (WIDTH - 100, -10), 270)
    pygame.draw.circle(screen, (37, 54, 78), (170, HEIGHT + 80), 330)
    text("SPLENDOR", (38, 27), TITLE, GOLD)
    text("璀璨宝石 · 四人标准局", (38, 62), SMALL, MUTED)
    text(f"第 {sum(len(p.cards) + len(p.reserved) for p in game.players) + 1} 回合", (355, 31), SMALL, MUTED)
    text("达成 15 分后完成本轮，分高者胜；平局时发展卡少者胜。", (355, 59), TINY, MUTED)

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

    rounded((405, 645, 675, 122), PANEL, 14)
    text("贵族", (425, 662), SMALL, TEXT)
    for i, noble in enumerate(game.nobles):
        draw_noble(noble, pygame.Rect(500 + i * 112, 651, 98, 98), game.phase == "noble" and noble in game.noble_choices)

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

    if game.phase == "over" and game.winner:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 8, 14, 170))
        screen.blit(overlay, (0, 0))
        rounded((WIDTH // 2 - 280, HEIGHT // 2 - 120, 560, 240), (43, 57, 79), 22, GOLD, 3)
        text("本局结束", (WIDTH // 2, HEIGHT // 2 - 75), TITLE, GOLD, "center")
        text(f"{game.winner.name} 获胜！", (WIDTH // 2, HEIGHT // 2 - 22), TITLE, TEXT, "center")
        text(f"得分 {game.winner.points} · 发展卡 {len(game.winner.cards)} · 贵族 {len(game.winner.nobles)}", (WIDTH // 2, HEIGHT // 2 + 22), SMALL, MUTED, "center")
        text("按 R 键开始新的一局", (WIDTH // 2, HEIGHT // 2 + 68), SMALL, ACCENT, "center")


def click(pos):
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
            if pygame.Rect(500 + i * 112, 651, 98, 98).collidepoint(pos) and noble in game.noble_choices:
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


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            game.restart()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            click(event.pos)
    render()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
