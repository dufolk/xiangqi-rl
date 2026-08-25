import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from card_data import STANDARD_CARD_ROWS

GEM_KEYS = ["white", "blue", "green", "red", "black"]
ALL_GEMS = GEM_KEYS + ["gold"]
COLOR_NAMES = ["钻石", "蓝宝石", "祖母绿", "红宝石", "黑玛瑙"]


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
    gems: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in ALL_GEMS})
    cards: List[Card] = field(default_factory=list)
    reserved: List[Optional[Card]] = field(default_factory=list)
    nobles: List[Dict[str, int]] = field(default_factory=list)

    def discount(self, gem: str) -> int:
        return sum(card is not None and card.bonus == gem for card in self.cards)

    @property
    def points(self) -> int:
        return sum(card.points for card in self.cards if card is not None) + 3 * len(self.nobles)

    @property
    def gem_count(self) -> int:
        return sum(self.gems.values())


class Game:
    def __init__(self, ai_count=3, human_seat=0, target_score=15, player_name="你", team_mode=False, rng=None):
        self.ai_count = max(0, min(3, ai_count))
        self.human_seat = max(0, min(3, human_seat))
        self.team_mode = bool(team_mode)
        self.target_score = 30 if self.team_mode else max(5, min(30, target_score))
        self.player_name = player_name.strip() or "你"
        self.rng = rng or random.Random()
        self.restart()

    def restart(self):
        self.bank = {key: 7 for key in GEM_KEYS} | {"gold": 5}
        self.decks = {tier: self.make_deck(tier) for tier in (1, 2, 3)}
        self.table = {tier: [] for tier in (1, 2, 3)}
        for tier in self.table:
            for _ in range(4):
                self.table[tier].append(self.decks[tier].pop())
        noble_templates = [
            {"white": 4, "blue": 4}, {"blue": 4, "green": 4},
            {"green": 4, "red": 4}, {"red": 4, "black": 4},
            {"black": 4, "white": 4}, {"white": 3, "blue": 3, "green": 3},
            {"blue": 3, "green": 3, "red": 3}, {"green": 3, "red": 3, "black": 3},
            {"red": 3, "black": 3, "white": 3}, {"black": 3, "white": 3, "blue": 3},
        ]
        assert all(
            (len(noble) == 2 and set(noble.values()) == {4})
            or (len(noble) == 3 and set(noble.values()) == {3})
            for noble in noble_templates
        )
        self.nobles = self.rng.sample(noble_templates, 5)
        colors = [(113, 184, 255), (239, 129, 139), (108, 210, 157), (226, 176, 88)]
        ai_seats = [seat for seat in range(4) if seat != self.human_seat][:self.ai_count]
        self.ai_seats = set(ai_seats)
        self.players = [
            Player(self.player_name if seat == self.human_seat else (f"AI {seat + 1}" if seat in self.ai_seats else f"玩家 {seat + 1}"), colors[seat])
            for seat in range(4)
        ]
        self.turn = 0
        self.selected_card: Optional[Tuple[str, int, int]] = None
        self.selected_gems: Dict[str, int] = {}
        self.phase = "action"
        self.noble_choices: List[Dict[str, int]] = []
        self.round_end = False
        self.winner: Optional[Player] = None
        self.winner_team: Optional[int] = None
        self.message = "轮到玩家 1：选择宝石，或选中一张发展卡。"

    def make_deck(self, tier):
        rows = [row for row in STANDARD_CARD_ROWS if row["tier"] == tier]
        cards = [
            Card(
                tier=row["tier"], bonus=row["bonus"], points=row["points"],
                cost={gem: row["cost"].get(gem, 0) for gem in GEM_KEYS}, art_index=index % 12,
            )
            for index, row in enumerate(rows)
        ]
        assert len(cards) == {1: 40, 2: 30, 3: 20}[tier]
        self.rng.shuffle(cards)
        return cards

    @property
    def player(self):
        return self.players[self.turn]

    def card_from_ref(self, ref):
        where, tier, index = ref
        card = self.table[tier][index] if where == "table" else self.player.reserved[index]
        if card is None:
            raise ValueError("hidden card cannot be selected")
        return card

    def can_afford(self, card):
        missing = sum(max(0, card.cost[gem] - self.player.discount(gem) - self.player.gems[gem]) for gem in GEM_KEYS)
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
        self.message = "已选 " + ("、".join(f"{COLOR_NAMES[GEM_KEYS.index(key)]} ×{value}" for key, value in self.selected_gems.items()) or "0 枚")

    def confirm_gems(self):
        if self.phase != "action" or not self.selected_gems:
            self.message = "请先选择宝石。"
            return
        total = sum(self.selected_gems.values())
        available_colors = sum(self.bank[gem] > 0 for gem in GEM_KEYS)
        distinct_take = all(amount == 1 for amount in self.selected_gems.values())
        valid_distinct = distinct_take and total == min(3, available_colors)
        valid_double = total == 2 and len(self.selected_gems) == 1 and self.bank[next(iter(self.selected_gems))] >= 4
        if not (valid_distinct or valid_double):
            self.message = "规则：尽量拿 3 种不同宝石；不足 3 种时可拿完现有颜色，或从库存 ≥4 的同色拿 2 枚。"
            return
        if any(self.bank[gem] < amount for gem, amount in self.selected_gems.items()):
            self.message = "宝石库存不足。"
            return
        for gem, amount in self.selected_gems.items():
            self.bank[gem] -= amount
            self.player.gems[gem] += amount
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
        self._grant_reserve_gold()
        self.selected_card = None
        self.after_action()

    def reserve_from_deck(self, tier):
        if self.phase != "action" or tier not in self.decks or not self.decks[tier]:
            return
        if len(self.player.reserved) >= 3:
            self.message = "每位玩家最多保留 3 张发展卡。"
            return
        self.player.reserved.append(self.decks[tier].pop())
        self._grant_reserve_gold()
        self.selected_card = None
        self.after_action()

    def _grant_reserve_gold(self):
        if self.bank["gold"]:
            self.bank["gold"] -= 1
            self.player.gems["gold"] += 1

    def buy_selected(self):
        if self.phase != "action" or not self.selected_card:
            self.message = "请先选中一张桌面或保留的发展卡。"
            return
        card = self.card_from_ref(self.selected_card)
        if not self.can_afford(card):
            self.message = "宝石和折扣不足，暂时买不起这张卡。"
            return
        for gem in GEM_KEYS:
            need = max(0, card.cost[gem] - self.player.discount(gem))
            paid = min(need, self.player.gems[gem])
            self.player.gems[gem] -= paid
            self.bank[gem] += paid
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
        eligible = [noble for noble in self.nobles if all(self.player.discount(gem) >= noble.get(gem, 0) for gem in GEM_KEYS)]
        if eligible:
            self.phase = "noble"
            self.noble_choices = eligible
            self.message = "你满足了贵族条件：请选择一位贵族拜访。"
        else:
            self.after_action()

    def choose_noble(self, noble):
        if self.phase != "noble" or noble not in self.noble_choices:
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

    def team_score(self, team):
        return sum(self.players[seat].points for seat in range(team, len(self.players), 2))

    def finish_turn(self):
        reached_target = any(self.team_score(team) >= self.target_score for team in (0, 1)) if self.team_mode else self.player.points >= self.target_score
        if reached_target:
            self.round_end = True
        self.turn = (self.turn + 1) % len(self.players)
        if self.turn == 0 and self.round_end:
            if self.team_mode:
                scores = [self.team_score(0), self.team_score(1)]
                team_cards = [sum(len(self.players[seat].cards) for seat in range(team, len(self.players), 2)) for team in (0, 1)]
                self.winner_team = max((0, 1), key=lambda team: (scores[team], -team_cards[team]))
                self.winner = self.players[self.winner_team]
                self.message = f"游戏结束：{self.winner_team + 1} 队以 {scores[self.winner_team]} 分获胜！"
            else:
                best = max(player.points for player in self.players)
                candidates = [player for player in self.players if player.points == best]
                self.winner = min(candidates, key=lambda player: len(player.cards))
                self.message = f"游戏结束：{self.winner.name} 获胜！"
            self.phase = "over"
            return
        self.phase = "action"
        self.selected_gems = {}
        self.selected_card = None
        self.message = f"轮到 {self.player.name}：选择宝石，或选中一张发展卡。"
