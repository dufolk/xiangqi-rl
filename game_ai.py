from itertools import combinations
from typing import Any, Dict, List

from game_engine import ALL_GEMS, GEM_KEYS, Game
from state_codec import card_to_data


def legal_actions(game: Game) -> List[Dict[str, Any]]:
    player = game.player
    if game.phase == "noble":
        return [{"id": f"noble:{game.nobles.index(noble)}", "type": "noble", "points": 3} for noble in game.noble_choices]
    if game.phase == "return":
        return [
            {"id": f"return:{gem}", "type": "return", "usefulness": player.discount(gem) if gem != "gold" else 5}
            for gem in ALL_GEMS if player.gems[gem] > 0
        ]

    actions: List[Dict[str, Any]] = []
    teammate = game.players[(game.turn + 2) % 4] if game.team_mode else None
    enemies = [
        game.players[seat]
        for seat in range(4)
        if seat != game.turn and (not game.team_mode or seat % 2 != game.turn % 2)
    ]

    def missing_cost(target, card):
        return sum(max(0, card.cost[gem] - target.discount(gem) - target.gems[gem]) for gem in GEM_KEYS) - target.gems["gold"]

    def noble_progress(card):
        return sum(1 for noble in game.nobles if card.bonus in noble and player.discount(card.bonus) < noble[card.bonus])

    for tier in (1, 2, 3):
        for index, card in enumerate(game.table[tier]):
            enemy_threat = max((8 - max(0, missing_cost(enemy, card)) * 2 + card.points * 2 for enemy in enemies), default=0)
            teammate_interest = max(0, 7 - missing_cost(teammate, card)) + card.points if teammate else 0
            metadata = {
                "tier": tier,
                "points": card.points,
                "noble_progress": noble_progress(card),
                "enemy_threat": enemy_threat,
                "teammate_interest": teammate_interest,
            }
            if game.can_afford(card):
                actions.append({"id": f"buy:table:{tier}:{index}", "type": "buy", **metadata})
            if len(player.reserved) < 3:
                actions.append({"id": f"reserve:{tier}:{index}", "type": "reserve", **metadata})
        if len(player.reserved) < 3 and game.decks[tier]:
            actions.append({"id": f"reserve_deck:{tier}", "type": "reserve", "tier": tier, "points": 0})
    for index, card in enumerate(player.reserved):
        if card is not None and game.can_afford(card):
            actions.append({"id": f"buy:reserved:0:{index}", "type": "buy", "tier": card.tier, "points": card.points})

    available = [gem for gem in GEM_KEYS if game.bank[gem] > 0]
    wanted = {gem: 0 for gem in GEM_KEYS}
    enemy_need = {gem: 0 for gem in GEM_KEYS}
    teammate_need = {gem: 0 for gem in GEM_KEYS}
    for tier in game.table:
        for card in game.table[tier]:
            for gem in GEM_KEYS:
                wanted[gem] += max(0, card.cost[gem] - player.discount(gem) - player.gems[gem])
                enemy_need[gem] += sum(max(0, card.cost[gem] - enemy.discount(gem) - enemy.gems[gem]) for enemy in enemies)
                if teammate:
                    teammate_need[gem] += max(0, card.cost[gem] - teammate.discount(gem) - teammate.gems[gem])
    for combo in combinations(available, min(3, len(available))):
        if combo:
            actions.append({
                "id": "take:" + ",".join(combo),
                "type": "take",
                "usefulness": sum(wanted[gem] for gem in combo),
                "deny_value": sum(enemy_need[gem] for gem in combo),
                "teammate_need": sum(teammate_need[gem] for gem in combo),
            })
    for gem in GEM_KEYS:
        if game.bank[gem] >= 4:
            actions.append({
                "id": f"take:{gem},{gem}",
                "type": "take",
                "usefulness": wanted[gem] * 2,
                "deny_value": enemy_need[gem] * 2,
                "teammate_need": teammate_need[gem] * 2,
            })
    return actions


def observation(game: Game) -> Dict[str, Any]:
    return {
        "target_score": game.target_score,
        "team_mode": game.team_mode,
        "team_scores": [game.team_score(0), game.team_score(1)] if game.team_mode else None,
        "bank": game.bank,
        "current_player": {
            "seat": game.turn,
            "score": game.player.points,
            "gems": game.player.gems,
            "discounts": {gem: game.player.discount(gem) for gem in GEM_KEYS},
            "reserved": len(game.player.reserved),
        },
        "players": [{
            "seat": seat,
            "relation": "self" if seat == game.turn else "teammate" if game.team_mode and seat % 2 == game.turn % 2 else "opponent",
            "score": player.points,
            "cards": len(player.cards),
            "gems": player.gems,
            "discounts": {gem: player.discount(gem) for gem in GEM_KEYS},
        } for seat, player in enumerate(game.players)],
        "table": [card_to_data(card) for tier in (1, 2, 3) for card in game.table[tier]],
        "nobles": game.nobles,
        "round_end": game.round_end,
    }


def apply_action(game: Game, action_id: str) -> None:
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
    elif parts[0] == "reserve_deck":
        game.reserve_from_deck(int(parts[1]))
    elif parts[0] == "take":
        gems = parts[1].split(",")
        game.selected_gems = {gem: gems.count(gem) for gem in set(gems)}
        game.confirm_gems()
