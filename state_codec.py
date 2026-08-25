from typing import Any, Dict, Optional

from game_engine import GEM_KEYS, Card, Game, Player


def card_to_data(card: Card) -> Dict[str, Any]:
    return {
        "tier": card.tier,
        "bonus": card.bonus,
        "points": card.points,
        "cost": card.cost,
        "art_index": card.art_index,
    }


def card_from_data(data: Dict[str, Any]) -> Card:
    return Card(
        int(data["tier"]),
        data["bonus"],
        int(data["points"]),
        {gem: int(data["cost"].get(gem, 0)) for gem in GEM_KEYS},
        int(data.get("art_index", 0)),
    )


def game_to_data(game: Game, viewer_seat: Optional[int] = None) -> Dict[str, Any]:
    """Serialize an authoritative game into a privacy-filtered client view."""
    winner_seat = game.players.index(game.winner) if game.winner in game.players else None
    players = []
    for seat, player in enumerate(game.players):
        reserved = [card_to_data(card) for card in player.reserved if card is not None] if seat == viewer_seat else [None] * len(player.reserved)
        players.append({
            "name": player.name,
            "color": list(player.color),
            "gems": player.gems,
            "cards": [card_to_data(card) for card in player.cards],
            "reserved": reserved,
            "nobles": player.nobles,
        })
    return {
        "ai_count": game.ai_count,
        "human_seat": game.human_seat,
        "target_score": game.target_score,
        "team_mode": game.team_mode,
        "player_name": game.player_name,
        "bank": game.bank,
        "deck_counts": {str(tier): len(cards) for tier, cards in game.decks.items()},
        "table": {str(tier): [card_to_data(card) for card in cards] for tier, cards in game.table.items()},
        "nobles": game.nobles,
        "players": players,
        "turn": game.turn,
        "selected_card": list(game.selected_card) if game.selected_card else None,
        "selected_gems": game.selected_gems,
        "phase": game.phase,
        "noble_choices": game.noble_choices,
        "round_end": game.round_end,
        "winner_seat": winner_seat,
        "winner_team": game.winner_team,
        "message": game.message,
        "ai_seats": sorted(game.ai_seats),
    }


def game_from_data(data: Dict[str, Any]) -> Game:
    """Build a read-only client projection from a privacy-filtered state payload."""
    restored = Game.__new__(Game)
    restored.ai_count = int(data["ai_count"])
    restored.human_seat = int(data["human_seat"])
    restored.target_score = int(data["target_score"])
    restored.team_mode = bool(data.get("team_mode", False))
    restored.player_name = data["player_name"]
    restored.rng = None
    restored.bank = {gem: int(value) for gem, value in data["bank"].items()}
    restored.decks = {int(tier): [None] * int(count) for tier, count in data["deck_counts"].items()}
    restored.table = {int(tier): [card_from_data(card) for card in cards] for tier, cards in data["table"].items()}
    restored.nobles = data["nobles"]
    restored.players = []
    for item in data["players"]:
        player = Player(item["name"], tuple(item["color"]))
        player.gems = {gem: int(value) for gem, value in item["gems"].items()}
        player.cards = [card_from_data(card) for card in item["cards"]]
        player.reserved = [card_from_data(card) if card else None for card in item["reserved"]]
        player.nobles = item["nobles"]
        restored.players.append(player)
    restored.turn = int(data["turn"])
    restored.selected_card = tuple(data["selected_card"]) if data["selected_card"] else None
    restored.selected_gems = {gem: int(value) for gem, value in data["selected_gems"].items()}
    restored.phase = data["phase"]
    restored.noble_choices = data["noble_choices"]
    restored.round_end = bool(data["round_end"])
    winner_seat = data.get("winner_seat")
    restored.winner = restored.players[winner_seat] if winner_seat is not None else None
    restored.winner_team = data.get("winner_team")
    restored.message = data["message"]
    restored.ai_seats = set(data["ai_seats"])
    return restored
