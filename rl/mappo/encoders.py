"""Fixed observation and action encoders for Splendor MAPPO.

Observation layout (278 floats):
    bank 6 | globals 5 | current player 9 | players 4*15 |
    table 3*4*14 | nobles 5*6

Action layout (71 unique slots):
    buy table 12 | reserve table 12 | reserve deck 3 | buy reserved 3 |
    take 30 | noble 5 | return 6
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GEM_KEYS = ["white", "blue", "green", "red", "black", "gold"]
COLORED_GEMS = GEM_KEYS[:-1]
GEM_IDX = {gem: index for index, gem in enumerate(GEM_KEYS)}

MAX_TABLE_PER_TIER = 4
N_TIERS = 3
MAX_RESERVED = 3
MAX_NOBLES = 5
N_PLAYERS = 4

BANK_DIM = 6
GLOBAL_DIM = 5
CURRENT_PLAYER_DIM = 9
PLAYER_DIM = 15
CARD_DIM = 14
NOBLE_DIM = 6
OBS_DIM = BANK_DIM + GLOBAL_DIM + CURRENT_PLAYER_DIM + N_PLAYERS * PLAYER_DIM + N_TIERS * MAX_TABLE_PER_TIER * CARD_DIM + MAX_NOBLES * NOBLE_DIM

_TAKE_COMBOS: List[Tuple[str, ...]] = [
    combo
    for size in (1, 2, 3)
    for combo in itertools.combinations(COLORED_GEMS, size)
]
_TAKE_COMBOS += [(gem, gem) for gem in COLORED_GEMS]
_TAKE_TO_INDEX = {combo: index for index, combo in enumerate(_TAKE_COMBOS)}
N_TAKE = len(_TAKE_COMBOS)

SLOT_BUY_TABLE = 0
SLOT_RESERVE_TABLE = SLOT_BUY_TABLE + 12
SLOT_RESERVE_DECK = SLOT_RESERVE_TABLE + 12
SLOT_BUY_RESERVED = SLOT_RESERVE_DECK + 3
SLOT_TAKE = SLOT_BUY_RESERVED + 3
SLOT_NOBLE = SLOT_TAKE + N_TAKE
SLOT_RETURN = SLOT_NOBLE + MAX_NOBLES
ACTION_SPACE = SLOT_RETURN + len(GEM_KEYS)

assert OBS_DIM == 278
assert N_TAKE == 30
assert ACTION_SPACE == 71


def _gem_vec(values: Dict[str, int]) -> List[float]:
    return [float(values.get(gem, 0)) for gem in GEM_KEYS]


def _card_vec(card: Optional[Dict[str, Any]]) -> List[float]:
    if not card:
        return [0.0] * CARD_DIM
    tier = int(card.get("tier", 0))
    bonus = card.get("bonus")
    return (
        [float(tier) / N_TIERS]
        + [1.0 if bonus == gem else 0.0 for gem in GEM_KEYS]
        + [float(card.get("points", 0)) / 5.0]
        + [float(card.get("cost", {}).get(gem, 0)) / 7.0 for gem in GEM_KEYS]
    )


def encode_observation(obs: Dict[str, Any], priv_seat: Optional[int] = None) -> np.ndarray:
    """Encode the authoritative public state into the fixed 278-dimensional vector.

    ``priv_seat`` is reserved for future private actor observations. The current
    game observation contains only public information and reserve-card counts.
    """
    del priv_seat
    vector: List[float] = []

    vector += _gem_vec(obs.get("bank", {}))
    target = max(1, int(obs.get("target_score", 15)))
    team_mode = bool(obs.get("team_mode", False))
    team_scores = obs.get("team_scores") or [0, 0]
    vector += [
        float(target) / 30.0,
        float(team_mode),
        float(team_scores[0]) / 30.0 if team_mode else 0.0,
        float(team_scores[1]) / 30.0 if team_mode else 0.0,
        float(bool(obs.get("round_end", False))),
    ]

    current = obs.get("current_player", {})
    vector.append(float(current.get("seat", 0)) / (N_PLAYERS - 1))
    vector += _gem_vec(current.get("gems", {}))
    vector.append(float(current.get("reserved", 0)) / MAX_RESERVED)
    vector.append(float(current.get("score", 0)) / target)

    players = obs.get("players", [])
    for seat in range(N_PLAYERS):
        if seat >= len(players):
            vector += [0.0] * PLAYER_DIM
            continue
        player = players[seat]
        vector.append(float(player.get("score", 0)) / target)
        vector += _gem_vec(player.get("gems", {}))
        vector += _gem_vec(player.get("discounts", {}))
        vector.append(float(player.get("cards", 0)) / 30.0)
        vector.append(float(player.get("reserved", 0)) / MAX_RESERVED)

    cards_by_tier = {tier: [] for tier in range(1, N_TIERS + 1)}
    for card in obs.get("table", []):
        tier = int(card.get("tier", 0))
        if tier in cards_by_tier and len(cards_by_tier[tier]) < MAX_TABLE_PER_TIER:
            cards_by_tier[tier].append(card)
    for tier in range(1, N_TIERS + 1):
        tier_cards = cards_by_tier[tier]
        for position in range(MAX_TABLE_PER_TIER):
            vector += _card_vec(tier_cards[position] if position < len(tier_cards) else None)

    nobles = obs.get("nobles", [])[:MAX_NOBLES]
    for index in range(MAX_NOBLES):
        vector += _gem_vec(nobles[index]) if index < len(nobles) else [0.0] * NOBLE_DIM

    encoded = np.asarray(vector, dtype=np.float32)
    if encoded.shape != (OBS_DIM,):
        raise ValueError(f"observation encoder produced {encoded.shape}, expected {(OBS_DIM,)}")
    return encoded


def _table_offset(tier: int, position: int) -> Optional[int]:
    if 1 <= tier <= N_TIERS and 0 <= position < MAX_TABLE_PER_TIER:
        return (tier - 1) * MAX_TABLE_PER_TIER + position
    return None


def _normalize_take(gems: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
    if not gems or any(gem not in COLORED_GEMS for gem in gems):
        return None
    if len(gems) == 2 and gems[0] == gems[1]:
        return gems
    if len(set(gems)) != len(gems) or len(gems) > 3:
        return None
    return tuple(sorted(gems, key=COLORED_GEMS.index))


def action_to_slot(action_id: str) -> Optional[int]:
    parts = action_id.split(":")
    if not parts:
        return None
    try:
        kind = parts[0]
        if kind == "buy" and len(parts) == 4:
            if parts[1] == "table":
                offset = _table_offset(int(parts[2]), int(parts[3]))
                return None if offset is None else SLOT_BUY_TABLE + offset
            if parts[1] == "reserved":
                index = int(parts[3])
                return SLOT_BUY_RESERVED + index if 0 <= index < MAX_RESERVED else None
        if kind == "reserve" and len(parts) == 3:
            offset = _table_offset(int(parts[1]), int(parts[2]))
            return None if offset is None else SLOT_RESERVE_TABLE + offset
        if kind == "reserve_deck" and len(parts) == 2:
            tier = int(parts[1])
            return SLOT_RESERVE_DECK + tier - 1 if 1 <= tier <= N_TIERS else None
        if kind == "take" and len(parts) == 2:
            combo = _normalize_take(tuple(parts[1].split(",")))
            index = _TAKE_TO_INDEX.get(combo) if combo else None
            return None if index is None else SLOT_TAKE + index
        if kind == "noble" and len(parts) == 2:
            index = int(parts[1])
            return SLOT_NOBLE + index if 0 <= index < MAX_NOBLES else None
        if kind == "return" and len(parts) == 2 and parts[1] in GEM_IDX:
            return SLOT_RETURN + GEM_IDX[parts[1]]
    except (ValueError, IndexError):
        return None
    return None


def build_action_mask(legal_actions: List[Dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE, dtype=np.float32)
    seen = set()
    for action in legal_actions:
        slot = action_to_slot(action["id"])
        if slot is None:
            raise ValueError(f"legal action has no fixed slot: {action['id']}")
        if slot in seen:
            raise ValueError(f"action slot collision at {slot}: {action['id']}")
        seen.add(slot)
        mask[slot] = 1.0
    return mask


def slot_to_action_id(slot: int, legal_actions: List[Dict[str, Any]]) -> str:
    matches = [action["id"] for action in legal_actions if action_to_slot(action["id"]) == slot]
    if len(matches) != 1:
        raise ValueError(f"slot {slot} maps to {len(matches)} legal actions")
    return matches[0]
