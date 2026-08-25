"""Observation / action encoders for MAPPO on Splendor.

Design goals (CTDE):
- A *fixed-length* global observation vector so the centralized critic can
  ingest the full joint state, while decentralized actors can share the same
  encoder (optionally privacy-masked downstream).
- A *fixed-size* discrete action space. The actual legal actions are produced
  by ``game_ai.legal_actions``; we map each legal action onto a fixed slot via
  an action mask so PPO can operate over a constant dimensionality.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

GEM_KEYS = ["diamond", "sapphire", "emerald", "ruby", "onyx", "gold"]
GEM_IDX = {g: i for i, g in enumerate(GEM_KEYS)}

# Fixed board sizes -------------------------------------------------------
MAX_TABLE_PER_TIER = 4          # game_engine lays out 4 cards per tier
N_TIERS = 3
MAX_TABLE_CARDS = MAX_TABLE_PER_TIER * N_TIERS   # 12
MAX_RESERVED = 3
MAX_NOBLES = 5
N_PLAYERS = 4

# Fixed action space -------------------------------------------------------
# Slots:
#   0..11     buy table card (tier*4 + pos)          12
#   12..14    reserve table card (tier*4 + pos)       3
#   15..17    reserve deck (tier)                     3
#   18..20    buy reserved (reserved index)           3
#   21..25    take 3 distinct gems (one slot per gem as "anchor"; see below) -> NOT used
#   We instead enumerate concrete take combos explicitly.
# To keep it simple & exhaustive enough, we enumerate take combos programmatically.
import itertools
_TAKE_COMBOS: List[Tuple[str, ...]] = []
for r in (1, 2, 3):
    for combo in itertools.combinations(GEM_KEYS[:-1], r):  # exclude gold from 3-distinct
        _TAKE_COMBOS.append(combo)
# double-take (need >=4 in bank): one gem twice
for g in GEM_KEYS[:-1]:
    _TAKE_COMBOS.append((g, g))
# gold-only special: taking gold alone not allowed by rules, skip.
N_TAKE = len(_TAKE_COMBOS)

# Slot layout
SLOT_BUY_TABLE = 0                      # +12
SLOT_RESERVE_TABLE = 12                 # +3
SLOT_RESERVE_DECK = 15                  # +3
SLOT_BUY_RESERVED = 18                  # +3
SLOT_TAKE = 21                          # +N_TAKE
SLOT_NOBLE = 21 + N_TAKE                # +MAX_NOBLES
SLOT_RETURN = SLOT_NOBLE + MAX_NOBLES   # +6 (return each gem)
ACTION_DIM = SLOT_RETURN + 6

# Noble / return only happen in specific phases; mask handles legality.


def _gem_vec(d: Dict[str, int]) -> List[float]:
    return [float(d.get(g, 0)) for g in GEM_KEYS]


def encode_observation(obs: Dict[str, Any], priv_seat: int = None) -> np.ndarray:
    """Encode a global observation dict into a fixed-length float vector.

    ``priv_seat`` is accepted for API symmetry (privacy masking can be applied
    here later); currently the full global state is encoded for CTDE.
    """
    vec: List[float] = []

    # Bank (6)
    vec += _gem_vec(obs.get("bank", {}))

    # Global scalars
    vec.append(float(obs.get("target_score", 15)))
    vec.append(float(obs.get("team_mode", 0)))
    if obs.get("team_mode"):
        ts = obs.get("team_scores") or [0, 0]
        vec += [float(ts[0]), float(ts[1])]
    else:
        vec += [0.0, 0.0]
    vec.append(float(obs.get("round_end", 0)))

    # Current player block (6): seat one-hot-ish (seat norm) + gems
    cur = obs.get("current_player", {})
    vec.append(float(cur.get("seat", 0)) / max(1, N_PLAYERS - 1))
    vec += _gem_vec(cur.get("gems", {}))
    vec.append(float(cur.get("reserved", 0)) / MAX_RESERVED)
    vec.append(float(cur.get("score", 0)) / max(1, obs.get("target_score", 15)))

    # All players (4 x (score1 + gems6 + discounts6 + cards1 + reserved1))
    players = obs.get("players", [])
    for seat in range(N_PLAYERS):
        if seat < len(players):
            p = players[seat]
            vec.append(float(p.get("score", 0)) / max(1, obs.get("target_score", 15)))
            vec += _gem_vec(p.get("gems", {}))
            vec += _gem_vec(p.get("discounts", {}))
            vec.append(float(p.get("cards", 0)) / 30.0)
            vec.append(float(p.get("reserved", 0)) / MAX_RESERVED)
        else:
            vec += [0.0] * (1 + 6 + 6 + 1 + 1)

    # Table cards: MAX_TABLE_CARDS x (tier1 + bonus6 + points1 + cost6)
    table_flat: List[Dict[str, Any]] = obs.get("table", [])
    # table_flat may be a flat list already (as produced by observation())
    card_vecs = []
    for card in table_flat[:MAX_TABLE_CARDS]:
        cv = [float(card.get("tier", 0)) / N_TIERS]
        bonus = card.get("bonus")
        cv += [1.0 if bonus == g else 0.0 for g in GEM_KEYS]
        cv.append(float(card.get("points", 0)) / 5.0)
        cv += [float(card.get("cost", {}).get(g, 0)) / 7.0 for g in GEM_KEYS]
        card_vecs.append(cv)
    while len(card_vecs) < MAX_TABLE_CARDS:
        card_vecs.append([0.0] * (1 + 6 + 1 + 6))
    for cv in card_vecs:
        vec += cv

    # Nobles: MAX_NOBLES x 6 (each noble is a gem->count dict)
    nobles = obs.get("nobles", [])[:MAX_NOBLES]
    for noble in nobles:
        vec += _gem_vec(noble)
    while len(nobles) < MAX_NOBLES:
        vec += [0.0] * 6
        nobles = nobles + [None]  # keep loop bound stable

    return np.asarray(vec, dtype=np.float32)


def build_action_mask(legal: List[Dict[str, Any]]) -> np.ndarray:
    """Return a boolean-ish float mask of size ACTION_DIM for legal actions."""
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for a in legal:
        idx = _action_to_slot(a["id"])
        if idx is not None and 0 <= idx < ACTION_DIM:
            mask[idx] = 1.0
    return mask


def _action_to_slot(action_id: str):
    parts = action_id.split(":")
    if not parts:
        return None
    kind = parts[0]
    try:
        if kind == "buy":
            loc = parts[1]
            if loc == "table":
                tier, pos = int(parts[2]), int(parts[3])
                return SLOT_BUY_TABLE + tier * MAX_TABLE_PER_TIER + pos
            elif loc == "reserved":
                return SLOT_BUY_RESERVED + int(parts[3])
        elif kind == "reserve":
            if len(parts) == 3:  # reserve:table:tier:pos
                tier, pos = int(parts[1]), int(parts[2])
                return SLOT_RESERVE_TABLE + tier * MAX_TABLE_PER_TIER + pos
        elif kind == "reserve_deck":
            tier = int(parts[1])
            return SLOT_RESERVE_DECK + tier
        elif kind == "take":
            gems = tuple(parts[1].split(","))
            # normalize ordering for combo lookup
            norm = tuple(sorted(gems)) if len(set(gems)) == len(gems) else gems
            for i, combo in enumerate(_TAKE_COMBOS):
                if tuple(sorted(combo)) == tuple(sorted(gems)) or combo == gems:
                    return SLOT_TAKE + i
            return None
        elif kind == "noble":
            idx = int(parts[1])
            if 0 <= idx < MAX_NOBLES:
                return SLOT_NOBLE + idx
        elif kind == "return":
            gem = parts[1]
            if gem in GEM_IDX:
                return SLOT_RETURN + GEM_IDX[gem]
    except (ValueError, IndexError):
        return None
    return None


def slot_to_action_id(slot: int, legal: List[Dict[str, Any]]) -> str:
    """Map a chosen slot back to a legal action id (must be within mask)."""
    # Recompute mapping from legal list to be safe
    for a in legal:
        if _action_to_slot(a["id"]) == slot:
            return a["id"]
    # fallback: first legal
    return legal[0]["id"] if legal else ""


# Convenience shapes
def obs_dim() -> int:
    # compute once by encoding an empty-ish structure is overkill; hardcode via probe
    probe = {
        "bank": {g: 0 for g in GEM_KEYS},
        "target_score": 15, "team_mode": 0, "team_scores": [0, 0],
        "round_end": 0,
        "current_player": {"seat": 0, "gems": {g: 0 for g in GEM_KEYS}, "reserved": 0, "score": 0},
        "players": [{"score": 0, "gems": {g: 0 for g in GEM_KEYS}, "discounts": {g: 0 for g in GEM_KEYS}, "cards": 0, "reserved": 0} for _ in range(N_PLAYERS)],
        "table": [],
        "nobles": [],
    }
    return int(encode_observation(probe).shape[0])


OBS_DIM = obs_dim()
ACTION_SPACE = ACTION_DIM
