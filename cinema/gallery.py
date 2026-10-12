"""Catalogue of every cutscene, used by the in-game gallery and offline renders."""

from __future__ import annotations

from game_engine import ADVISOR, CANNON, ELEPHANT, HORSE, KING, KINDS, PAWN, RED, ROOK, Piece, opponent
from cinema.detect import MATES, OPENINGS, STALEMATE, StoryEvent, capture_event, mate_event, opening_event

OPENING_ACTOR = {
    "center_cannon": CANNON, "cross_palace_cannon": CANNON, "advisor_cannon": CANNON, "pawn_base_cannon": CANNON,
    "retreat_cannon": CANNON, "edge_cannon": CANNON, "river_cannon": CANNON, "hook_cannon": CANNON,
    "cross_river_cannon": CANNON, "immortal_pawn": PAWN, "edge_pawn": PAWN, "center_pawn": PAWN,
    "flying_elephant": ELEPHANT, "edge_elephant": ELEPHANT, "horse_opening": HORSE, "edge_horse": HORSE,
    "screen_horses": HORSE, "same_cannons": CANNON, "crossed_cannons": CANNON, "advisor_opening": ADVISOR,
    "rook_opening": ROOK, "king_opening": KING,
}

MATE_ACTOR = {
    "double_rook": ROOK, "wocao_horse": HORSE, "corner_horse": HORSE, "fishing_horse": HORSE,
    "horse_cannon": CANNON, "double_cannon": CANNON, "heaven_earth": CANNON, "iron_bolt": ROOK,
    "heart_strike": ROOK, "facing_kings": KING, "sea_moon": ROOK, "two_ghosts": PAWN, "generic": ROOK,
}


def catalogue() -> list[tuple[str, str]]:
    items = [("opening", key) for key in OPENINGS]
    items += [("capture", a + v) for a in KINDS for v in KINDS if v != KING]
    items += [("mate", key) for key in MATES]
    items.append(("stalemate", "stalemate"))
    return items


def build_event(category: str, key: str, side: str = RED) -> StoryEvent:
    if category == "opening":
        return opening_event(key, Piece(side, OPENING_ACTOR[key]))
    if category == "capture":
        return capture_event(Piece(side, key[0]), Piece(opponent(side), key[1]))
    if category in ("mate", "check"):
        return mate_event(key, Piece(side, MATE_ACTOR[key]), finale=category == "mate")
    title, subtitle = STALEMATE
    return StoryEvent("stalemate", "stalemate", title, subtitle, Piece(side, ROOK))


def build_scene(category: str, key: str, side: str = RED):
    from cinema.director import scene_for_event

    return scene_for_event(build_event(category, key, side))
