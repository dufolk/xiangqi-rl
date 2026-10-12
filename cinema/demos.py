"""Ready-made positions that trigger every story beat: used by the gallery, tests and tools."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from cinema.detect import CAPTURES, MATES, OPENINGS, TACTICS, VALUE, material_balance
from game_engine import (
    ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, KINDS, PAWN, RED, ROOK, Move, Piece, XiangqiGame,
)

Square = tuple[int, int]
BK, RK = Piece(BLACK, KING), Piece(RED, KING)


def R(kind: str) -> Piece:
    return Piece(RED, kind)


def B(kind: str) -> Piece:
    return Piece(BLACK, kind)


@dataclass(frozen=True)
class Demo:
    category: str  # opening | capture | tone | tactic | mate | stalemate
    key: str
    title: str
    pieces: Optional[tuple[tuple[Square, Piece], ...]]  # None = standard opening position
    prelude: tuple[tuple[Square, Square], ...]
    move: tuple[Square, Square]

    def build(self) -> XiangqiGame:
        """The position right before ``move``, with any prelude moves already played."""
        game = XiangqiGame()
        if self.pieces is not None:
            game.clear()
            for (row, col), piece in self.pieces:
                game.set_piece(row, col, piece)
            game.turn = RED
        for start, end in self.prelude:
            game.push(Move(start, end))
        return game


OPENING_MOVES: dict[str, tuple[tuple[Square, Square], ...]] = {
    "center_cannon": (((7, 7), (7, 4)),),
    "cross_palace_cannon": (((7, 7), (7, 3)),),
    "advisor_cannon": (((7, 7), (7, 5)),),
    "pawn_base_cannon": (((6, 6), (5, 6)), ((2, 7), (2, 6))),
    "retreat_cannon": (((7, 7), (8, 7)),),
    "edge_cannon": (((7, 7), (7, 8)),),
    "river_cannon": (((7, 7), (5, 7)),),
    "hook_cannon": (((7, 7), (6, 7)),),
    "cross_river_cannon": (((7, 7), (4, 7)),),
    "immortal_pawn": (((6, 6), (5, 6)),),
    "edge_pawn": (((6, 0), (5, 0)),),
    "center_pawn": (((6, 4), (5, 4)),),
    "flying_elephant": (((9, 6), (7, 4)),),
    "edge_elephant": (((9, 6), (7, 8)),),
    "horse_opening": (((9, 7), (7, 6)),),
    "edge_horse": (((9, 7), (7, 8)),),
    "screen_horses": (((7, 7), (7, 4)), ((0, 7), (2, 6)), ((9, 7), (7, 6)), ((0, 1), (2, 2))),
    "same_cannons": (((7, 7), (7, 4)), ((2, 7), (2, 4))),
    "crossed_cannons": (((7, 7), (7, 4)), ((2, 1), (2, 4))),
    "advisor_opening": (((9, 3), (8, 4)),),
    "rook_opening": (((9, 0), (8, 0)),),
    "king_opening": (((9, 4), (8, 4)),),
}

TACTIC_POSITIONS: dict[str, tuple[dict[Square, Piece], Square, Square]] = {
    "fork": ({(9, 4): RK, (0, 3): BK, (5, 3): R(HORSE), (1, 3): B(ROOK), (1, 5): B(ROOK)}, (5, 3), (3, 4)),
    "discovered_attack": ({(9, 4): RK, (0, 3): BK, (8, 0): R(ROOK), (5, 0): R(HORSE), (2, 0): B(HORSE)}, (5, 0), (3, 1)),
    "discovered_check": ({(9, 5): RK, (0, 3): BK, (6, 3): R(ROOK), (3, 3): R(HORSE)}, (3, 3), (4, 1)),
    "double_check": ({(9, 5): RK, (0, 3): BK, (6, 3): R(ROOK), (4, 3): R(HORSE)}, (4, 3), (2, 4)),
    "check_win": ({(9, 5): RK, (0, 4): BK, (4, 2): R(HORSE), (4, 4): B(ROOK)}, (4, 2), (2, 3)),
}

MATE_POSITIONS: dict[str, tuple[dict[Square, Piece], Square, Square]] = {
    "double_rook": ({(9, 3): RK, (0, 4): BK, (1, 8): R(ROOK), (5, 0): R(ROOK)}, (5, 0), (0, 0)),
    "wocao_horse": ({(9, 3): RK, (0, 4): BK, (0, 3): B(ADVISOR), (1, 4): B(ADVISOR), (5, 5): R(ROOK), (3, 3): R(HORSE)},
                    (3, 3), (1, 2)),
    "corner_horse": ({(9, 3): RK, (0, 4): BK, (0, 3): B(ADVISOR), (0, 5): B(ADVISOR), (1, 8): R(ROOK), (4, 4): R(HORSE)},
                     (4, 4), (2, 3)),
    "fishing_horse": ({(9, 5): RK, (0, 3): BK, (5, 4): R(ROOK), (2, 3): R(PAWN), (4, 3): R(HORSE)}, (4, 3), (2, 2)),
    "horse_cannon": ({(9, 3): RK, (0, 4): BK, (2, 4): R(HORSE), (6, 0): R(CANNON)}, (6, 0), (6, 4)),
    "double_cannon": ({(9, 3): RK, (0, 4): BK, (0, 3): B(ADVISOR), (0, 5): B(ADVISOR), (3, 4): R(CANNON), (6, 0): R(CANNON)},
                      (6, 0), (6, 4)),
    "heaven_earth": ({(9, 3): RK, (0, 4): BK, (0, 3): B(ADVISOR), (0, 2): B(ELEPHANT), (0, 0): R(CANNON),
                      (1, 5): R(PAWN), (3, 4): R(PAWN), (5, 0): R(CANNON)}, (5, 0), (5, 4)),
    "iron_bolt": ({(9, 3): RK, (0, 4): BK, (1, 4): B(ADVISOR), (0, 5): B(ADVISOR), (2, 4): B(ELEPHANT),
                   (5, 4): R(CANNON), (2, 2): R(HORSE), (5, 3): R(ROOK)}, (5, 3), (0, 3)),
    "heart_strike": ({(9, 3): RK, (0, 4): BK, (1, 4): B(ADVISOR), (1, 3): R(PAWN), (1, 5): R(PAWN), (5, 4): R(ROOK)},
                     (5, 4), (1, 4)),
    "facing_kings": ({(9, 4): RK, (0, 3): BK, (2, 3): R(PAWN), (5, 3): R(ROOK)}, (2, 3), (1, 3)),
    "sea_moon": ({(9, 3): RK, (1, 4): BK, (3, 4): R(HORSE), (0, 3): R(PAWN), (0, 8): R(ROOK)}, (0, 8), (0, 4)),
    "two_ghosts": ({(9, 3): RK, (0, 4): BK, (1, 5): R(PAWN), (1, 3): R(PAWN)}, (1, 3), (0, 3)),
}

STALEMATE_POSITION = ({(9, 4): RK, (0, 3): BK, (2, 4): R(PAWN)}, (2, 4), (1, 4))

# Extra material dropped in the corners to tilt the balance for the tone demos.
TONE_EXTRAS: dict[str, dict[Square, Piece]] = {
    "dominant": {(8, 0): R(ROOK), (8, 8): R(ROOK)},
    "even": {},
    "desperate": {(1, 0): B(ROOK), (1, 8): B(ROOK)},
    "comeback": {(1, 0): B(CANNON)},
}
TONE_TITLES = {"dominant": "大优 · 乘胜追击", "even": "相持 · 针锋相对", "desperate": "大劣 · 绝地反击", "comeback": "小劣 · 扳回一城"}


def _valid_squares(piece: Piece) -> list[Square]:
    red = piece.side == RED
    if piece.kind == KING:
        rows = (7, 8, 9) if red else (0, 1, 2)
        return [(r, c) for r in rows for c in (3, 4, 5)]
    if piece.kind == ADVISOR:
        return [(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)] if red else [(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)]
    if piece.kind == ELEPHANT:
        own = [(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)]
        return [(9 - r, c) for r, c in own] if red else own
    if piece.kind == PAWN:
        if red:
            return [(r, c) for r in (5, 6) for c in (0, 2, 4, 6, 8)] + [(r, c) for r in range(5) for c in range(9)]
        return [(r, c) for r in (3, 4) for c in (0, 2, 4, 6, 8)] + [(r, c) for r in range(5, 10) for c in range(9)]
    return [(r, c) for r in range(10) for c in range(9)]


def _quiet(game: XiangqiGame) -> bool:
    return not game.generals_facing() and not game.is_in_check(RED) and not game.is_in_check(BLACK)


def _game(pieces: dict[Square, Piece]) -> XiangqiGame:
    game = XiangqiGame()
    game.clear()
    for (r, c), p in pieces.items():
        game.set_piece(r, c, p)
    return game


def _capture_ok(pieces: dict[Square, Piece], start: Square, end: Square) -> bool:
    """Legal, quiet before, and the capture neither ends the game nor gives check."""
    game = _game(pieces)
    if not _quiet(game) or not any(m.end == end for m in game.legal_moves_from(start)):
        return False
    game.push(Move(start, end))
    return game.winner is None and not game.is_in_check(BLACK)


SLOTS = {RED: [(9, 0), (9, 8), (9, 1), (9, 7), (8, 0), (8, 8)], BLACK: [(0, 0), (0, 8), (0, 1), (0, 7), (1, 0), (1, 8)]}


def _balanced(pieces: dict[Square, Piece], start: Square, end: Square) -> Optional[dict[Square, Piece]]:
    """Top up the poorer side from the corners until the material is roughly level."""
    merged = dict(pieces)
    for _ in range(6):
        balance = material_balance(_game(merged).board)
        if abs(balance) < 2.0:
            break
        side = BLACK if balance > 0 else RED
        kind = min((ROOK, CANNON, HORSE), key=lambda k: abs(VALUE[k] - abs(balance)))
        for slot in SLOTS[side]:
            if slot not in merged and _capture_ok({**merged, slot: Piece(side, kind)}, start, end):
                merged[slot] = Piece(side, kind)
                break
        else:
            break
    return merged if _capture_ok(merged, start, end) else None


def _capture_score(attacker: Square, victim: Square, kind: str) -> float:
    dist = abs(attacker[0] - victim[0]) + abs(attacker[1] - victim[1])
    ideal = {ROOK: 4, CANNON: 4}.get(kind, 2)
    centre = abs(victim[1] - 4) * 0.4 + abs(victim[0] - 4.5) * 0.3
    return abs(dist - ideal) + centre


@lru_cache(maxsize=None)
def capture_position(attacker_kind: str, victim_kind: str) -> Optional[tuple[tuple[tuple[Square, Piece], ...], Square, Square]]:
    """A small legal position where a red ``attacker_kind`` can capture a black ``victim_kind``."""
    if victim_kind == KING:
        return None
    attacker, victim = R(attacker_kind), B(victim_kind)
    kings = [((9, 3), (0, 5)), ((9, 5), (0, 3)), ((8, 4), (0, 3)), ((9, 4), (0, 3)), ((9, 3), (2, 5))]
    candidates = sorted(
        (_capture_score(asq, vsq, attacker_kind), asq, vsq)
        for vsq in _valid_squares(victim) for asq in _valid_squares(attacker) if asq != vsq
    )
    for _, asq, vsq in candidates:
        screens: list[Optional[Square]] = [None]
        if attacker_kind == CANNON:
            if asq[0] == vsq[0]:
                lo, hi = sorted((asq[1], vsq[1]))
                screens = [(asq[0], c) for c in range(lo + 1, hi)]
            elif asq[1] == vsq[1]:
                lo, hi = sorted((asq[0], vsq[0]))
                screens = [(r, asq[1]) for r in range(lo + 1, hi)]
            else:
                continue
        elif attacker_kind != HORSE and abs(asq[0] - vsq[0]) + abs(asq[1] - vsq[1]) > 6:
            continue
        for rk, bk in kings:
            if len({asq, vsq, rk, bk}) < 4:
                continue
            for screen in screens:
                if screen in (rk, bk):
                    continue
                pieces = {rk: RK, bk: BK, asq: attacker, vsq: victim}
                if screen is not None:
                    pieces[screen] = R(PAWN) if screen[0] <= 4 else B(PAWN)
                game = _game(pieces)
                if not _quiet(game) or not any(m.end == vsq for m in game.legal_moves_from(asq)):
                    continue
                balanced = _balanced(pieces, asq, vsq)
                if balanced is not None:
                    return tuple(balanced.items()), asq, vsq
    return None


def _with(pieces, extra: dict[Square, Piece]):
    merged = dict(pieces)
    for sq, p in extra.items():
        if sq not in merged:
            merged[sq] = p
    return tuple(merged.items())


@lru_cache(maxsize=None)
def catalogue() -> tuple[Demo, ...]:
    demos: list[Demo] = []
    for key, moves in OPENING_MOVES.items():
        demos.append(Demo("opening", key, OPENINGS[key][0], None, moves[:-1], moves[-1]))
    for a in KINDS:
        for v in KINDS:
            found = capture_position(a, v)
            if found is not None:
                pieces, start, end = found
                demos.append(Demo("capture", a + v, CAPTURES[a + v][0], pieces, (), (start, end)))
    base = capture_position(ROOK, HORSE)
    if base is not None:
        for tone, extra in TONE_EXTRAS.items():
            demos.append(Demo("tone", tone, TONE_TITLES[tone], _with(base[0], extra), (), (base[1], base[2])))
    for key, (pieces, start, end) in TACTIC_POSITIONS.items():
        demos.append(Demo("tactic", key, TACTICS[key][0].format(x="车"), tuple(pieces.items()), (), (start, end)))
    for key, (pieces, start, end) in MATE_POSITIONS.items():
        demos.append(Demo("mate", key, MATES[key][0], tuple(pieces.items()), (), (start, end)))
    pieces, start, end = STALEMATE_POSITION
    demos.append(Demo("stalemate", "stalemate", "困毙", tuple(pieces.items()), (), (start, end)))
    return tuple(demos)


def find(category: str, key: str) -> Optional[Demo]:
    return next((d for d in catalogue() if d.category == category and d.key == key), None)
