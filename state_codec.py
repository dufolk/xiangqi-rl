"""Serialization helpers shared by GUI, tests and training tools."""

from __future__ import annotations

from typing import Any

from game_engine import Piece, XiangqiGame


def game_to_data(game: XiangqiGame) -> dict[str, Any]:
    return {
        "version": 1,
        "fen": game.to_fen(),
        "winner": game.winner,
        "reason": game.reason,
        "halfmove": game.halfmove,
        "last_move": None if game.last_move is None else {
            "start": list(game.last_move.start),
            "end": list(game.last_move.end),
        },
    }


def game_from_data(data: dict[str, Any]) -> XiangqiGame:
    game = XiangqiGame.from_fen(str(data["fen"]))
    winner = data.get("winner")
    game.winner = winner if winner in ("red", "black") else None
    game.reason = str(data.get("reason", ""))
    game.halfmove = max(0, int(data.get("halfmove", 0)))
    return game


def board_signature(game: XiangqiGame) -> tuple[tuple[Piece | None, ...], ...]:
    return tuple(tuple(row) for row in game.board)
