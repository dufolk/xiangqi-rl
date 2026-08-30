"""Traditional alpha-beta (negamax) search opponent for Xiangqi PPO training.

Used as the fixed opponent when training against a search-based baseline instead
of pure self-play. The search depth is configurable via :class:`SearchOpponent`.

The evaluation combines material with small piece-square style positional terms
(cheap, always on). An optional quiescence (capture-only) search removes the
horizon effect for stronger tactical play, at the cost of much higher per-move
latency -- enable it only when you can afford the slowdown.
"""

from __future__ import annotations

from typing import Optional

from game_engine import (
    CANNON, HORSE, KING, PAWN, RED, ROOK, ROWS, COLS, Move, iter_pieces,
)

PIECE_VALUE = {
    "R": 9.0, "C": 4.5, "H": 4.2, "E": 2.1, "A": 2.1, "P": 1.0, "K": 1000.0,
}
MATE_SCORE = 100000.0


def _positional(side: str, kind: str, row: int, col: int) -> float:
    """Positional bonus from RED's reference frame (RED advances toward row 0)."""
    r = row if side == RED else (ROWS - 1 - row)
    center = abs(col - 4)
    if kind == PAWN:
        if r <= 4:  # already crossed the river
            return 0.4 + (4 - r) * 0.2 - 0.05 * center
        return 0.05 - 0.05 * center
    if kind in (HORSE, CANNON):
        return -0.08 * center
    if kind == ROOK:
        return -0.04 * center
    return 0.0


def evaluate(game) -> float:
    """Static evaluation (material + position) from the side-to-move's view."""
    score = 0.0
    for row, col, piece in iter_pieces(game.board):
        value = PIECE_VALUE.get(piece.kind, 0.0) + _positional(piece.side, piece.kind, row, col)
        score += value if piece.side == game.turn else -value
    return score


def _ordered_moves(game) -> list[Move]:
    """Legal moves ordered with captures first to improve alpha-beta pruning."""
    moves = game.legal_moves()
    moves.sort(
        key=lambda move: PIECE_VALUE.get(move.captured.kind, 0.0) if move.captured else 0.0,
        reverse=True,
    )
    return moves


def _capture_moves(game) -> list[Move]:
    return [move for move in game.legal_moves() if move.captured is not None]


def quiesce(game, alpha: float, beta: float) -> float:
    """Stand-pat quiescence search over captures only, from side-to-move's view."""
    if game.winner is not None:
        return -MATE_SCORE
    stand = evaluate(game)
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand
    for move in sorted(_capture_moves(game), key=lambda move: PIECE_VALUE.get(move.captured.kind, 0.0), reverse=True):
        game.push(move)
        value = -quiesce(game, -beta, -alpha)
        game.undo()
        if value >= beta:
            return beta
        if value > alpha:
            alpha = value
    return alpha


def negamax(game, depth: int, alpha: float, beta: float, use_quiescence: bool = False) -> float:
    """Negamax with alpha-beta pruning; returns value from the side-to-move view."""
    if game.winner is not None:
        # The side to move has already lost (previous move delivered mate).
        return -(MATE_SCORE + depth)
    moves = _ordered_moves(game)
    if not moves:
        return -(MATE_SCORE + depth)
    if depth == 0:
        return quiesce(game, alpha, beta) if use_quiescence else evaluate(game)
    best = -float("inf")
    for move in moves:
        game.push(move)
        value = -negamax(game, depth - 1, -beta, -alpha, use_quiescence)
        game.undo()
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


class SearchOpponent:
    """Fixed-strength opponent that picks moves via alpha-beta search."""

    def __init__(self, depth: int = 3, use_quiescence: bool = False):
        if depth < 1:
            raise ValueError("search depth must be at least 1")
        self.depth = depth
        self.use_quiescence = use_quiescence

    def select_move(self, game) -> Optional[Move]:
        """Return the best move for the side to move, or ``None`` if none exist."""
        best_move: Optional[Move] = None
        best_value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        for move in _ordered_moves(game):
            game.push(move)
            value = -negamax(game, self.depth - 1, -beta, -alpha, self.use_quiescence)
            game.undo()
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
        return best_move
