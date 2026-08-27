"""Lightweight deterministic Xiangqi opponent used by the desktop app."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from game_engine import (
    ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK,
    Move, XiangqiGame, iter_pieces, opponent,
)

VALUES = {KING: 100_000, ROOK: 900, CANNON: 450, HORSE: 420, ELEPHANT: 210, ADVISOR: 210, PAWN: 100}


@dataclass
class SearchResult:
    move: Optional[Move]
    score: float
    nodes: int


class SimpleAI:
    """Material/position alpha-beta player.

    Depth 2 is intentionally the GUI default: it responds quickly while still
    spotting captures, checks and one-ply tactical replies.
    """

    def __init__(self, depth: int = 2, seed: Optional[int] = None):
        self.depth = max(1, min(4, depth))
        self.rng = random.Random(seed)
        self.nodes = 0

    def choose_move(self, game: XiangqiGame) -> Optional[Move]:
        if game.winner is not None:
            return None
        root = game.turn
        moves = self._ordered_moves(game)
        if not moves:
            return None
        self.nodes = 0
        best_score = -math.inf
        best: list[Move] = []
        for move in moves:
            game.push(move, validate=False)
            score = self._search(game, self.depth - 1, -math.inf, math.inf, root)
            game.undo()
            if score > best_score + 1e-6:
                best_score, best = score, [move]
            elif abs(score - best_score) <= 1e-6:
                best.append(move)
        return self.rng.choice(best)

    def analyze(self, game: XiangqiGame) -> SearchResult:
        move = self.choose_move(game)
        return SearchResult(move, self.evaluate(game, game.turn), self.nodes)

    def _search(self, game: XiangqiGame, depth: int, alpha: float, beta: float, root: str) -> float:
        self.nodes += 1
        if game.winner is not None:
            return 1_000_000.0 if game.winner == root else -1_000_000.0
        if depth <= 0:
            return self.evaluate(game, root)
        moves = self._ordered_moves(game)
        if not moves:
            return self.evaluate(game, root)
        maximizing = game.turn == root
        if maximizing:
            value = -math.inf
            for move in moves:
                game.push(move, validate=False)
                value = max(value, self._search(game, depth - 1, alpha, beta, root))
                game.undo()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        value = math.inf
        for move in moves:
            game.push(move, validate=False)
            value = min(value, self._search(game, depth - 1, alpha, beta, root))
            game.undo()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    @staticmethod
    def _ordered_moves(game: XiangqiGame) -> list[Move]:
        return sorted(
            game.legal_moves(),
            key=lambda move: (VALUES.get(move.captured.kind, 0) if move.captured else 0, -abs(4 - move.end[1])),
            reverse=True,
        )

    @staticmethod
    def evaluate(game: XiangqiGame, perspective: str) -> float:
        score = 0.0
        for row, col, piece in iter_pieces(game.board):
            value = float(VALUES[piece.kind])
            if piece.kind == PAWN:
                crossed = row <= 4 if piece.side == RED else row >= 5
                value += 55 if crossed else 0
                value += (4 - abs(4 - col)) * 3
            elif piece.kind in (HORSE, CANNON):
                value += (4 - abs(4 - col)) * 4
            elif piece.kind == ROOK:
                value += (4 - abs(4 - col)) * 2
            score += value if piece.side == perspective else -value
        if game.is_in_check(opponent(perspective)):
            score += 35
        if game.is_in_check(perspective):
            score -= 45
        return score
