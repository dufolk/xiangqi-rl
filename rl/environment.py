"""Canonical, masked Xiangqi environment for PPO self-play.

Every observation is expressed from the side-to-move's perspective: the acting
side is at the bottom of the board and attacks upward. This makes red/black
mirror positions share one policy representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from game_engine import (
    ADVISOR, BLACK, CANNON, COLS, ELEPHANT, HORSE, KING, PAWN, RED, ROOK, ROWS,
    Move, XiangqiGame, action_to_squares, iter_pieces, square_index,
)

ACTION_SIZE = ROWS * COLS * ROWS * COLS
OBSERVATION_PLANES = 16
PIECE_ORDER = (KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN)
KIND_INDEX = {kind: index for index, kind in enumerate(PIECE_ORDER)}
OWN_CHECK_PLANE = 14
OPPONENT_CHECK_PLANE = 15


@dataclass(frozen=True)
class StepResult:
    observation: "object"
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def rotate_square(square: tuple[int, int]) -> tuple[int, int]:
    """Rotate a raw board square by 180 degrees."""
    row, col = square
    return ROWS - 1 - row, COLS - 1 - col


def canonical_square(square: tuple[int, int], side_to_move: str) -> tuple[int, int]:
    """Map a raw square into the current player's canonical viewpoint."""
    return square if side_to_move == RED else rotate_square(square)


def raw_square(square: tuple[int, int], side_to_move: str) -> tuple[int, int]:
    """Map a canonical square back into the rules engine's raw coordinates."""
    return square if side_to_move == RED else rotate_square(square)


def encode_move(move: Move, side_to_move: str = RED) -> int:
    """Encode a raw move as an action in ``side_to_move``'s canonical view."""
    start = canonical_square(move.start, side_to_move)
    end = canonical_square(move.end, side_to_move)
    return square_index(*start) * (ROWS * COLS) + square_index(*end)


def decode_action(action: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Decode an action into canonical start/end squares."""
    return action_to_squares(int(action))


class XiangqiEnv:
    """Self-play environment with a 16×10×9 canonical observation.

    Planes 0–6 contain the acting side's seven piece types, planes 7–13 the
    opponent's types, and planes 14–15 are current-position check indicators.
    The action space remains a fixed 90×90 grid (8,100 slots), protected by a
    legal-action mask. Rewards are terminal-only to avoid greedy material play.
    """

    action_size = ACTION_SIZE
    observation_shape = (OBSERVATION_PLANES, ROWS, COLS)

    def __init__(self, max_steps: int = 400, start_fen: Optional[str] = None):
        self.max_steps = max_steps
        self.start_fen = start_fen
        self.game = XiangqiGame()
        self.steps = 0

    def reset(self, seed: Optional[int] = None) -> tuple["object", dict[str, object]]:
        del seed
        self.game = XiangqiGame.from_fen(self.start_fen) if self.start_fen else XiangqiGame()
        self.steps = 0
        return self.observe(), {"action_mask": self.action_mask()}

    def observe(self):
        import numpy as np

        side = self.game.turn
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        for row, col, piece in iter_pieces(self.game.board):
            canonical_row, canonical_col = canonical_square((row, col), side)
            offset = 0 if piece.side == side else len(PIECE_ORDER)
            obs[offset + KIND_INDEX[piece.kind], canonical_row, canonical_col] = 1.0
        if self.game.is_in_check(side):
            obs[OWN_CHECK_PLANE, :, :] = 1.0
        if self.game.is_in_check(BLACK if side == RED else RED):
            obs[OPPONENT_CHECK_PLANE, :, :] = 1.0
        return obs

    def legal_actions(self) -> list[int]:
        return [encode_move(move, self.game.turn) for move in self.game.legal_moves()]

    def action_mask(self):
        import numpy as np

        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def step(self, action: int) -> StepResult:
        actor = self.game.turn
        canonical_start, canonical_end = decode_action(action)
        start = raw_square(canonical_start, actor)
        end = raw_square(canonical_end, actor)
        legal = next((move for move in self.game.legal_moves_from(start) if move.end == end), None)
        if legal is None:
            raise ValueError(f"illegal canonical action: {action}")

        played = self.game.push(legal)
        self.steps += 1
        terminated = self.game.winner is not None
        truncated = not terminated and self.steps >= self.max_steps
        reward = 1.0 if terminated and self.game.winner == actor else 0.0
        gives_check = not terminated and self.game.is_in_check(self.game.turn)
        action_mask = self.action_mask()
        return StepResult(
            self.observe(), reward, terminated, truncated,
            {
                "action_mask": action_mask,
                "actor": actor,
                "winner": self.game.winner,
                "reason": self.game.reason,
                "move": played,
                "capture": played.captured is not None,
                "captured_kind": played.captured.kind if played.captured else None,
                "gives_check": gives_check,
                "legal_action_count": int(action_mask.sum()),
            },
        )
