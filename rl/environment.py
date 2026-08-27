"""Minimal dependency-light Xiangqi environment for PPO/self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from game_engine import (
    ADVISOR, BLACK, CANNON, ELEPHANT, HORSE, KING, KINDS, PAWN, RED, ROOK,
    Move, XiangqiGame, action_to_squares, iter_pieces,
)

ACTION_SIZE = 90 * 90
OBSERVATION_PLANES = 16
KIND_INDEX = {kind: index for index, kind in enumerate((KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN))}
MATERIAL = {KING: 0, ROOK: 9.0, CANNON: 4.5, HORSE: 4.2, ELEPHANT: 2.1, ADVISOR: 2.1, PAWN: 1.0}


@dataclass(frozen=True)
class StepResult:
    observation: "object"
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def encode_move(move: Move) -> int:
    return move.action


def decode_action(action: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return action_to_squares(int(action))


class XiangqiEnv:
    """Turn-based self-play environment with an 8,100-action discrete space.

    Observation shape is ``(16, 10, 9)``: fourteen piece planes, side-to-move,
    and the latest destination square. NumPy is imported lazily so the GUI can
    run with only the base requirements installed.
    """

    action_size = ACTION_SIZE
    observation_shape = (OBSERVATION_PLANES, 10, 9)

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

        obs = np.zeros(self.observation_shape, dtype=np.float32)
        for row, col, piece in iter_pieces(self.game.board):
            offset = 0 if piece.side == RED else 7
            obs[offset + KIND_INDEX[piece.kind], row, col] = 1.0
        obs[14, :, :] = 1.0 if self.game.turn == RED else 0.0
        if self.game.last_move is not None:
            row, col = self.game.last_move.end
            obs[15, row, col] = 1.0
        return obs

    def legal_actions(self) -> list[int]:
        return [encode_move(move) for move in self.game.legal_moves()]

    def action_mask(self):
        import numpy as np

        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def step(self, action: int) -> StepResult:
        start, end = decode_action(action)
        legal = next((move for move in self.game.legal_moves_from(start) if move.end == end), None)
        if legal is None:
            raise ValueError(f"illegal action: {action}")
        actor = self.game.turn
        capture_reward = MATERIAL.get(legal.captured.kind, 0.0) / 10.0 if legal.captured else 0.0
        self.game.push(legal)
        self.steps += 1
        terminated = self.game.winner is not None
        truncated = not terminated and self.steps >= self.max_steps
        reward = capture_reward
        if terminated:
            reward += 1.0 if self.game.winner == actor else -1.0
        return StepResult(
            self.observe(), reward, terminated, truncated,
            {"action_mask": self.action_mask(), "actor": actor, "winner": self.game.winner},
        )
