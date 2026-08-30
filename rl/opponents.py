"""Frozen neural-network opponent for PPO training.

Loads a fixed checkpoint once (lazily inside each worker process) and plays the
side to move via the same canonical encoding the agent uses. Unlike
``SearchOpponent`` this opponent has real playing strength, so it is suitable for
the "train against a fixed latest-model teacher" curriculum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ModelOpponent:
    """Plays a fixed checkpoint model; never updated during training."""

    def __init__(self, model_path: Path, deterministic: bool = True, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.device = device
        self.model = None  # loaded lazily (and per-worker after unpickling)
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from rl.model import build_model

        if not self.model_path.is_file():
            raise FileNotFoundError(f"opponent model not found: {self.model_path}")
        self.model = build_model().to(self.device)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:  # older PyTorch
            checkpoint = torch.load(self.model_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"invalid Xiangqi PPO checkpoint: {self.model_path}")
        if checkpoint.get("policy_format") != "autoregressive_cross_attention_pointer_v1":
            raise ValueError(
                f"opponent checkpoint does not use the autoregressive pointer policy: {self.model_path}"
            )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()

    # The torch model is not picklable across spawn (nested class). Drop it and
    # reload from the path inside the worker process instead.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model()

    def select_move(self, game) -> Optional["object"]:
        """Return the best move for the side to move, or ``None`` if none exist."""
        import numpy as np
        import torch
        from game_engine import BLACK, RED, iter_pieces

        from rl.environment import (
            ACTION_SIZE, OWN_CHECK_PLANE, OPPONENT_CHECK_PLANE, PIECE_ORDER,
            KIND_INDEX, ROWS, COLS, canonical_square, raw_square,
            decode_action, encode_move, build_observation,
        )

        if self.model is None:
            self._load_model()

        side = game.turn
        obs = build_observation(game)
        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        for move in game.legal_moves():
            mask[encode_move(move, side)] = True

        with torch.no_grad():
            action, _, _ = self.model.act(
                torch.as_tensor(obs).unsqueeze(0),
                torch.as_tensor(mask).unsqueeze(0),
                deterministic=self.deterministic,
            )
        action = int(action[0])
        canonical_start, canonical_end = decode_action(action)
        start = raw_square(canonical_start, side)
        end = raw_square(canonical_end, side)
        return next((move for move in game.legal_moves_from(start) if move.end == end), None)
