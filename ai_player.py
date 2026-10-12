"""Xiangqi opponents used by the desktop app.

``SimpleAI`` is the old material/positional alpha-beta search (kept for tests
and as a fallback). ``PPOAI`` is the production opponent: it loads the trained
PPO actor-critic network and picks moves from the canonical 16×10×9 policy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from game_engine import (
    ADVISOR, CANNON, ELEPHANT, HORSE, KING, PAWN, RED, ROOK,
    Move, XiangqiGame, iter_pieces, opponent,
)

VALUES = {KING: 100_000, ROOK: 900, CANNON: 450, HORSE: 420, ELEPHANT: 210, ADVISOR: 210, PAWN: 100}


def latest_checkpoint(directory: str = "checkpoints") -> Optional[Path]:
    """Return the most recently modified ``*.pt``/``*.pth`` checkpoint, if any."""
    path = Path(directory)
    if not path.is_dir():
        return None
    candidates = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


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


class PPOAI:
    """In-game opponent backed by the trained PPO actor-critic network.

    The network is loaded lazily on the first move request (so the GUI starts
    instantly even before the weights are read) and reused for every later
    decision. Observations and action masks are produced from the canonical
    16×10×9 environment so the model always sees the side-to-move at the
    bottom of the board.

    ``depth`` maps onto three strength levels:
      * 1 (萌新): stochastic sampling with a high temperature (weaker, varied);
      * 2 (进阶): stochastic sampling with a mild temperature;
      * 3 (高手): deterministic greedy play (best policy move).
    """

    def __init__(self, checkpoint: Optional[str] = None, depth: int = 2):
        self.depth = max(1, min(3, depth))
        self.checkpoint_path = Path(checkpoint) if checkpoint else latest_checkpoint()
        self._model = None
        self._device = None
        self.load_error: Optional[str] = None
        self._unavailable = False
        from rl.environment import XiangqiEnv

        self._env = XiangqiEnv()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            import torch
            from rl.model import build_model

            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            model = build_model()
            if self.checkpoint_path and self.checkpoint_path.is_file():
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                if not isinstance(checkpoint, dict) or "model" not in checkpoint:
                    raise ValueError(f"无效的 PPO 检查点：{self.checkpoint_path}")
                model.load_state_dict(checkpoint["model"], strict=True)
            else:
                self.load_error = "未找到训练权重，使用随机初始化的 PPO 网络（走法接近随机）"
            model.to(device).eval()
            self._model = model
            self._device = device
            return True
        except ImportError:
            self._unavailable = True
            self.load_error = "未安装 PyTorch，已改用 Alpha-Beta 搜索 AI"
            return False
        except Exception as exc:  # noqa: BLE001 - surface load failures to the UI
            self._unavailable = True
            self.load_error = f"加载 PPO 模型失败：{exc}，已改用 Alpha-Beta 搜索 AI"
            return False

    def choose_move(self, game: XiangqiGame) -> Optional[Move]:
        if game.winner is not None:
            return None
        if not self._ensure_loaded():
            return self._fallback_move(game)

        import torch
        from torch.distributions import Categorical
        from rl.environment import decode_action, raw_square

        self._env.game = game
        observation = torch.from_numpy(self._env.observe()).unsqueeze(0).to(self._device)
        mask = torch.from_numpy(self._env.action_mask()).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits, _ = self._model(observation, mask)
            logits = logits.squeeze(0)
        if self.depth >= 3:
            action = int(torch.argmax(logits).item())
        else:
            temperature = 1.8 if self.depth == 1 else 1.0
            action = int(Categorical(logits=logits / temperature).sample().item())

        canonical_start, canonical_end = decode_action(action)
        start = raw_square(canonical_start, game.turn)
        end = raw_square(canonical_end, game.turn)
        return next((move for move in game.legal_moves_from(start) if move.end == end), None)

    def _fallback_move(self, game: XiangqiGame) -> Optional[Move]:
        return SimpleAI(depth=min(2, self.depth)).choose_move(game)
