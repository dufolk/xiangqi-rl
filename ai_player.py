"""Xiangqi opponents used by the desktop app.

``PikafishAI`` is the production opponent: it drives the Pikafish UCI engine
(NNUE, Stockfish lineage) as a subprocess. ``PPOAI`` plays the trained PPO
actor-critic network and is used when the engine binary is not installed.
``SimpleAI`` is the old material/positional alpha-beta search, kept for tests
and as the last-resort fallback.
"""

from __future__ import annotations

import math
import os
import platform
import queue
import random
import shutil
import subprocess
import threading
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


ENGINE_DIR = Path(__file__).resolve().parent / "engines" / "pikafish"
PIKAFISH_BINARIES = {
    "Windows": "Pikafish-Windows-x86-64-universal.exe",
    "Darwin": "Pikafish-MacOS-universal",
    "Linux": "Pikafish-Linux-x86-64-universal",
}
MATE_SCORE = 30_000
FILES = "abcdefghi"


def find_pikafish() -> Optional[Path]:
    """Locate the engine: ``$PIKAFISH_PATH``, then ``engines/pikafish/``, then ``PATH``."""
    env = os.environ.get("PIKAFISH_PATH")
    if env:
        return Path(env) if Path(env).is_file() else None
    preferred = PIKAFISH_BINARIES.get(platform.system())
    if preferred and (ENGINE_DIR / preferred).is_file():
        return ENGINE_DIR / preferred
    if ENGINE_DIR.is_dir():
        suffix = ".exe" if platform.system() == "Windows" else ""
        for candidate in sorted(ENGINE_DIR.iterdir()):
            if candidate.name.lower().startswith("pikafish") and candidate.suffix == suffix and candidate.is_file():
                return candidate
    found = shutil.which("pikafish")
    return Path(found) if found else None


def square_to_uci(square: tuple[int, int]) -> str:
    row, col = square
    return f"{FILES[col]}{9 - row}"


def uci_to_squares(text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(text) != 4 or text[0] not in FILES or text[2] not in FILES or not (text[1] + text[3]).isdigit():
        raise ValueError(f"invalid UCI move: {text!r}")
    return (9 - int(text[1]), FILES.index(text[0])), (9 - int(text[3]), FILES.index(text[2]))


def move_to_uci(move: Move) -> str:
    return square_to_uci(move.start) + square_to_uci(move.end)


def position_command(game: XiangqiGame) -> str:
    """``position fen <root> moves ...`` so the engine sees repetitions and the 60-move rule.

    The history is replayed first; if the board was edited mid-game and the
    replay no longer reaches the current position, only the current FEN is sent.
    """
    moves = [snap.last_move for snap in game.history[1:]] + [game.last_move]
    if game.history and all(moves):
        root = game.history[0]
        replay = XiangqiGame(root.board, root.turn)
        try:
            for move in moves:
                replay.push(move, validate=False)
        except ValueError:
            replay = None
        if replay is not None and replay.board == game.board and replay.turn == game.turn:
            start = XiangqiGame(root.board, root.turn).to_fen()
            return f"position fen {start} - - 0 1 moves " + " ".join(move_to_uci(move) for move in moves)
    return f"position fen {game.to_fen()} - - 0 1"


@dataclass(frozen=True)
class EngineLevel:
    name: str
    go: str
    multipv: int = 1
    temperature: float = 0.0
    window: int = 0


@dataclass(frozen=True)
class EngineEval:
    """Latest search summary; ``score`` is centipawns from red's point of view."""

    score: int
    mate: Optional[int]
    depth: int
    move: str

    def label(self) -> str:
        if self.mate is not None:
            return f"红方 {abs(self.mate)} 步杀" if self.mate > 0 else f"黑方 {abs(self.mate)} 步杀"
        return f"评估 {self.score / 100:+.1f}"


class PikafishAI:
    """Opponent backed by the Pikafish UCI engine.

    Pikafish has no ``Skill Level`` option, so the weaker levels search
    shallowly with MultiPV and sample among the near-best lines (softmax over
    the centipawn gap, lines further than ``window`` behind are dropped); the
    strong levels play the engine's best move under a time budget.
    """

    LEVELS = (
        EngineLevel("萌新", "go depth 1", multipv=6, temperature=160, window=450),
        EngineLevel("入门", "go depth 4", multipv=4, temperature=70, window=250),
        EngineLevel("进阶", "go depth 8", multipv=3, temperature=25, window=100),
        EngineLevel("高手", "go movetime 800"),
        EngineLevel("大师", "go movetime 2500"),
    )

    def __init__(self, path: Optional[str] = None, depth: int = 4, fallback=None,
                 threads: Optional[int] = None, hash_mb: int = 64, seed: Optional[int] = None):
        self.depth = max(1, min(len(self.LEVELS), depth))
        self.path = Path(path) if path else find_pikafish()
        self.fallback = fallback
        self.threads = threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        self.hash_mb = hash_mb
        self.rng = random.Random(seed)
        self.load_error: Optional[str] = None
        self.last_eval: Optional[EngineEval] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lines: Optional[queue.Queue[Optional[str]]] = None
        self._broken = False
        self._lock = threading.Lock()

    @property
    def level(self) -> EngineLevel:
        return self.LEVELS[self.depth - 1]

    @property
    def available(self) -> bool:
        return self.path is not None and not self._broken

    @property
    def name(self) -> str:
        return "Pikafish" if self.available else "PPO" if isinstance(self.fallback, PPOAI) else "AI"

    def cycle_level(self) -> EngineLevel:
        self.depth = self.depth % len(self.LEVELS) + 1
        return self.level

    def choose_move(self, game: XiangqiGame) -> Optional[Move]:
        if game.winner is not None:
            return None
        if self.available:
            try:
                with self._lock:
                    move = self._search(game)
                if move is not None:
                    return move
            except (OSError, TimeoutError, ValueError) as exc:
                self._broken = True
                self.close()
                self.load_error = f"Pikafish 引擎异常：{exc}，已改用备用 AI"
        elif self.load_error is None:
            self.load_error = "未找到 Pikafish 引擎（运行 python tools/fetch_pikafish.py 下载），已改用备用 AI"
        return self._fallback_move(game)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            proc.kill()
            proc.wait()
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except OSError:
                pass

    def _fallback_move(self, game: XiangqiGame) -> Optional[Move]:
        if self.fallback is not None:
            self.fallback.depth = min(3, self.depth)
            move = self.fallback.choose_move(game)
            error = getattr(self.fallback, "load_error", None)
            if error and self.load_error and error not in self.load_error:
                self.load_error += f"；{error}"
            return move
        return SimpleAI(depth=2).choose_move(game)

    def _send(self, command: str) -> None:
        assert self._proc is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def _read(self, timeout: float) -> str:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Pikafish 响应超时") from None
        if line is None:
            raise OSError("Pikafish 进程已退出")
        return line

    def _wait_for(self, token: str, timeout: float) -> None:
        while not self._read(timeout).startswith(token):
            pass

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [str(self.path)], cwd=str(self.path.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, creationflags=flags,
        )
        lines: queue.Queue[Optional[str]] = queue.Queue()
        self._lines = lines
        stdout = self._proc.stdout

        def pump() -> None:
            try:
                for raw in stdout:
                    lines.put(raw.strip())
            except (OSError, ValueError):
                pass
            lines.put(None)

        threading.Thread(target=pump, name="pikafish-reader", daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", 10)
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        nnue = self.path.parent / "pikafish.nnue"
        if nnue.is_file():
            self._send(f"setoption name EvalFile value {nnue}")
        self._send("isready")
        self._wait_for("readyok", 30)
        self.load_error = None

    def _search(self, game: XiangqiGame) -> Optional[Move]:
        self._ensure_started()
        level = self.level
        self._send(f"setoption name MultiPV value {level.multipv}")
        self._send(position_command(game))
        self._send(level.go)
        lines: dict[int, tuple[int, int, Optional[int], str]] = {}
        best = ""
        while True:
            line = self._read(60)
            if line.startswith("bestmove"):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else ""
                break
            parsed = _parse_info(line)
            if parsed is not None:
                index, depth, score, mate, move = parsed
                lines[index] = (depth, score, mate, move)
        if best in ("", "(none)", "0000"):
            return None
        choice = self._sample(lines, level) if level.multipv > 1 else best
        top = lines.get(1)
        if top is not None:
            sign = 1 if game.turn == RED else -1
            mate = top[2] * sign if top[2] is not None else None
            self.last_eval = EngineEval(top[1] * sign, mate, top[0], choice)
        start, end = uci_to_squares(choice)
        return next((move for move in game.legal_moves_from(start) if move.end == end), None)

    def _sample(self, lines: dict[int, tuple[int, int, Optional[int], str]], level: EngineLevel) -> str:
        depth = max(entry[0] for entry in lines.values())
        ranked = [entry for _, entry in sorted(lines.items()) if entry[0] == depth] or list(lines.values())
        top = max(entry[1] for entry in ranked)
        pool = [entry for entry in ranked if top - entry[1] <= level.window]
        weights = [math.exp((entry[1] - top) / level.temperature) for entry in pool]
        return self.rng.choices(pool, weights=weights)[0][3]


def _parse_info(line: str) -> Optional[tuple[int, int, int, Optional[int], str]]:
    """Parse ``info ... multipv k score cp|mate x ... pv m ...`` into (k, depth, score, mate, move)."""
    if not line.startswith("info ") or " pv " not in line or " score " not in line:
        return None
    tokens = line.split()
    if "upperbound" in tokens or "lowerbound" in tokens:
        return None
    try:
        depth = int(tokens[tokens.index("depth") + 1])
        index = int(tokens[tokens.index("multipv") + 1]) if "multipv" in tokens else 1
        kind, value = tokens[tokens.index("score") + 1], int(tokens[tokens.index("score") + 2])
        move = tokens[tokens.index("pv") + 1]
    except (ValueError, IndexError):
        return None
    if kind == "mate":
        return index, depth, (MATE_SCORE - abs(value)) * (1 if value > 0 else -1), value, move
    return index, depth, value, None, move
