"""Historical checkpoint pool and round-robin Elo tracking for MAPPO training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .selfplay import SelfPlayArena


def _cpu_state_dict(state_dict: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def elo_has_converged(latest_elos: List[float], window: int, threshold: float) -> bool:
    """Return whether the newest model's Elo has settled into a narrow band."""
    if window < 2 or len(latest_elos) < window:
        return False
    recent = latest_elos[-window:]
    return max(recent) - min(recent) <= threshold


class HistoryManager:
    """Persist timed snapshots, expose a bounded opponent pool, and track Elo."""

    def __init__(
        self,
        directory: str,
        team_mode: bool,
        target_score: int,
        snapshot_every_seconds: float = 600.0,
        snapshot_every_steps: int = 0,
        opponent_pool_size: int = 32,
        elo_games_per_match: int = 4,
        elo_k_factor: float = 24.0,
        elo_pool_size: int = 40,
        convergence_window: int = 6,
        convergence_threshold: float = 25.0,
        seed: int = 0,
    ):
        if snapshot_every_seconds < 0:
            raise ValueError("snapshot_every_seconds must be non-negative")
        if opponent_pool_size < 1:
            raise ValueError("opponent_pool_size must be positive")
        if elo_games_per_match < 1:
            raise ValueError("elo_games_per_match must be positive")
        if elo_pool_size < 2:
            raise ValueError("elo_pool_size must be at least 2")
        if convergence_window < 0:
            raise ValueError("convergence_window must be non-negative")
        if convergence_threshold < 0:
            raise ValueError("convergence_threshold must be non-negative")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.team_mode = team_mode
        self.target_score = target_score
        self.snapshot_every_seconds = float(snapshot_every_seconds)
        self.snapshot_every_steps = int(snapshot_every_steps)
        self._last_snapshot_steps = 0
        self.opponent_pool_size = int(opponent_pool_size)
        self.elo_games_per_match = int(elo_games_per_match)
        self.elo_k_factor = float(elo_k_factor)
        self.elo_pool_size = int(elo_pool_size)
        self.convergence_window = int(convergence_window)
        self.convergence_threshold = float(convergence_threshold)
        self.seed = int(seed)
        self._last_snapshot_at = time.monotonic()
        self._latest_elo_history = self._load_elo_history()
        self._latest_pool_scores = self._load_pool_scores()

    @property
    def latest_elo_history(self) -> List[float]:
        """Newest-model Elo after each snapshot, in snapshot order."""
        return list(self._latest_elo_history)

    @property
    def latest_pool_scores(self) -> List[float]:
        """Newest-model win rate against the historical pool, in snapshot order."""
        return list(self._latest_pool_scores)

    @property
    def converged(self) -> bool:
        return elo_has_converged(
            self._latest_elo_history,
            self.convergence_window,
            self.convergence_threshold,
        )

    def convergence_span(self):
        if self.convergence_window < 2 or len(self._latest_elo_history) < self.convergence_window:
            return None
        recent = self._latest_elo_history[-self.convergence_window :]
        return max(recent) - min(recent)

    def snapshot_paths(self) -> List[Path]:
        return sorted(self.directory.glob("model_*.pt"))

    def opponent_pool(self) -> List[dict]:
        """Return the newest bounded subset of historical policies on CPU."""
        paths = self.snapshot_paths()[-self.opponent_pool_size :]
        return [self._load_state_dict(path) for path in paths]

    def maybe_snapshot(self, state_dict: dict, collected_steps: int) -> bool:
        """Save and rate a model after the step interval and/or wall-clock interval.

        Either trigger alone is sufficient. `snapshot_every_steps=0` disables the
        step trigger (wall-clock only); `snapshot_every_seconds=0` disables the
        wall-clock trigger (step only).
        """
        step_trigger = (
            self.snapshot_every_steps > 0
            and collected_steps - self._last_snapshot_steps >= self.snapshot_every_steps
        )
        time_trigger = (
            self.snapshot_every_seconds > 0
            and (time.monotonic() - self._last_snapshot_at) >= self.snapshot_every_seconds
        )
        if not step_trigger and not time_trigger:
            return False
        self._last_snapshot_at = time.monotonic()
        self._last_snapshot_steps = collected_steps
        self.snapshot(state_dict, collected_steps)
        return True

    def snapshot(self, state_dict: dict, collected_steps: int) -> Path:
        """Persist one checkpoint, recompute all historical Elo scores, and plot them."""
        ordinal = len(self.snapshot_paths()) + 1
        path = self.directory / f"model_{ordinal:05d}_step_{int(collected_steps):012d}.pt"
        torch.save(_cpu_state_dict(state_dict), path)
        ratings, latest_elo, latest_pool_score = self.refresh_elo()
        if latest_elo is not None:
            self._latest_elo_history.append(float(latest_elo))
            self._latest_pool_scores.append(float(latest_pool_score))
            self._write_elo(ratings)
        return path

    def refresh_elo(self) -> Tuple[Dict[str, float], Optional[float], float]:
        """Run a round robin over the newest checkpoints.

        Returns every persisted checkpoint's Elo (older ones keep their last
        known rating), the newest model's Elo, and its win rate against the
        historical pool. The newest Elo stays None until a second checkpoint
        exists, because Elo needs at least one opponent to mean anything.
        """
        paths = self.snapshot_paths()
        rated_paths = paths[-self.elo_pool_size :]
        models: List[Tuple[str, dict]] = [(path.stem, self._load_state_dict(path)) for path in rated_paths]
        ratings = {name: 1000.0 for name, _ in models}
        latest_elo: Optional[float] = None
        latest_pool_score = 0.0
        if len(models) > 1:
            arena = SelfPlayArena(
                team_mode=self.team_mode,
                target_score=self.target_score,
                n_games=self.elo_games_per_match,
                seed=self.seed,
            )
            for index, (name_a, state_a) in enumerate(models):
                for pair_index, (name_b, state_b) in enumerate(models[index + 1 :], start=index + 1):
                    score_a = arena.match_score(
                        state_a,
                        state_b,
                        seed_offset=index * len(models) + pair_index,
                    )
                    expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[name_b] - ratings[name_a]) / 400.0))
                    delta = self.elo_k_factor * (score_a - expected_a)
                    ratings[name_a] += delta
                    ratings[name_b] -= delta
            latest_name, latest_state = models[-1]
            latest_elo = ratings[latest_name]
            opponents = [state for _, state in models[:-1]]
            latest_pool_score = arena.pool_score(latest_state, opponents)
        ordered = self._merge_stale_ratings(paths, ratings)
        self._write_elo(ordered)
        self._write_plot(ordered)
        return ordered, latest_elo, latest_pool_score

    def _merge_stale_ratings(self, paths: List[Path], ratings: Dict[str, float]) -> Dict[str, float]:
        """Return ratings for every checkpoint, reusing old values outside the round robin."""
        if len(paths) <= self.elo_pool_size:
            return ratings
        previous = self._previous_ratings()
        return {
            path.stem: ratings.get(path.stem, previous.get(path.stem, 1000.0))
            for path in paths
        }

    def _previous_ratings(self) -> Dict[str, float]:
        path = self.directory / "elo.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {name: float(score) for name, score in payload.get("elo", {}).items()}
        except (OSError, ValueError, TypeError):
            return {}

    def _load_state_dict(self, path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_elo_history(self) -> List[float]:
        return self._load_history_list("latest_elo_history")

    def _load_pool_scores(self) -> List[float]:
        return self._load_history_list("latest_pool_scores")

    def _load_history_list(self, key: str) -> List[float]:
        path = self.directory / "elo.json"
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [float(value) for value in payload.get(key, [])]
        except (OSError, ValueError, TypeError):
            return []

    def _write_elo(self, ratings: Dict[str, float]) -> None:
        payload = {
            "mode": "team" if self.team_mode else "standard",
            "updated_at": time.time(),
            "elo": {name: round(score, 3) for name, score in ratings.items()},
            "latest_elo_history": [round(value, 3) for value in self._latest_elo_history],
            "latest_pool_scores": [round(value, 4) for value in self._latest_pool_scores],
            "convergence": {
                "window": self.convergence_window,
                "threshold": self.convergence_threshold,
                "span": self.convergence_span(),
                "converged": self.converged,
            },
        }
        (self.directory / "elo.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def elo_plot_path(self) -> Optional[Path]:
        """Path to the latest Elo bar-chart PNG (or None before the first snapshot)."""
        path = self.directory / "elo.png"
        return path if path.is_file() else None

    def _write_plot(self, ratings: Dict[str, float]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(ratings)
        values = [ratings[name] for name in names]
        fig_width = max(8.0, len(names) * 0.85)
        figure, axis = plt.subplots(figsize=(fig_width, 5.5))
        bars = axis.bar(range(len(names)), values, color="#3b82f6")
        axis.axhline(1000, color="#64748b", linewidth=1, linestyle="--")
        axis.set_ylabel("Elo")
        axis.set_title("Historical checkpoint Elo")
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.bar_label(bars, labels=[f"{value:.0f}" for value in values], padding=3, fontsize=8)
        figure.tight_layout()
        figure.savefig(self.directory / "elo.png", dpi=160)
        plt.close(figure)
