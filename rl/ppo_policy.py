from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class PPOPolicy:
    """Optional PPO inference adapter; training libraries stay outside the game runtime."""

    def __init__(self, checkpoint: str, encoder: Callable[[Dict[str, Any], List[Dict[str, Any]]], Any]):
        self.checkpoint = Path(checkpoint)
        self.encoder = encoder
        self._model: Optional[Any] = None

    def load(self) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("Install optional PPO dependencies before loading a checkpoint") from exc
        self._model = PPO.load(str(self.checkpoint))

    def choose_action(self, observation: Dict[str, Any], legal_actions: List[Dict[str, Any]]) -> str:
        if not legal_actions:
            return ""
        if self._model is None:
            self.load()
        encoded = self.encoder(observation, legal_actions)
        action_index, _ = self._model.predict(encoded, deterministic=True)
        index = int(action_index)
        if index < 0 or index >= len(legal_actions):
            raise ValueError("PPO policy returned an action outside the legal action mask")
        return legal_actions[index]["id"]
