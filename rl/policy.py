from typing import Any, Dict, List, Protocol


class Policy(Protocol):
    """Common contract for heuristic, remote-model, and future PPO policies."""

    def choose_action(self, observation: Dict[str, Any], legal_actions: List[Dict[str, Any]]) -> str:
        """Return one action id from legal_actions."""
        ...
