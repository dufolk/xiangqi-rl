import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from game_ai import apply_action, legal_actions, observation
from game_engine import Game


@dataclass(frozen=True)
class StepResult:
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    info: Dict[str, Any]


class SplendorEnv:
    """UI-free environment boundary for self-play and future PPO training."""

    def __init__(self, team_mode: bool = False, target_score: int = 15, seed: Optional[int] = None):
        self.team_mode = team_mode
        self.target_score = 30 if team_mode else target_score
        self.seed = seed
        self.game: Optional[Game] = None

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self.game = Game(
            ai_count=3,
            human_seat=0,
            target_score=self.target_score,
            player_name="训练智能体",
            team_mode=self.team_mode,
            rng=random.Random(self.seed),
        )
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        if self.game is None:
            raise RuntimeError("reset() must be called before observe()")
        return observation(self.game)

    def legal_actions(self) -> List[Dict[str, Any]]:
        if self.game is None:
            raise RuntimeError("reset() must be called before legal_actions()")
        return legal_actions(self.game)

    def step(self, action_id: str) -> StepResult:
        if self.game is None:
            raise RuntimeError("reset() must be called before step()")
        actions = {action["id"] for action in legal_actions(self.game)}
        if action_id not in actions:
            raise ValueError(f"illegal action: {action_id}")

        acting_seat = self.game.turn
        acting_team = acting_seat % 2
        before = self.game.team_score(acting_team) if self.team_mode else self.game.players[acting_seat].points
        apply_action(self.game, action_id)
        after = self.game.team_score(acting_team) if self.team_mode else self.game.players[acting_seat].points
        reward = float(after - before)
        terminated = self.game.phase == "over"
        if terminated:
            won = self.game.winner_team == acting_team if self.team_mode else self.game.winner is self.game.players[acting_seat]
            reward += 10.0 if won else -10.0
        return StepResult(
            observation=self.observe(),
            reward=reward,
            terminated=terminated,
            info={"acting_seat": acting_seat, "acting_team": acting_team if self.team_mode else None},
        )
