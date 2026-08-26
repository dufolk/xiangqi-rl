"""Multi-agent (decentralized) environment wrapper for Splendor MAPPO.

Splendor is a *turn-based* game: exactly one of the 4 seats acts at a time.
MAPPO (CTDE) treats each seat as an agent. We model a single episode as a
sequence of *agent-steps*: at each game turn the acting seat observes the
global state, picks an action, and receives a shaped reward. The centralized
critic consumes the global state; decentralized actors consume the same
encoded observation (the global observation is agent-agnostic here, which is
fine for CTDE and simplifies the network; privacy masking can be added later).

Public API used by the collector / trainer:
    env = SplendorMAEnv(team_mode=..., target_score=..., seed=...)
    obs, gstate, mask, info = env.reset()
    # info contains 'acting_agent' (seat)
    obs, gstate, mask, reward, done, info = env.step(slot: int)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from game_ai import legal_actions, observation
from game_engine import Game

from .encoders import (
    ACTION_SPACE,
    OBS_DIM,
    build_action_mask,
    encode_observation,
    slot_to_action_id,
)


class SplendorMAEnv:
    def __init__(
        self,
        team_mode: bool = False,
        target_score: int = 15,
        seed: Optional[int] = None,
        n_players: int = 4,
    ):
        self.team_mode = team_mode
        self.target_score = 30 if team_mode else target_score
        self.base_seed = seed
        self.n_players = n_players
        self.game: Optional[Game] = None
        self._episode_seed = seed

    # -- gym-ish helpers --------------------------------------------------
    @property
    def observation_space_dim(self) -> int:
        return OBS_DIM

    @property
    def action_space_dim(self) -> int:
        return ACTION_SPACE

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._episode_seed = seed
        rng_seed = self._episode_seed
        if rng_seed is None:
            rng_seed = int(np.random.randint(0, 2**31 - 1))
        self.game = Game(
            ai_count=self.n_players - 1,
            human_seat=0,
            target_score=self.target_score,
            player_name="mappo",
            team_mode=self.team_mode,
            rng=__import__("random").Random(rng_seed),
        )
        self._last_reward = 0.0
        return self._observe()

    def _observe(self):
        obs = observation(self.game)
        local = encode_observation(obs)            # actor view (global encoded)
        gstate = local.copy()                       # critic uses same encoded global
        legal = legal_actions(self.game)
        mask = build_action_mask(legal)
        acting = self.game.turn
        info = {
            "acting_agent": acting,
            "legal": legal,
            "phase": self.game.phase,
            "team_mode": self.team_mode,
        }
        return local, gstate, mask, info

    def step(self, slot: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        """Advance one agent-step using a chosen action *slot*.

        Returns the next observation (for the new acting agent), the global
        state, the new legal mask, the reward for the agent that just acted,
        a done flag (episode over), and info.
        """
        if self.game is None:
            raise RuntimeError("reset() before step()")
        legal = legal_actions(self.game)
        action_id = slot_to_action_id(slot, legal)
        acting_seat = self.game.turn
        acting_team = acting_seat % 2 if self.team_mode else None

        before = (
            self.game.team_score(acting_team)
            if self.team_mode
            else self.game.players[acting_seat].points
        )
        # apply via game_ai.apply_action
        from game_ai import apply_action
        apply_action(self.game, action_id)
        after = (
            self.game.team_score(acting_team)
            if self.team_mode
            else self.game.players[acting_seat].points
        )
        reward = float(after - before)

        terminated = self.game.phase == "over"
        terminal_rewards = [0.0] * self.n_players
        if terminated:
            if self.team_mode:
                terminal_rewards = [
                    10.0 if seat % 2 == self.game.winner_team else -10.0
                    for seat in range(self.n_players)
                ]
            else:
                winner_seat = self.game.players.index(self.game.winner)
                terminal_rewards = [10.0 if seat == winner_seat else -10.0 for seat in range(self.n_players)]

        local, gstate, mask, info = self._observe()
        info["terminal_rewards"] = terminal_rewards
        return local, gstate, mask, reward, terminated, info


def make_env(team_mode: bool, target_score: int, seed: Optional[int]):
    return SplendorMAEnv(team_mode=team_mode, target_score=target_score, seed=seed)
