"""Self-play evaluation arena for Splendor MAPPO/PPO.

Purpose:
- Measure how a *current* policy performs when seated against opponents
  (other checkpoints, or random baselines). This produces the win-rate curves
  that matter for self-play training stability.
- Supports fictitious self-play style evaluation: seat 0 = current policy,
  seats 1..3 sampled from an opponent pool (historical checkpoints).

Usage (synchronous, CPU, headless):
    arena = SelfPlayArena(team_mode=True, target_score=30, n_games=64, seed=0)
    stats = arena.evaluate(current_state_dict, opponent_pool=[sd1, sd2])
    # stats["win_rate_seat0"], stats["mean_score"], stats["mean_length"] ...
"""

import random
from typing import Dict, List, Optional

import numpy as np
import torch

from .encoders import ACTION_SPACE, OBS_DIM
from .envs import SplendorMAEnv
from .networks import ActorCritic


class SelfPlayArena:
    def __init__(
        self,
        team_mode: bool = False,
        target_score: int = 15,
        n_games: int = 64,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.team_mode = team_mode
        self.target_score = 30 if team_mode else target_score
        self.n_games = n_games
        self.seed = seed
        self.device = torch.device(device)
        self.n_agents = 4

    def _make_actor(self, state_dict: Optional[dict]):
        ac = ActorCritic(OBS_DIM, ACTION_SPACE, self.n_agents,
                         OBS_DIM).to(self.device)
        if state_dict is not None:
            ac.load_state_dict(state_dict)
        ac.eval()
        return ac

    @torch.no_grad()
    def _act(self, ac, obs, agent_id, mask):
        o = torch.as_tensor(obs, device=self.device).float()[None]
        aid = torch.as_tensor([agent_id], device=self.device, dtype=torch.long)
        m = torch.as_tensor(mask, device=self.device).float()[None]
        logits = ac.actor(o, aid, m)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().cpu().numpy()[0])

    @staticmethod
    def _random_action(mask):
        legal = np.where(mask > 0.5)[0]
        return int(np.random.choice(legal)) if len(legal) else 0

    def evaluate(
        self,
        current_state_dict: dict,
        opponent_pool: Optional[List[dict]] = None,
        random_opponents: bool = False,
    ) -> Dict[str, float]:
        """Run `n_games` full games (all 4 seats) and return aggregate stats.

        - seat 0 always uses `current_state_dict`.
        - If `opponent_pool` is given, seats 1..3 are sampled (with replacement)
          from it (fictitious self-play evaluation).
        - If `random_opponents` is True, seats 1..3 play randomly.
        - Otherwise (default) ALL seats use the current policy (pure self-play).
        """
        current_ac = self._make_actor(current_state_dict)
        opp_acs = [self._make_actor(sd) for sd in (opponent_pool or [])]

        seat_wins = [0] * self.n_agents
        team_wins = [0, 0] if self.team_mode else None
        scores_sum = [0.0] * self.n_agents
        lengths = []

        rng = random.Random(self.seed)
        for g in range(self.n_games):
            env = SplendorMAEnv(team_mode=self.team_mode,
                                target_score=self.target_score,
                                seed=self.seed * 1000 + g)
            obs, gstate, mask, info = env.reset()
            agent_memory = [obs.copy() for _ in range(self.n_agents)]
            steps = 0
            done = False
            while not done:
                acting = info["acting_agent"]
                agent_memory[acting] = obs.copy()

                if acting == 0:
                    action = self._act(current_ac, obs, acting, mask)
                elif opp_acs:
                    opp = rng.choice(opp_acs)
                    action = self._act(opp, obs, acting, mask)
                elif random_opponents:
                    action = self._random_action(mask)
                else:
                    action = self._act(current_ac, obs, acting, mask)

                obs, gstate, mask, reward, terminated, info = env.step(action)
                steps += 1
                if terminated:
                    done = True

            game = env.game
            if self.team_mode:
                wteam = game.winner_team
                team_wins[wteam] += 1
                for seat in range(self.n_agents):
                    scores_sum[seat] += game.team_score(seat % 2)
            else:
                winner = game.winner
                wseat = game.players.index(winner)
                seat_wins[wseat] += 1
                for seat in range(self.n_agents):
                    scores_sum[seat] += game.players[seat].points
            lengths.append(steps)

        stats: Dict[str, float] = {}
        n = max(1, self.n_games)
        if self.team_mode:
            stats["win_rate_team0"] = team_wins[0] / n
            stats["win_rate_team1"] = team_wins[1] / n
            stats["mean_team_score"] = sum(scores_sum) / (n * 2)
        else:
            for s in range(self.n_agents):
                stats[f"win_rate_seat{s}"] = seat_wins[s] / n
            stats["win_rate_seat0"] = seat_wins[0] / n
            stats["mean_score"] = sum(scores_sum) / (n * self.n_agents)
        stats["mean_length"] = float(np.mean(lengths)) if lengths else 0.0
        return stats
