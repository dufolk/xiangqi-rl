"""Actor-Critic networks for MAPPO (CTDE).

- Actor: decentralized. Each agent has its own Actor head (or shares weights).
  Here we use *one shared Actor* across agents for sample efficiency, with an
  agent-id one-hot appended so agents can differentiate roles. Input is the
  (privacy-agnostic) encoded observation; output is a masked action distribution.
- Critic: centralized. Consumes the *joint global state* (concatenation of all
  agents' encoded observations) and outputs a single state value V(s) shared by
  all agents (standard MAPPO centralized critic).

All networks are torch nn.Modules and run on CUDA when available (RTX 5070).
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: List[int], activation=nn.ReLU, final_activation=None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Shared actor with agent-id conditioning. Masked categorical output."""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, hidden: List[int] = None):
        super().__init__()
        self.n_agents = n_agents
        if hidden is None:
            hidden = [256, 256, 128]
        self.agent_embed = nn.Embedding(n_agents, n_agents)
        layers = [obs_dim + n_agents] + hidden + [action_dim]
        self.net = _mlp(layers)

    def forward(self, obs: torch.Tensor, agent_id: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        obs: (B, obs_dim)
        agent_id: (B,) long
        mask: (B, action_dim) float (1=legal, 0=illegal)
        returns logits (B, action_dim)
        """
        aid = self.agent_embed(agent_id)            # (B, n_agents)
        x = torch.cat([obs, aid], dim=-1)
        logits = self.net(x)
        if mask is not None:
            # mask out illegal actions; use large negative logit
            logits = logits.masked_fill(mask < 0.5, -1e9)
        return logits


class CentralizedCritic(nn.Module):
    """Centralized critic: joint global state -> scalar value."""

    def __init__(self, global_dim: int, hidden: List[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [512, 256, 128]
        layers = [global_dim] + hidden + [1]
        self.net = _mlp(layers)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        global_state: (B, global_dim)  -- concatenation of all agents' obs
        returns: (B, 1)
        """
        return self.net(global_state)


class ActorCritic(nn.Module):
    """Bundle for convenience (shared actor + centralized critic)."""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, global_dim: int):
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, n_agents)
        self.critic = CentralizedCritic(global_dim)

    def act(self, obs, agent_id, mask, deterministic=False):
        with torch.no_grad():
            logits = self.actor(obs, agent_id, mask)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            logp = dist.log_prob(action)
        return action, logp, dist

    def evaluate(self, global_state, obs, agent_id, mask, action):
        logits = self.actor(obs, agent_id, mask)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(global_state).squeeze(-1)
        return logp, entropy, value, dist


def global_state_from_obs_list(obs_list: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-agent encoded observations into a joint global state.

    obs_list: list of (B, obs_dim) tensors, one per agent
    returns: (B, obs_dim * n_agents)
    """
    return torch.cat(obs_list, dim=-1)
