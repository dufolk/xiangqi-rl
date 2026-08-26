"""Trajectory-safe rollout utilities for MAPPO.

GAE is computed independently for every ``(worker rollout, episode, seat)``
trajectory. Non-terminal horizon cuts bootstrap from the critic value of the
next authoritative state; true episode ends bootstrap with zero.
"""

from typing import Dict, Tuple

import numpy as np
import torch


def compute_grouped_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    agent_ids: np.ndarray,
    episode_ids: np.ndarray,
    bootstrap_values: Dict[Tuple[int, int], float],
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """Compute GAE without crossing episode or agent trajectory boundaries."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    agent_ids = np.asarray(agent_ids, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if not (rewards.shape == values.shape == agent_ids.shape == episode_ids.shape):
        raise ValueError("GAE arrays must have identical one-dimensional shapes")

    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    for episode_id in np.unique(episode_ids):
        episode_mask = episode_ids == episode_id
        for agent_id in np.unique(agent_ids[episode_mask]):
            indices = np.flatnonzero(episode_mask & (agent_ids == agent_id))
            next_value = float(bootstrap_values.get((int(episode_id), int(agent_id)), 0.0))
            next_advantage = 0.0
            for index in indices[::-1]:
                delta = rewards[index] + gamma * next_value - values[index]
                next_advantage = delta + gamma * lam * next_advantage
                advantages[index] = next_advantage
                returns[index] = next_advantage + values[index]
                next_value = float(values[index])
    return advantages, returns


def concatenate_rollouts(payloads):
    """Compute per-payload GAE first, then concatenate independent rollouts."""
    fields = ("obs", "global_state", "agent_id", "action", "logp", "mask")
    merged = {field: [] for field in fields}
    merged["advantages"] = []
    merged["returns"] = []

    for payload in payloads:
        bootstrap = {
            (int(episode), int(agent)): float(value)
            for episode, values in payload.get("bootstrap_values", {}).items()
            for agent, value in enumerate(values)
        }
        advantages, returns = compute_grouped_gae(
            payload["reward"], payload["value"], payload["agent_id"],
            payload["episode_id"], bootstrap,
        )
        for field in fields:
            merged[field].append(payload[field])
        merged["advantages"].append(advantages)
        merged["returns"].append(returns)

    return {field: np.concatenate(parts) for field, parts in merged.items()}


def to_tensors(batch, device):
    tensors = {
        "obs": torch.as_tensor(np.ascontiguousarray(batch["obs"]), device=device),
        "global_state": torch.as_tensor(np.ascontiguousarray(batch["global_state"]), device=device),
        "agent_id": torch.as_tensor(np.ascontiguousarray(batch["agent_id"]), device=device, dtype=torch.long),
        "action": torch.as_tensor(np.ascontiguousarray(batch["action"]), device=device, dtype=torch.long),
        "old_logp": torch.as_tensor(np.ascontiguousarray(batch["logp"]), device=device),
        "mask": torch.as_tensor(np.ascontiguousarray(batch["mask"]), device=device),
        "advantages": torch.as_tensor(np.ascontiguousarray(batch["advantages"]), device=device),
        "returns": torch.as_tensor(np.ascontiguousarray(batch["returns"]), device=device),
    }
    advantages = tensors["advantages"]
    if advantages.numel() > 1:
        tensors["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return tensors


class RolloutBuffer:
    """Compatibility wrapper for single-trajectory callers."""

    def __init__(self, obs_dim: int, global_dim: int, action_dim: int, capacity: int = 4096):
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.cap = capacity
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.cap, self.obs_dim), dtype=np.float32)
        self.global_state = np.zeros((self.cap, self.global_dim), dtype=np.float32)
        self.agent_id = np.zeros(self.cap, dtype=np.int64)
        self.action = np.zeros(self.cap, dtype=np.int64)
        self.logp = np.zeros(self.cap, dtype=np.float32)
        self.reward = np.zeros(self.cap, dtype=np.float32)
        self.mask = np.zeros((self.cap, self.action_dim), dtype=np.float32)
        self.value = np.zeros(self.cap, dtype=np.float32)
        self.done = np.zeros(self.cap, dtype=bool)
        self.ptr = self.size = 0

    def add(self, obs, global_state, agent_id, action, logp, reward, mask, value, done):
        if self.ptr >= self.cap:
            raise OverflowError("rollout buffer capacity exceeded")
        index = self.ptr
        self.obs[index] = obs
        self.global_state[index] = global_state
        self.agent_id[index] = agent_id
        self.action[index] = action
        self.logp[index] = logp
        self.reward[index] = reward
        self.mask[index] = mask
        self.value[index] = value
        self.done[index] = done
        self.ptr += 1
        self.size = self.ptr

    def compute_returns_advantages(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        episode_ids = np.zeros(self.size, dtype=np.int64)
        bootstrap = {(0, int(agent)): float(last_value) for agent in np.unique(self.agent_id[:self.size])}
        self.advantages, self.returns = compute_grouped_gae(
            self.reward[:self.size], self.value[:self.size], self.agent_id[:self.size],
            episode_ids, bootstrap, gamma, lam,
        )
        return self.advantages, self.returns

    def get_tensors(self, device):
        batch = {
            "obs": self.obs[:self.size], "global_state": self.global_state[:self.size],
            "agent_id": self.agent_id[:self.size], "action": self.action[:self.size],
            "logp": self.logp[:self.size], "mask": self.mask[:self.size],
            "advantages": self.advantages, "returns": self.returns,
        }
        return to_tensors(batch, device)

    def __len__(self):
        return self.size
