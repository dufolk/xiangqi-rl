"""Rollout buffer for MAPPO.

Stores per-agent-step transitions collected during distributed rollouts. The
buffer is *episode-segment* based: we store transitions and the episode end
flags, then compute GAE advantages over the collected trajectory treating each
episode as an independent segment (bootstrapping only at true episode ends).

A single buffer aggregates transitions from potentially many environments and
many agents (flattened), which is what the MAPPO update consumes.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, obs_dim: int, global_dim: int, action_dim: int, capacity: int = 4096):
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.cap = capacity
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.cap, self.obs_dim), dtype=np.float32)
        self.global_state = np.zeros((self.cap, self.global_dim), dtype=np.float32)
        self.agent_id = np.zeros((self.cap,), dtype=np.int64)
        self.action = np.zeros((self.cap,), dtype=np.int64)
        self.logp = np.zeros((self.cap,), dtype=np.float32)
        self.reward = np.zeros((self.cap,), dtype=np.float32)
        self.mask = np.zeros((self.cap, self.action_dim), dtype=np.float32)
        self.value = np.zeros((self.cap,), dtype=np.float32)
        self.done = np.zeros((self.cap,), dtype=bool)
        self.ptr = 0
        self.size = 0

    def add(self, obs, global_state, agent_id, action, logp, reward, mask, value, done):
        if self.ptr >= self.cap:
            # roll: drop oldest by shifting (simple, sufficient for collection)
            self._shift()
        i = self.ptr
        self.obs[i] = obs
        self.global_state[i] = global_state
        self.agent_id[i] = agent_id
        self.action[i] = action
        self.logp[i] = logp
        self.reward[i] = reward
        self.mask[i] = mask
        self.value[i] = value
        self.done[i] = done
        self.ptr += 1
        self.size = max(self.size, self.ptr)

    def _shift(self):
        keep = self.cap // 2
        for attr in ("obs", "global_state", "agent_id", "action", "logp",
                     "reward", "mask", "value"):
            arr = getattr(self, attr)
            setattr(self, attr, np.concatenate([arr[keep:], np.zeros_like(arr[:keep])], axis=0))
        self.done = np.concatenate([self.done[keep:], np.zeros_like(self.done[:keep])])
        self.ptr = keep
        self.size = keep

    def compute_returns_advantages(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """GAE over the buffer. Assumes `done` marks true episode ends;
        segment boundaries are handled by zeroing advantage at episode starts.
        """
        n = self.size
        rewards = self.reward[:n]
        values = self.value[:n]
        dones = self.done[:n].astype(np.float32)

        # append bootstrap value
        vals = np.append(values, last_value)
        ads = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            if dones[t]:
                last_adv = 0.0
            next_val = vals[t + 1]
            delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
            last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
            ads[t] = last_adv
        returns = ads + values
        self.advantages = ads
        self.returns = returns
        return ads, returns

    def get_tensors(self, device):
        def to_np(x):
            return np.ascontiguousarray(x[: self.size])
        obs = torch.as_tensor(to_np(self.obs), device=device)
        gstate = torch.as_tensor(to_np(self.global_state), device=device)
        agent_id = torch.as_tensor(to_np(self.agent_id), device=device, dtype=torch.long)
        action = torch.as_tensor(to_np(self.action), device=device, dtype=torch.long)
        logp = torch.as_tensor(to_np(self.logp), device=device)
        mask = torch.as_tensor(to_np(self.mask), device=device)
        advantages = torch.as_tensor(to_np(self.advantages), device=device)
        returns = torch.as_tensor(to_np(self.returns), device=device)
        # normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "obs": obs,
            "global_state": gstate,
            "agent_id": agent_id,
            "action": action,
            "old_logp": logp,
            "mask": mask,
            "advantages": advantages,
            "returns": returns,
        }

    def __len__(self):
        return self.size
