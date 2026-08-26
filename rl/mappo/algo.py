"""MAPPO (CTDE) update core.

Parameter-sharing MAPPO: a single Actor and a single centralized Critic are
shared across all agents (seats). The Critic is trained on the *joint global
state* (CTDE); the Actor is trained on per-agent observations with an agent-id
embedding. This is the standard centralized-training / decentralized-execution
recipe.

Losses:
    L = L_clip^actor + c1 * L_value + c2 * entropy
where
    L_clip^actor = -E[ min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv) ]  (per agent-step)
    L_value      = E[ (V(s_global) - return)^2 ]
    entropy      = E[ H(pi(.|s,id)) ]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .networks import ActorCritic


class MAPPO:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        global_dim: int,
        device: torch.device,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.device = device
        self.n_agents = n_agents
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.ac = ActorCritic(obs_dim, action_dim, n_agents, global_dim).to(device)
        self.actor_opt = optim.Adam(self.ac.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.ac.critic.parameters(), lr=lr_critic)

    @torch.no_grad()
    def act(self, obs, agent_id, mask, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device).float()
        agent_id = torch.as_tensor(agent_id, device=self.device, dtype=torch.long)
        mask = torch.as_tensor(mask, device=self.device).float()
        logits = self.ac.actor(obs, agent_id, mask)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(action)
        values = self.ac.critic(self._global_from_obs(obs))
        value = values.gather(1, agent_id.unsqueeze(-1)).squeeze(-1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy(), dist

    def _global_from_obs(self, obs_batch):
        # joint global state = concat of all agents' obs.
        # During collection we pass the full per-agent obs set; here we accept
        # a single obs tensor and build the global by repeating/concatenating
        # across the agent axis. For simplicity the collector precomputes the
        # global_state tensor (concat over agents) and passes it separately.
        return obs_batch  # placeholder; real global passed in evaluate()

    @torch.no_grad()
    def value_only(self, global_state, agent_id=None):
        g = torch.as_tensor(global_state, device=self.device).float()
        values = self.ac.critic(g)
        if agent_id is not None:
            seats = torch.as_tensor(agent_id, device=self.device, dtype=torch.long)
            values = values.gather(1, seats.unsqueeze(-1)).squeeze(-1)
        return values.cpu().numpy()

    def evaluate(self, global_state, obs, agent_id, mask, action):
        logits = self.ac.actor(obs, agent_id, mask)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.ac.critic(global_state)
        value = values.gather(1, agent_id.unsqueeze(-1)).squeeze(-1)
        return logp, entropy, value

    def update(self, batch: dict, n_epochs: int = 4, minibatch_size: int = 1024):
        obs = batch["obs"]
        gstate = batch["global_state"]
        agent_id = batch["agent_id"]
        action = batch["action"]
        old_logp = batch["old_logp"]
        mask = batch["mask"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        n = obs.shape[0]
        mb = min(minibatch_size, n)
        all_metrics = []

        for _ in range(n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                idx = perm[start : start + mb]
                b_obs = obs[idx]
                b_g = gstate[idx]
                b_aid = agent_id[idx]
                b_act = action[idx]
                b_old = old_logp[idx]
                b_mask = mask[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                logp, entropy, value = self.evaluate(b_g, b_obs, b_aid, b_mask, b_act)

                ratio = torch.exp(logp - b_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value, b_ret)

                entropy_loss = entropy.mean()

                actor_loss = policy_loss - self.entropy_coef * entropy_loss
                critic_loss = self.value_coef * value_loss

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                all_metrics.append({
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_loss.item()),
                    "approx_kl": float(((b_old - logp).abs().mean()).item()),
                })
        return all_metrics

    def save(self, path: str):
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path, map_location=self.device))
