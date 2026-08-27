"""Single-process masked PPO self-play baseline.

Run with: ``python -m rl.train --updates 100``.
The implementation is intentionally compact and serves as a dependable platform
for later vector environments, league play and distributed rollout workers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Transition:
    observation: object
    mask: object
    action: object
    old_log_prob: object
    value: object
    reward: float
    terminal: bool
    episode: int


def train(updates: int, rollout_steps: int, epochs: int, learning_rate: float, output: Path, seed: int) -> None:
    import numpy as np
    import torch

    from rl.environment import XiangqiEnv
    from rl.model import build_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    env = XiangqiEnv()
    observation, info = env.reset(seed=seed)
    episode = 0

    for update in range(1, updates + 1):
        transitions: list[Transition] = []
        for _ in range(rollout_steps):
            obs_tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
            mask_tensor = torch.as_tensor(info["action_mask"], device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = model.act(obs_tensor, mask_tensor)
            result = env.step(int(action.item()))
            transitions.append(Transition(observation, info["action_mask"], action.cpu(), log_prob.cpu(), value.cpu(), result.reward, result.terminated or result.truncated, episode))
            observation, info = result.observation, result.info
            if result.terminated or result.truncated:
                episode += 1
                observation, info = env.reset()

        observations = torch.as_tensor(np.stack([item.observation for item in transitions]), device=device)
        masks = torch.as_tensor(np.stack([item.mask for item in transitions]), device=device)
        actions = torch.cat([item.action for item in transitions]).to(device)
        old_log_probs = torch.cat([item.old_log_prob for item in transitions]).to(device)
        old_values = torch.cat([item.value for item in transitions]).to(device)
        advantages = torch.zeros(len(transitions), device=device)
        gae = torch.tensor(0.0, device=device)
        next_value = torch.tensor(0.0, device=device)
        gamma, lam = 0.99, 0.95
        for index in reversed(range(len(transitions))):
            item = transitions[index]
            if item.terminal or index == len(transitions) - 1 or transitions[index + 1].episode != item.episode:
                next_value = torch.tensor(0.0, device=device)
                gae = torch.tensor(0.0, device=device)
            delta = item.reward - gamma * next_value - old_values[index]
            gae = delta - gamma * lam * gae
            advantages[index] = gae
            next_value = old_values[index]
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            logits, values = model(observations, masks)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            ratio = (log_probs - old_log_probs).exp()
            clipped = ratio.clamp(0.8, 1.2) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        print(f"update={update:04d} episodes={episode:04d} loss={loss.item():.4f} entropy={entropy.item():.3f} device={device}")
        if update % 10 == 0 or update == updates:
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update}, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiangqi masked PPO self-play trainer")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/xiangqi_ppo.pt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train(args.updates, args.rollout_steps, args.epochs, args.learning_rate, args.output, args.seed)


if __name__ == "__main__":
    main()
