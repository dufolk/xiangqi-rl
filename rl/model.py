"""Masked convolutional Actor-Critic model for Xiangqi PPO."""

from __future__ import annotations


def build_model(channels: int = 128):
    """Build and return the model class instance; keeps torch optional for GUI users."""
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
            )
            self.activation = nn.ReLU()

        def forward(self, x):
            return self.activation(x + self.net(x))

    class XiangqiActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(16, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(),
                ResidualBlock(channels), ResidualBlock(channels),
            )
            self.policy = nn.Sequential(
                nn.Conv2d(channels, 32, 1), nn.ReLU(), nn.Flatten(),
                nn.Linear(32 * 10 * 9, 8100),
            )
            self.value = nn.Sequential(
                nn.Conv2d(channels, 16, 1), nn.ReLU(), nn.Flatten(),
                nn.Linear(16 * 10 * 9, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh(),
            )

        def forward(self, observation, action_mask=None):
            features = self.trunk(observation)
            logits = self.policy(features)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
            return logits, self.value(features).squeeze(-1)

        def act(self, observation, action_mask, deterministic=False):
            distribution = torch.distributions.Categorical(logits=self(observation, action_mask)[0])
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            _, value = self(observation, action_mask)
            return action, distribution.log_prob(action), value

    return XiangqiActorCritic()
