"""Residual, paired-action Actor-Critic network for canonical Xiangqi PPO."""

from __future__ import annotations

from rl.environment import ACTION_SIZE, OBSERVATION_PLANES, ROWS, COLS


def build_model(channels: int = 128, blocks: int = 6, pair_width: int = 64):
    """Build an Actor-Critic with 8,100 start/end paired policy logits.

    The policy head scores each ``from → to`` pair by a learned query/key
    interaction instead of a large flat classifier, while retaining the flat
    action interface required by masked PPO.
    """
    import torch
    from torch import nn

    def group_count(width: int) -> int:
        for groups in (16, 8, 4, 2, 1):
            if width % groups == 0:
                return groups
        return 1

    class ResidualBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            groups = group_count(width)
            self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
            self.norm1 = nn.GroupNorm(groups, width)
            self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
            self.norm2 = nn.GroupNorm(groups, width)
            self.activation = nn.SiLU()

        def forward(self, x):
            residual = x
            x = self.activation(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return self.activation(x + residual)

    class PairedPolicyHead(nn.Module):
        def __init__(self, width: int, latent_width: int):
            super().__init__()
            self.project = nn.Sequential(
                nn.Conv2d(width, latent_width, 1, bias=False),
                nn.GroupNorm(group_count(latent_width), latent_width),
                nn.SiLU(),
            )
            self.query = nn.Linear(latent_width, latent_width, bias=False)
            self.key = nn.Linear(latent_width, latent_width, bias=False)
            self.from_bias = nn.Linear(latent_width, 1)
            self.to_bias = nn.Linear(latent_width, 1)
            self.scale = latent_width ** -0.5

        def forward(self, features):
            tokens = self.project(features).flatten(2).transpose(1, 2)
            query = self.query(tokens)
            key = self.key(tokens)
            pair_scores = torch.matmul(query, key.transpose(1, 2)) * self.scale
            pair_scores = pair_scores + self.from_bias(tokens) + self.to_bias(tokens).transpose(1, 2)
            return pair_scores.flatten(1)

    class ValueHead(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.project = nn.Sequential(
                nn.Conv2d(width, 64, 1, bias=False),
                nn.GroupNorm(group_count(64), 64),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )

        def forward(self, features):
            return self.project(features).squeeze(-1)

    class XiangqiActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(OBSERVATION_PLANES, channels, 3, padding=1, bias=False),
                nn.GroupNorm(group_count(channels), channels),
                nn.SiLU(),
            )
            self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.policy = PairedPolicyHead(channels, pair_width)
            self.value = ValueHead(channels)

        def forward(self, observation, action_mask=None):
            features = self.trunk(self.stem(observation))
            logits = self.policy(features)
            if logits.shape[-1] != ACTION_SIZE:
                raise RuntimeError(f"expected {ACTION_SIZE} policy logits, got {logits.shape[-1]}")
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
            return logits, self.value(features)

        def act(self, observation, action_mask, deterministic=False):
            logits, value = self(observation, action_mask)
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
            return action, distribution.log_prob(action), value

    return XiangqiActorCritic()
