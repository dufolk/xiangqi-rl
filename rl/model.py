"""Residual Actor-Critic with autoregressive source and target move selection."""

from __future__ import annotations

from rl.environment import ACTION_SIZE, OBSERVATION_PLANES, ROWS, COLS


def build_model(channels: int = 128, blocks: int = 6, pair_width: int = 64):
    """Build an autoregressive Actor-Critic for canonical Xiangqi PPO.

    The policy first uses cross-attention to choose one of 90 source squares,
    then applies a Pointer head over 90 target squares conditioned on that
    source. The environment-facing action remains a flat ``from * 90 + to``
    index, but the network never constructs 8,100 policy logits.
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

    class AutoregressivePolicyHead(nn.Module):
        """Choose a source square with cross-attention, then point to its target."""

        def __init__(self, width: int, latent_width: int):
            super().__init__()
            self.project = nn.Sequential(
                nn.Conv2d(width, latent_width, 1, bias=False),
                nn.GroupNorm(group_count(latent_width), latent_width),
                nn.SiLU(),
            )
            attention_heads = next(heads for heads in (8, 4, 2, 1) if latent_width % heads == 0)
            self.source_query = nn.Linear(latent_width, latent_width, bias=False)
            self.board_key = nn.Linear(latent_width, latent_width, bias=False)
            self.board_value = nn.Linear(latent_width, latent_width, bias=False)
            self.piece_cross_attention = nn.MultiheadAttention(
                latent_width, attention_heads, batch_first=True
            )
            self.source_norm = nn.LayerNorm(latent_width)
            self.source_score = nn.Linear(latent_width, 1)
            self.pointer_query = nn.Linear(latent_width * 2, latent_width, bias=False)
            self.pointer_key = nn.Linear(latent_width, latent_width, bias=False)
            self.scale = latent_width ** -0.5

        def tokens_and_source_logits(self, features, source_mask):
            tokens = self.project(features).flatten(2).transpose(1, 2)
            attended, _ = self.piece_cross_attention(
                self.source_query(tokens), self.board_key(tokens), self.board_value(tokens),
                need_weights=False,
            )
            source_context = self.source_norm(tokens + attended)
            source_logits = self.source_score(source_context).squeeze(-1)
            source_logits = source_logits.masked_fill(
                ~source_mask, torch.finfo(source_logits.dtype).min
            )
            return tokens, source_context, source_logits

        def target_logits(self, tokens, source_context, source_indices, target_mask):
            batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
            selected = torch.cat((
                tokens[batch_indices, source_indices],
                source_context[batch_indices, source_indices],
            ), dim=-1)
            query = self.pointer_query(selected)
            keys = self.pointer_key(tokens)
            logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) * self.scale
            return logits.masked_fill(~target_mask, torch.finfo(logits.dtype).min)

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
            self.policy = AutoregressivePolicyHead(channels, pair_width)
            self.value = ValueHead(channels)

        @staticmethod
        def _split_action_mask(action_mask):
            if action_mask is None:
                raise ValueError("an action mask is required for autoregressive move selection")
            if action_mask.ndim != 2 or action_mask.shape[-1] != ACTION_SIZE:
                raise ValueError(f"expected action_mask shape [batch, {ACTION_SIZE}], got {tuple(action_mask.shape)}")
            pair_mask = action_mask.bool().reshape(-1, ROWS * COLS, ROWS * COLS)
            source_mask = pair_mask.any(dim=-1)
            if not bool(source_mask.any(dim=-1).all()):
                raise ValueError("every batch item must have at least one legal action")
            return pair_mask, source_mask

        @staticmethod
        def _split_actions(actions):
            source_indices = torch.div(actions.long(), ROWS * COLS, rounding_mode="floor")
            target_indices = torch.remainder(actions.long(), ROWS * COLS)
            return source_indices, target_indices

        def _policy_terms(self, observation, action_mask, source_indices=None):
            features = self.trunk(self.stem(observation))
            pair_mask, source_mask = self._split_action_mask(action_mask)
            tokens, source_context, source_logits = self.policy.tokens_and_source_logits(features, source_mask)
            if source_indices is None:
                return source_logits, None, self.value(features)
            batch_indices = torch.arange(source_logits.shape[0], device=source_logits.device)
            target_logits = self.policy.target_logits(
                tokens, source_context, source_indices, pair_mask[batch_indices, source_indices]
            )
            return source_logits, target_logits, self.value(features)

        def forward(self, observation, action_mask=None):
            """Return masked source-square logits and a state value.

            Target logits are action-conditioned and are exposed by
            :meth:`evaluate_actions` to avoid constructing an 8,100-way tensor.
            """
            source_logits, _, value = self._policy_terms(observation, action_mask)
            return source_logits, value

        def evaluate_actions(self, observation, action_mask, actions):
            """Evaluate flat actions under ``π(source) · π(target | source)``."""
            source_indices, target_indices = self._split_actions(actions)
            source_logits, target_logits, value = self._policy_terms(
                observation, action_mask, source_indices
            )
            source_distribution = torch.distributions.Categorical(logits=source_logits)
            target_distribution = torch.distributions.Categorical(logits=target_logits)
            log_prob = source_distribution.log_prob(source_indices) + target_distribution.log_prob(target_indices)
            entropy = source_distribution.entropy() + target_distribution.entropy()
            return log_prob, entropy, value

        def act(self, observation, action_mask, deterministic=False):
            features = self.trunk(self.stem(observation))
            pair_mask, source_mask = self._split_action_mask(action_mask)
            tokens, source_context, source_logits = self.policy.tokens_and_source_logits(features, source_mask)
            source_distribution = torch.distributions.Categorical(logits=source_logits)
            source_indices = source_logits.argmax(dim=-1) if deterministic else source_distribution.sample()
            batch_indices = torch.arange(source_logits.shape[0], device=source_logits.device)
            target_logits = self.policy.target_logits(
                tokens, source_context, source_indices, pair_mask[batch_indices, source_indices]
            )
            target_distribution = torch.distributions.Categorical(logits=target_logits)
            target_indices = target_logits.argmax(dim=-1) if deterministic else target_distribution.sample()
            action = source_indices * (ROWS * COLS) + target_indices
            log_prob = source_distribution.log_prob(source_indices) + target_distribution.log_prob(target_indices)
            return action, log_prob, self.value(features)

    return XiangqiActorCritic()
