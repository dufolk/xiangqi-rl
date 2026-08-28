"""TensorBoard-instrumented, multiprocess PPO self-play trainer for Xiangqi."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeStats:
    length: int = 0
    captures: int = 0
    checks: int = 0
    capture_value: float = 0.0
    red_moves: int = 0
    black_moves: int = 0

    def reset(self) -> None:
        self.length = 0
        self.captures = 0
        self.checks = 0
        self.capture_value = 0.0
        self.red_moves = 0
        self.black_moves = 0


CAPTURE_VALUES = {"R": 9.0, "C": 4.5, "H": 4.2, "E": 2.1, "A": 2.1, "P": 1.0, "K": 0.0}


def select_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def explained_variance(torch, prediction, target) -> float:
    variance = torch.var(target)
    if variance < 1e-8:
        return 0.0
    return float(1.0 - torch.var(target - prediction) / variance)


def restore_checkpoint(torch, model, optimizer, device, resume: Path) -> dict[str, int]:
    """Restore model, optimizer and counters from an explicitly supplied checkpoint."""
    if not resume.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resume}")
    try:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(resume, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid Xiangqi PPO checkpoint: {resume}")
    if tuple(checkpoint.get("observation_shape", (16, 10, 9))) != (16, 10, 9):
        raise ValueError("checkpoint observation shape is incompatible with the current 16×10×9 environment")
    if int(checkpoint.get("action_size", 8100)) != 8100:
        raise ValueError("checkpoint action space is incompatible with the current 8100-action environment")
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if hasattr(value, "to"):
                    state[key] = value.to(device)
    return {
        "update": int(checkpoint.get("update", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "completed_episodes": int(checkpoint.get("completed_episodes", 0)),
    }


def train(
    updates: int,
    rollout_steps: int,
    epochs: int,
    learning_rate: float,
    output: Path,
    seed: int,
    log_dir: Path,
    num_envs: int = 8,
    save_interval: int = 10,
    log_interval: int = 1,
    resume: Optional[Path] = None,
) -> None:
    """Train with batch inference in the main process and parallel env workers.

    ``rollout_steps`` is the number of half-moves sampled by *each* environment
    per update. The PPO batch size is therefore ``num_envs × rollout_steps``.
    """
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from game_engine import BLACK, RED
    from rl.model import build_model
    from rl.vector_env import SubprocessVectorEnv

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device(torch)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gamma, gae_lambda = 0.997, 0.95
    global_step = 0
    completed_episodes = 0
    resumed_from = "fresh start"
    if resume is not None:
        restored = restore_checkpoint(torch, model, optimizer, device, resume)
        starting_update = restored["update"]
        global_step = restored["global_step"]
        completed_episodes = restored["completed_episodes"]
        resumed_from = str(resume.resolve())
    else:
        starting_update = 0
    current_episodes = [EpisodeStats() for _ in range(num_envs)]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    batch_size = num_envs * rollout_steps

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("run/config", "\n".join((
        f"device: {device}", f"seed: {seed}", f"updates_to_run: {updates}",
        f"starting_update: {starting_update}", f"resume: {resumed_from}",
        f"num_envs: {num_envs}", f"rollout_steps_per_env: {rollout_steps}", f"batch_size: {batch_size}",
        f"epochs: {epochs}", f"learning_rate: {learning_rate}", f"gamma: {gamma}",
        f"gae_lambda: {gae_lambda}", f"parameters: {parameter_count:,}",
        "collection: spawned subprocess environments; batched learner inference in main process",
        "reward: terminal-only (+1 for a winning terminal move; 0 otherwise)",
    )))
    writer.add_scalar("model/parameter_count", parameter_count, starting_update)
    writer.add_scalar("model/parameter_memory_mb", parameter_count * 4 / 1024 / 1024, starting_update)
    writer.add_scalar("system/num_envs", num_envs, starting_update)
    writer.add_scalar("system/ppo_batch_size", batch_size, starting_update)
    writer.add_scalar("resume/enabled", float(resume is not None), starting_update)
    writer.add_scalar("resume/starting_update", starting_update, starting_update)

    try:
        with SubprocessVectorEnv(num_envs=num_envs) as envs:
            initial = envs.reset([seed + index for index in range(num_envs)])
            observations = np.stack([item[0] for item in initial])
            infos = [item[1] for item in initial]

            # --updates means additional updates, not an absolute target number.
            for update in range(starting_update + 1, starting_update + updates + 1):
                update_started_at = time.perf_counter()
                rollout_observations: list[object] = []
                rollout_masks: list[object] = []
                rollout_actions: list[object] = []
                rollout_log_probs: list[object] = []
                rollout_values: list[object] = []
                rollout_next_values: list[object] = []
                rollout_rewards: list[object] = []
                rollout_boundaries: list[object] = []
                rollout_captures = 0
                rollout_checks = 0
                rollout_legal_counts: list[int] = []
                rollout_terminal_count = 0
                rollout_truncated_count = 0

                for _ in range(rollout_steps):
                    masks = np.stack([item["action_mask"] for item in infos])
                    obs_tensor = torch.as_tensor(observations, device=device)
                    mask_tensor = torch.as_tensor(masks, device=device)
                    with torch.no_grad():
                        actions, log_probs, values = model.act(obs_tensor, mask_tensor)

                    results = envs.step(actions.detach().cpu().tolist())
                    next_observations = np.stack([item[0] for item in results])
                    rewards = np.asarray([item[1] for item in results], dtype=np.float32)
                    boundaries = np.asarray([item[2] or item[3] for item in results], dtype=np.bool_)
                    next_infos = [item[4] for item in results]
                    next_masks = np.stack([item["action_mask"] for item in next_infos])
                    with torch.no_grad():
                        _, next_values = model(
                            torch.as_tensor(next_observations, device=device),
                            torch.as_tensor(next_masks, device=device),
                        )
                    next_values = next_values.masked_fill(torch.as_tensor(boundaries, device=device), 0.0)

                    rollout_observations.append(observations.copy())
                    rollout_masks.append(masks)
                    rollout_actions.append(actions.detach().cpu())
                    rollout_log_probs.append(log_probs.detach().cpu())
                    rollout_values.append(values.detach().cpu())
                    rollout_next_values.append(next_values.detach().cpu())
                    rollout_rewards.append(rewards)
                    rollout_boundaries.append(boundaries)
                    global_step += num_envs

                    finished_indices: list[int] = []
                    for index, (_, _, terminated, truncated, info) in enumerate(results):
                        episode = current_episodes[index]
                        actor = info["actor"]
                        episode.length += 1
                        episode.red_moves += int(actor == RED)
                        episode.black_moves += int(actor == BLACK)
                        episode.captures += int(info["capture"])
                        episode.checks += int(info["gives_check"])
                        episode.capture_value += CAPTURE_VALUES.get(info["captured_kind"], 0.0)
                        rollout_captures += int(info["capture"])
                        rollout_checks += int(info["gives_check"])
                        rollout_legal_counts.append(int(info["legal_action_count"]))

                        if terminated or truncated:
                            completed_episodes += 1
                            rollout_terminal_count += int(terminated)
                            rollout_truncated_count += int(truncated)
                            winner = info["winner"]
                            writer.add_scalar("episode/length_halfmoves", episode.length, completed_episodes)
                            writer.add_scalar("episode/captures", episode.captures, completed_episodes)
                            writer.add_scalar("episode/checks", episode.checks, completed_episodes)
                            writer.add_scalar("episode/capture_value", episode.capture_value, completed_episodes)
                            writer.add_scalar("episode/red_moves", episode.red_moves, completed_episodes)
                            writer.add_scalar("episode/black_moves", episode.black_moves, completed_episodes)
                            writer.add_scalar("episode/red_win", float(winner == RED), completed_episodes)
                            writer.add_scalar("episode/black_win", float(winner == BLACK), completed_episodes)
                            writer.add_scalar("episode/truncated", float(truncated), completed_episodes)
                            writer.add_scalar("episode/terminal_reason_mate", float(info["reason"] == "将死"), completed_episodes)
                            writer.add_scalar("episode/terminal_reason_stalemate", float(info["reason"] == "困毙"), completed_episodes)
                            episode.reset()
                            finished_indices.append(index)

                    if finished_indices:
                        resets = envs.reset_at(finished_indices)
                        for index, (reset_observation, reset_info) in zip(finished_indices, resets):
                            next_observations[index] = reset_observation
                            next_infos[index] = reset_info
                    observations, infos = next_observations, next_infos

                rollout_observations_tensor = torch.as_tensor(np.stack(rollout_observations), device=device)
                rollout_masks_tensor = torch.as_tensor(np.stack(rollout_masks), device=device)
                rollout_actions_tensor = torch.stack(rollout_actions).to(device)
                rollout_log_probs_tensor = torch.stack(rollout_log_probs).to(device)
                rollout_values_tensor = torch.stack(rollout_values).to(device)
                rollout_next_values_tensor = torch.stack(rollout_next_values).to(device)
                rollout_rewards_tensor = torch.as_tensor(np.stack(rollout_rewards), device=device)
                rollout_boundaries_tensor = torch.as_tensor(np.stack(rollout_boundaries), device=device)
                advantages = torch.zeros((rollout_steps, num_envs), device=device)
                gae = torch.zeros(num_envs, device=device)

                # Next observations are canonical to the opponent, hence -V(s').
                for index in reversed(range(rollout_steps)):
                    delta = rollout_rewards_tensor[index] - gamma * rollout_next_values_tensor[index] - rollout_values_tensor[index]
                    gae = torch.where(
                        rollout_boundaries_tensor[index],
                        delta,
                        delta - gamma * gae_lambda * gae,
                    )
                    advantages[index] = gae

                returns = advantages + rollout_values_tensor
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                flat_observations = rollout_observations_tensor.flatten(0, 1)
                flat_masks = rollout_masks_tensor.flatten(0, 1)
                flat_actions = rollout_actions_tensor.flatten(0, 1)
                flat_log_probs = rollout_log_probs_tensor.flatten(0, 1)
                flat_returns = returns.flatten(0, 1)
                flat_advantages = normalized_advantages.flatten(0, 1)

                for _ in range(epochs):
                    logits, values = model(flat_observations, flat_masks)
                    distribution = torch.distributions.Categorical(logits=logits)
                    log_probs = distribution.log_prob(flat_actions)
                    ratio = (log_probs - flat_log_probs).exp()
                    unclipped = ratio * flat_advantages
                    clipped = ratio.clamp(0.8, 1.2) * flat_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (flat_returns - values).pow(2).mean()
                    entropy = distribution.entropy().mean()
                    loss = policy_loss + value_loss - 0.01 * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
                    optimizer.step()

                elapsed = time.perf_counter() - update_started_at
                if update % log_interval == 0:
                    approx_kl = float((flat_log_probs - log_probs).mean().detach())
                    clip_fraction = float((torch.abs(ratio - 1.0) > 0.2).float().mean().detach())
                    writer.add_scalar("train/loss_total", float(loss.detach()), update)
                    writer.add_scalar("train/loss_policy", float(policy_loss.detach()), update)
                    writer.add_scalar("train/loss_value", float(value_loss.detach()), update)
                    writer.add_scalar("train/entropy", float(entropy.detach()), update)
                    writer.add_scalar("train/approx_kl", approx_kl, update)
                    writer.add_scalar("train/clip_fraction", clip_fraction, update)
                    writer.add_scalar("train/gradient_norm", gradient_norm, update)
                    writer.add_scalar("train/explained_variance", explained_variance(torch, values.detach(), flat_returns), update)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], update)
                    writer.add_scalar("rollout/mean_reward", float(np.mean(rollout_rewards)), update)
                    writer.add_scalar("rollout/captures", rollout_captures, update)
                    writer.add_scalar("rollout/checks", rollout_checks, update)
                    writer.add_scalar("rollout/mean_legal_actions", float(np.mean(rollout_legal_counts)), update)
                    writer.add_scalar("rollout/terminal_games", rollout_terminal_count, update)
                    writer.add_scalar("rollout/truncated_games", rollout_truncated_count, update)
                    writer.add_scalar("rollout/completed_episodes", completed_episodes, update)
                    writer.add_scalar("rollout/global_halfmoves", global_step, update)
                    writer.add_scalar("system/update_seconds", elapsed, update)
                    writer.add_scalar("system/rollout_halfmoves_per_second", batch_size / elapsed, update)
                    writer.flush()

                print(
                    f"update={update:04d} envs={num_envs:02d} batch={batch_size:04d} episodes={completed_episodes:04d} "
                    f"loss={float(loss.detach()):.4f} entropy={float(entropy.detach()):.3f} "
                    f"checks={rollout_checks:03d} captures={rollout_captures:03d} "
                    f"fps={batch_size / elapsed:.1f} device={device}"
                )
                if update % save_interval == 0 or update == updates:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                        "global_step": global_step, "completed_episodes": completed_episodes,
                        "observation_shape": (16, 10, 9), "action_size": 8100,
                        "parameter_count": parameter_count, "num_envs": num_envs,
                        "checkpoint_format": 2,
                    }, output)
    finally:
        writer.flush()
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiprocess TensorBoard PPO self-play trainer for Xiangqi")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128, help="half-moves per environment per update")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/xiangqi_ppo.pt"))
    parser.add_argument("--resume", type=Path, default=None, help="explicit checkpoint path to continue from; omitted means fresh training")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    log_dir = args.log_dir or Path("runs") / f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(
        args.updates, args.rollout_steps, args.epochs, args.learning_rate, args.output, args.seed,
        log_dir, args.num_envs, args.save_interval, args.log_interval, args.resume,
    )


if __name__ == "__main__":
    main()
