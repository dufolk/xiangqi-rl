"""TensorBoard-instrumented, multiprocess PPO self-play trainer for Xiangqi."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from rl.opponents import ModelOpponent
from rl.search import SearchOpponent


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
    if checkpoint.get("policy_format") != "autoregressive_cross_attention_pointer_v1":
        raise ValueError(
            "checkpoint policy is incompatible with the autoregressive cross-attention pointer model; "
            "start a new run or use a checkpoint trained with this policy format"
        )
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
    opponent=None,
    target_win_rate: float = 1.0,
    win_rate_window: int = 100,
    min_updates_before_stop: int = 10,
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
    self_play = opponent is None
    # Self-play alternates viewpoints, so the opponent's value is the negation of
    # ours; when facing a fixed search opponent the observation stays on our side,
    # so the bootstrap value keeps its sign.
    next_value_sign = -1.0 if self_play else 1.0
    current_episodes = [EpisodeStats() for _ in range(num_envs)]
    agent_wins = 0
    agent_win_ema = 0.0
    recent_outcomes: deque[float] = deque(maxlen=win_rate_window)
    stop_training = False
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    batch_size = num_envs * rollout_steps

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("运行配置", "\n".join((
        f"device: {device}", f"seed: {seed}", f"updates_to_run: {updates}",
        f"starting_update: {starting_update}", f"resume: {resumed_from}",
        f"num_envs: {num_envs}", f"rollout_steps_per_env: {rollout_steps}", f"batch_size: {batch_size}",
        f"epochs: {epochs}", f"learning_rate: {learning_rate}", f"gamma: {gamma}",
        f"gae_lambda: {gae_lambda}", f"parameters: {parameter_count:,}",
        "collection: spawned subprocess environments; batched learner inference in main process",
        "reward: terminal-only (+1 for a winning terminal move; 0 otherwise)",
        f"opponent_mode: {'self-play' if self_play else (f'search(depth={opponent.depth})' if hasattr(opponent, 'depth') else f'model({opponent.model_path})')}",
    )))
    writer.add_scalar("模型/参数数量", parameter_count, starting_update)
    writer.add_scalar("模型/参数内存_MB", parameter_count * 4 / 1024 / 1024, starting_update)
    writer.add_scalar("系统/环境数", num_envs, starting_update)
    writer.add_scalar("系统/PPO批次大小", batch_size, starting_update)
    writer.add_scalar("恢复/是否启用", float(resume is not None), starting_update)
    writer.add_scalar("恢复/起始更新步数", starting_update, starting_update)

    try:
        with SubprocessVectorEnv(num_envs=num_envs, opponent=opponent) as envs:
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
                            agent_won = winner is not None and winner == actor
                            if agent_won:
                                agent_wins += 1
                            agent_win_ema = 0.95 * agent_win_ema + (1.0 if agent_won else 0.0) * 0.05
                            recent_outcomes.append(1.0 if agent_won else 0.0)
                            if (len(recent_outcomes) == win_rate_window
                                    and sum(recent_outcomes) >= win_rate_window * target_win_rate
                                    and (update - starting_update) >= min_updates_before_stop):
                                stop_training = True
                            writer.add_scalar("对局/智能体胜率_窗口", sum(recent_outcomes) / len(recent_outcomes), completed_episodes)
                            writer.add_scalar("对局/长度_半回合", episode.length, completed_episodes)
                            writer.add_scalar("对局/吃子数", episode.captures, completed_episodes)
                            writer.add_scalar("对局/将军次数", episode.checks, completed_episodes)
                            writer.add_scalar("对局/吃子价值", episode.capture_value, completed_episodes)
                            writer.add_scalar("对局/红方步数", episode.red_moves, completed_episodes)
                            writer.add_scalar("对局/黑方步数", episode.black_moves, completed_episodes)
                            writer.add_scalar("对局/红方胜", float(winner == RED), completed_episodes)
                            writer.add_scalar("对局/黑方胜", float(winner == BLACK), completed_episodes)
                            writer.add_scalar("对局/智能体胜率", agent_wins / completed_episodes, completed_episodes)
                            writer.add_scalar("对局/智能体胜率_EMA", agent_win_ema, completed_episodes)
                            writer.add_scalar("对局/截断", float(truncated), completed_episodes)
                            writer.add_scalar("对局/终局原因_将死", float(info["reason"] == "将死"), completed_episodes)
                            writer.add_scalar("对局/终局原因_困毙", float(info["reason"] == "困毙"), completed_episodes)
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
                    delta = rollout_rewards_tensor[index] + next_value_sign * gamma * rollout_next_values_tensor[index] - rollout_values_tensor[index]
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
                    log_probs, action_entropy, values = model.evaluate_actions(
                        flat_observations, flat_masks, flat_actions
                    )
                    ratio = (log_probs - flat_log_probs).exp()
                    unclipped = ratio * flat_advantages
                    clipped = ratio.clamp(0.8, 1.2) * flat_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (flat_returns - values).pow(2).mean()
                    entropy = action_entropy.mean()
                    loss = policy_loss + value_loss - 0.01 * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
                    optimizer.step()

                elapsed = time.perf_counter() - update_started_at
                if update % log_interval == 0:
                    approx_kl = float((flat_log_probs - log_probs).mean().detach())
                    clip_fraction = float((torch.abs(ratio - 1.0) > 0.2).float().mean().detach())
                    writer.add_scalar("训练/总损失", float(loss.detach()), update)
                    writer.add_scalar("训练/策略损失", float(policy_loss.detach()), update)
                    writer.add_scalar("训练/价值损失", float(value_loss.detach()), update)
                    writer.add_scalar("训练/熵", float(entropy.detach()), update)
                    writer.add_scalar("训练/近似KL", approx_kl, update)
                    writer.add_scalar("训练/裁剪比例", clip_fraction, update)
                    writer.add_scalar("训练/梯度范数", gradient_norm, update)
                    writer.add_scalar("训练/解释方差", explained_variance(torch, values.detach(), flat_returns), update)
                    writer.add_scalar("训练/学习率", optimizer.param_groups[0]["lr"], update)
                    writer.add_scalar("采样/平均奖励", float(np.mean(rollout_rewards)), update)
                    writer.add_scalar("采样/吃子数", rollout_captures, update)
                    writer.add_scalar("采样/将军次数", rollout_checks, update)
                    writer.add_scalar("采样/平均合法着法数", float(np.mean(rollout_legal_counts)), update)
                    writer.add_scalar("采样/对局结束数", rollout_terminal_count, update)
                    writer.add_scalar("采样/截断对局数", rollout_truncated_count, update)
                    writer.add_scalar("采样/已完成对局", completed_episodes, update)
                    writer.add_scalar("采样/全局半回合数", global_step, update)
                    writer.add_scalar("系统/单步更新耗时_秒", elapsed, update)
                    writer.add_scalar("系统/每秒半回合", batch_size / elapsed, update)
                    writer.flush()

                if stop_training:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                        "global_step": global_step, "completed_episodes": completed_episodes,
                        "observation_shape": (16, 10, 9), "action_size": 8100,
                        "parameter_count": parameter_count, "num_envs": num_envs,
                        "policy_format": "autoregressive_cross_attention_pointer_v1",
                        "checkpoint_format": 3,
                    }, output)
                    print(f"[early-stop] 稳定胜率达标（近 {win_rate_window} 局 >= {target_win_rate:.0%}），于 update={update} 停止并保存 -> {output}")
                    writer.add_text("状态", f"early-stop: stable win rate >= {target_win_rate} over {win_rate_window} episodes at update {update}")
                    break

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
                        "policy_format": "autoregressive_cross_attention_pointer_v1",
                        "checkpoint_format": 3,
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
    parser.add_argument("--opponent", choices=("selfplay", "search", "model"), default="selfplay",
                        help="selfplay = RL vs RL; search = RL vs fixed alpha-beta; model = RL vs frozen checkpoint")
    parser.add_argument("--search-depth", type=int, default=3, help="alpha-beta search depth when --opponent search")
    parser.add_argument("--search-quiescence", action="store_true", help="enable capture-only quiescence search (slower but stronger)")
    parser.add_argument("--opponent-model", type=Path, default=Path("checkpoints/xiangqi_ppo_continued.pt"),
                        help="checkpoint used as the frozen opponent when --opponent model")
    parser.add_argument("--opponent-stochastic", action="store_true",
                        help="sample opponent moves instead of greedy (model opponent only)")
    parser.add_argument("--target-win-rate", type=float, default=1.0,
                        help="stop training once the windowed agent win rate stays at/above this")
    parser.add_argument("--win-rate-window", type=int, default=100,
                        help="number of recent episodes used to judge a stable win rate")
    parser.add_argument("--min-updates-before-stop", type=int, default=10,
                        help="never early-stop before this many updates")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    opponent = None
    if args.opponent == "search":
        opponent = SearchOpponent(depth=args.search_depth, use_quiescence=args.search_quiescence)
    elif args.opponent == "model":
        opponent = ModelOpponent(model_path=args.opponent_model, deterministic=not args.opponent_stochastic)
    log_dir = args.log_dir or Path("runs") / f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(
        args.updates, args.rollout_steps, args.epochs, args.learning_rate, args.output, args.seed,
        log_dir, args.num_envs, args.save_interval, args.log_interval, args.resume, opponent,
        args.target_win_rate, args.win_rate_window, args.min_updates_before_stop,
    )


if __name__ == "__main__":
    main()
