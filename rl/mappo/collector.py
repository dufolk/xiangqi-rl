"""Distributed, trajectory-safe MAPPO rollout collection.

Each CPU worker owns one persistent environment and receives model weights on an
independent latest-only control queue. Rollouts are cut at ``horizon`` without
resetting the game; non-terminal cuts bootstrap from the next canonical global
state. GAE is computed per worker payload, episode and seat before aggregation.
"""

import multiprocessing as mp
import os
import pickle
import queue
import time
from typing import List, Optional

import numpy as np

from .buffer import concatenate_rollouts, to_tensors
from .encoders import ACTION_SPACE, OBS_DIM
from .envs import SplendorMAEnv


def _cpu_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _load_latest_weights(actor_critic, weight_queue, current_version):
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        version, blob = latest
        actor_critic.load_state_dict(pickle.loads(blob))
        actor_critic.eval()
        return int(version)
    return current_version


def assign_terminal_rewards(rewards, agent_ids, episode_ids, episode_id, terminal_rewards):
    """Attach each seat's terminal reward to its final action in the episode segment."""
    for seat, terminal_reward in enumerate(terminal_rewards):
        matching = [
            index for index in range(len(rewards) - 1, -1, -1)
            if episode_ids[index] == episode_id and agent_ids[index] == seat
        ]
        if not matching:
            raise RuntimeError(f"terminal segment is missing seat {seat} trajectory")
        rewards[matching[0]] += float(terminal_reward)


def worker_loop(
    worker_id: int,
    n_agents: int,
    team_mode: bool,
    target_score: int,
    horizon: int,
    initial_weights: bytes,
    initial_version: int,
    weight_queue: "mp.Queue",
    data_queue: "mp.Queue",
    seed_base: int,
    stop_event: "mp.Event",
):
    """Collect fixed-horizon segments while preserving environment continuity."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import torch
    from .networks import ActorCritic

    actor_critic = ActorCritic(OBS_DIM, ACTION_SPACE, n_agents, OBS_DIM)
    actor_critic.load_state_dict(pickle.loads(initial_weights))
    actor_critic.eval()
    policy_version = int(initial_version)

    env = SplendorMAEnv(
        team_mode=team_mode,
        target_score=target_score,
        seed=seed_base + worker_id,
        n_players=n_agents,
    )
    rng = np.random.RandomState(seed_base * 1000 + worker_id)
    obs, global_state, mask, info = env.reset(seed=int(rng.randint(0, 2**30)))
    episode_id = 0
    episode_reward = 0.0
    episode_length = 0

    while not stop_event.is_set():
        policy_version = _load_latest_weights(actor_critic, weight_queue, policy_version)
        rollout = {
            "obs": [], "global_state": [], "agent_id": [], "action": [],
            "logp": [], "reward": [], "mask": [], "value": [],
            "done": [], "episode_id": [],
        }
        bootstrap_values = {}
        episode_summaries = []

        segment_steps = 0
        while not stop_event.is_set():
            acting = int(info["acting_agent"])
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs[None], dtype=torch.float32)
                mask_tensor = torch.as_tensor(mask[None], dtype=torch.float32)
                seat_tensor = torch.as_tensor([acting], dtype=torch.long)
                logits = actor_critic.actor(obs_tensor, seat_tensor, mask_tensor)
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample().item())
                logp = float(distribution.log_prob(torch.as_tensor(action)).item())
                seat_values = actor_critic.critic(torch.as_tensor(global_state[None], dtype=torch.float32))
                value = float(seat_values[0, acting].item())

            next_obs, next_global_state, next_mask, reward, terminated, next_info = env.step(action)
            rollout["obs"].append(obs)
            rollout["global_state"].append(global_state)
            rollout["agent_id"].append(acting)
            rollout["action"].append(action)
            rollout["logp"].append(logp)
            rollout["reward"].append(float(reward))
            rollout["mask"].append(mask)
            rollout["value"].append(value)
            rollout["done"].append(bool(terminated))
            rollout["episode_id"].append(episode_id)
            episode_reward += float(reward)
            episode_length += 1
            segment_steps += 1

            obs, global_state, mask, info = next_obs, next_global_state, next_mask, next_info
            if terminated:
                terminal_rewards = next_info.get("terminal_rewards", [0.0] * n_agents)
                assign_terminal_rewards(
                    rollout["reward"], rollout["agent_id"], rollout["episode_id"],
                    episode_id, terminal_rewards,
                )
                episode_reward += sum(float(value) for value in terminal_rewards)
                bootstrap_values[str(episode_id)] = [0.0] * n_agents
                episode_summaries.append({
                    "reward": float(episode_reward),
                    "length": int(episode_length),
                })
                episode_id += 1
                episode_reward = 0.0
                episode_length = 0
                obs, global_state, mask, info = env.reset(seed=int(rng.randint(0, 2**30)))

            if segment_steps >= horizon and env.game.turn == 0 and env.game.phase == "action":
                break

        if not rollout["obs"]:
            continue

        active_episode = int(rollout["episode_id"][-1])
        if active_episode not in {int(key) for key in bootstrap_values}:
            with torch.no_grad():
                values = actor_critic.critic(torch.as_tensor(global_state[None], dtype=torch.float32))[0]
            bootstrap_values[str(active_episode)] = values.cpu().numpy().astype(np.float32).tolist()

        payload = {
            "worker_id": worker_id,
            "policy_version": policy_version,
            "bootstrap_values": bootstrap_values,
            "episode_summaries": episode_summaries,
        }
        for key, values in rollout.items():
            dtype = np.int64 if key in ("agent_id", "action", "episode_id") else bool if key == "done" else np.float32
            payload[key] = np.asarray(values, dtype=dtype)
        data_queue.put(payload)

    data_queue.put({"__worker_done__": worker_id})


class DistributedCollector:
    """Manage rollout workers and centralized PPO updates."""

    def __init__(
        self,
        n_workers: int = None,
        n_agents: int = 4,
        team_mode: bool = False,
        target_score: int = 15,
        horizon: int = 64,
        seed: int = 1234,
        device: str = "cpu",
    ):
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 4) - 1)
        self.n_workers = n_workers
        self.n_agents = n_agents
        self.team_mode = team_mode
        self.target_score = target_score
        self.horizon = horizon
        self.seed = seed
        self.device = device
        self.ctx = mp.get_context("spawn")
        self.weight_queues = [self.ctx.Queue(maxsize=1) for _ in range(n_workers)]
        self.data_queue = self.ctx.Queue(maxsize=n_workers * 4)
        self.stop_event = self.ctx.Event()
        self.workers: List[mp.Process] = []
        self.policy_version = 0

    def start(self, initial_state_dict):
        self.stop_event.clear()
        initial_blob = pickle.dumps(_cpu_state_dict(initial_state_dict))
        for worker_id in range(self.n_workers):
            process = self.ctx.Process(
                target=worker_loop,
                args=(
                    worker_id, self.n_agents, self.team_mode, self.target_score,
                    self.horizon, initial_blob, self.policy_version,
                    self.weight_queues[worker_id], self.data_queue, self.seed,
                    self.stop_event,
                ),
                daemon=True,
            )
            process.start()
            self.workers.append(process)

    def broadcast_weights(self, state_dict):
        self.policy_version += 1
        message = (self.policy_version, pickle.dumps(_cpu_state_dict(state_dict)))
        for weight_queue in self.weight_queues:
            try:
                while True:
                    weight_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                weight_queue.put_nowait(message)
            except queue.Full:
                pass

    def collect(self, timeout: float = 30.0) -> Optional[dict]:
        try:
            payload = self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if "__worker_done__" in payload else payload

    def drain(self, max_batches: int = None, timeout: float = 5.0):
        batches = []
        deadline = time.time() + timeout
        while max_batches is None or len(batches) < max_batches:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                payload = self.data_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if "__worker_done__" not in payload:
                batches.append(payload)
        return batches

    def stop(self):
        self.stop_event.set()
        for process in self.workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self.workers.clear()

    def train_loop(
        self,
        mappo,
        total_timesteps: int,
        rollouts_per_update: int = 8,
        n_epochs: int = 4,
        log_fn=None,
        save_path: str = None,
        save_every: int = 10,
        tb_writer=None,
        eval_fn=None,
        eval_every: int = 10,
        tag: str = "train",
    ):
        self.start(mappo.ac.state_dict())
        collected = 0
        update_count = 0
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []

        try:
            while collected < total_timesteps:
                payloads = self.drain(max_batches=rollouts_per_update, timeout=8.0)
                if not payloads:
                    continue
                for payload in payloads:
                    for summary in payload.get("episode_summaries", []):
                        episode_rewards.append(summary["reward"])
                        episode_lengths.append(summary["length"])

                merged = concatenate_rollouts(payloads)
                rewards = np.concatenate([payload["reward"] for payload in payloads])
                batch = to_tensors(merged, mappo.device)
                metrics = mappo.update(batch, n_epochs=n_epochs)
                collected += len(rewards)
                update_count += 1
                self.broadcast_weights(mappo.ac.state_dict())

                if tb_writer is not None:
                    last = metrics[-1] if metrics else {}
                    tb_writer.add_scalar(f"{tag}/policy_loss", last.get("policy_loss", 0), update_count)
                    tb_writer.add_scalar(f"{tag}/value_loss", last.get("value_loss", 0), update_count)
                    tb_writer.add_scalar(f"{tag}/entropy", last.get("entropy", 0), update_count)
                    tb_writer.add_scalar(f"{tag}/approx_kl", last.get("approx_kl", 0), update_count)
                    tb_writer.add_scalar(f"{tag}/mean_reward_per_step", float(rewards.mean()), update_count)
                    if episode_rewards:
                        tb_writer.add_scalar(f"{tag}/mean_episode_reward", float(np.mean(episode_rewards)), update_count)
                        tb_writer.add_scalar(f"{tag}/mean_episode_length", float(np.mean(episode_lengths)), update_count)
                        episode_rewards, episode_lengths = [], []
                    tb_writer.add_scalar(f"{tag}/collected_steps", collected, update_count)

                if eval_fn is not None and update_count % eval_every == 0:
                    try:
                        evaluation = eval_fn()
                        if tb_writer is not None and isinstance(evaluation, dict):
                            for key, value in evaluation.items():
                                if isinstance(value, (int, float)):
                                    tb_writer.add_scalar(f"{tag}/eval/{key}", float(value), update_count)
                    except Exception as error:
                        print(f"[eval] skipped: {error}")

                if log_fn:
                    log_fn(update_count, collected, metrics[-1] if metrics else {})
                if save_path and update_count % save_every == 0:
                    mappo.save(save_path)

            if save_path:
                mappo.save(save_path)
        finally:
            self.stop()
