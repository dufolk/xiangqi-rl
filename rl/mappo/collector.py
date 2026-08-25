"""Headless distributed rollout collector (multiprocessing).

Architecture (maps to "distributed data collection, centralized training"):

    [ Worker 0 (CPU) ]  -> rollout batch ->
    [ Worker 1 (CPU) ]  -> rollout batch ->  [ Queue ] -> [ Trainer (GPU/RTX5070) ]
    [   ...           ]                      collects ->   MAPPO.update()
            ^                                         |
            |  weights broadcast (periodic) ----------+

- Each worker runs its own SplendorMAEnv instance headlessly (no pygame/display).
- Workers do *inference only* on CPU with a local copy of the policy; weights
  are pushed from the trainer periodically via a shared dict / queue.
- The trainer (main process) owns the GPU model, performs MAPPO updates, and
  pushes new weights back to workers.
- Fully headless: set environment variables SDL_VIDEODRIVER=dummy if pygame
  ever touches display (it does not in the game_ai path used here).

This design lets you scale collection across all CPU cores while keeping the
single 5070 busy on training. It is process-based so it is safe with pygame
and CPython threads, and inherently "distributed" (independent processes).
"""

import multiprocessing as mp
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .algo import MAPPO
from .encoders import ACTION_SPACE, OBS_DIM
from .envs import SplendorMAEnv


def _build_global(obs_per_agent: List[np.ndarray]) -> np.ndarray:
    """Concatenate per-agent obs into joint global state (CTDE critic input)."""
    return np.concatenate(obs_per_agent, axis=-1)


def worker_loop(
    worker_id: int,
    n_agents: int,
    team_mode: bool,
    target_score: int,
    horizon: int,
    weight_q: "mp.Queue",      # trainer -> worker weights (pickle of state_dict)
    data_q: "mp.Queue",        # worker -> trainer rollouts
    seed_base: int,
    stop_event: "mp.Event",
):
    """Worker process: collect `horizon` agent-steps then send a rollout."""
    # Ensure headless
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import torch
    # local CPU model
    from .networks import ActorCritic
    ac = ActorCritic(OBS_DIM, ACTION_SPACE, n_agents, OBS_DIM * n_agents)
    ac.eval()

    env = SplendorMAEnv(team_mode=team_mode, target_score=target_score,
                        seed=seed_base + worker_id)
    rng = np.random.RandomState(seed_base * 1000 + worker_id)

    local_buffers = {
        "obs": [], "global_state": [], "agent_id": [], "action": [],
        "logp": [], "reward": [], "mask": [], "value": [], "done": [],
    }

    while not stop_event.is_set():
        # pull latest weights if available (non-blocking)
        while not weight_q.empty():
            try:
                sd = weight_q.get_nowait()
                ac.load_state_dict(pickle.loads(sd))
                ac.eval()
            except Exception:
                break

        obs, gstate, mask, info = env.reset(seed=int(rng.randint(0, 2**30)))
        # initial global state requires all agents' obs; we approximate per-step
        # global as the concatenation of current obs repeated across agents
        # (the critic input dim = obs_dim * n_agents). To keep global coherent,
        # we store the *current* obs for all agents by reusing the latest seen
        # obs of each agent; for the first step we broadcast the single obs.
        agent_obs_memory = [obs.copy() for _ in range(n_agents)]

        episode_steps = 0
        ep_reward = 0.0
        done = False
        while not done and episode_steps < horizon:
            acting = info["acting_agent"]
            agent_obs_memory[acting] = obs.copy()
            global_state = _build_global(agent_obs_memory)

            # inference
            o_t = np.asarray(obs)[None]
            m_t = np.asarray(mask)[None]
            aid_t = np.array([acting], dtype=np.int64)
            logits = ac.actor(
                torch.as_tensor(o_t, dtype=torch.float32),
                torch.as_tensor(aid_t, dtype=torch.long),
                torch.as_tensor(m_t, dtype=torch.float32),
            )
            dist = __import__("torch").distributions.Categorical(logits=logits)
            action = dist.sample().cpu().numpy()[0]
            logp = dist.log_prob(torch.as_tensor([int(action)])).item()
            value = ac.critic(torch.as_tensor(global_state[None], dtype=torch.float32)).item()

            n_obs, n_gstate, n_mask, reward, terminated, n_info = env.step(int(action))
            ep_reward += reward

            local_buffers["obs"].append(obs)
            local_buffers["global_state"].append(global_state)
            local_buffers["agent_id"].append(acting)
            local_buffers["action"].append(int(action))
            local_buffers["logp"].append(logp)
            local_buffers["reward"].append(reward)
            local_buffers["mask"].append(mask)
            local_buffers["value"].append(value)
            local_buffers["done"].append(terminated)

            obs, mask, info = n_obs, n_mask, n_info
            if terminated:
                done = True
            episode_steps += 1

        # send rollout if non-empty
        if local_buffers["obs"]:
            payload = {
                "worker_id": worker_id,
                "obs": np.asarray(local_buffers["obs"], dtype=np.float32),
                "global_state": np.asarray(local_buffers["global_state"], dtype=np.float32),
                "agent_id": np.asarray(local_buffers["agent_id"], dtype=np.int64),
                "action": np.asarray(local_buffers["action"], dtype=np.int64),
                "logp": np.asarray(local_buffers["logp"], dtype=np.float32),
                "reward": np.asarray(local_buffers["reward"], dtype=np.float32),
                "mask": np.asarray(local_buffers["mask"], dtype=np.float32),
                "value": np.asarray(local_buffers["value"], dtype=np.float32),
                "done": np.asarray(local_buffers["done"], dtype=bool),
            }
            data_q.put(payload)
            for k in local_buffers:
                local_buffers[k].clear()
            # episode-level summary (only meaningful when truly terminated)
            if done:
                data_q.put({
                    "__episode__": worker_id,
                    "reward": float(ep_reward),
                    "length": int(episode_steps),
                })

    # signal exit
    data_q.put({"__worker_done__": worker_id})


class DistributedCollector:
    """Manages worker processes and aggregates rollouts for the trainer."""

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

        self.weight_q: "mp.Queue" = self.ctx.Queue(maxsize=n_workers * 2)
        self.data_q: "mp.Queue" = self.ctx.Queue(maxsize=n_workers * 4)
        self.stop_event = self.ctx.Event()
        self.workers: List[mp.Process] = []

    def start(self):
        self.stop_event.clear()
        for wid in range(self.n_workers):
            p = self.ctx.Process(
                target=worker_loop,
                args=(wid, self.n_agents, self.team_mode, self.target_score,
                      self.horizon, self.weight_q, self.data_q, self.seed,
                      self.stop_event),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def broadcast_weights(self, state_dict: dict):
        blob = pickle.dumps(state_dict)
        for _ in range(self.n_workers):
            self.weight_q.put(blob)

    def collect(self, timeout: float = 30.0) -> Optional[Dict[str, np.ndarray]]:
        """Fetch one rollout payload from any worker (blocking, with timeout)."""
        try:
            payload = self.data_q.get(timeout=timeout)
        except Exception:
            return None
        if "__worker_done__" in payload:
            return None
        return payload

    def drain(self, max_batches: int = None, timeout: float = 5.0):
        """Collect up to max_batches rollouts (or all currently available)."""
        batches = []
        t0 = time.time()
        while True:
            if max_batches and len(batches) >= max_batches:
                break
            try:
                p = self.data_q.get(timeout=timeout)
            except Exception:
                break
            if "__worker_done__" in p:
                continue
            batches.append(p)
            if time.time() - t0 > timeout * 4 and not max_batches:
                break
        return batches

    def stop(self):
        self.stop_event.set()
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.workers.clear()

    # convenience: train a MAPPO instance using this collector
    def train_loop(
        self,
        mappo: "MAPPO",
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
        """Training loop with optional TensorBoard logging and self-play eval.

        - tb_writer: torch.utils.tensorboard.SummaryWriter (or compatible).
        - eval_fn: callable() -> Dict[str, float] returning scalar stats to log
          (e.g. win-rate from a SelfPlayArena). Called every `eval_every` updates.
        - tag: prefix for TensorBoard scalar names (e.g. "team" / "standard").
        """
        import torch
        self.start()
        collected = 0
        update_count = 0
        ep_rewards: List[float] = []
        ep_lengths: List[int] = []

        while collected < total_timesteps:
            batches = self.drain(max_batches=rollouts_per_update, timeout=8.0)
            if not batches:
                # keep training loop alive; push weights once
                self.broadcast_weights(mappo.ac.state_dict())
                continue

            # split episode summaries from step payloads
            step_batches = []
            for b in batches:
                if "__episode__" in b:
                    ep_rewards.append(b["reward"])
                    ep_lengths.append(b["length"])
                else:
                    step_batches.append(b)
            batches = step_batches
            if not batches:
                continue

            # aggregate into a single buffer tensor set
            obs_list, g_list, aid_list, act_list = [], [], [], []
            logp_list, rew_list, mask_list, val_list, done_list = [], [], [], [], []
            for b in batches:
                obs_list.append(b["obs"])
                g_list.append(b["global_state"])
                aid_list.append(b["agent_id"])
                act_list.append(b["action"])
                logp_list.append(b["logp"])
                rew_list.append(b["reward"])
                mask_list.append(b["mask"])
                val_list.append(b["value"])
                done_list.append(b["done"])

            obs = np.concatenate(obs_list)
            gstate = np.concatenate(g_list)
            aid = np.concatenate(aid_list)
            act = np.concatenate(act_list)
            logp = np.concatenate(logp_list)
            rew = np.concatenate(rew_list)
            mask = np.concatenate(mask_list)
            val = np.concatenate(val_list)
            done = np.concatenate(done_list)

            # GAE using last value bootstrap from final non-terminal
            last_val = 0.0
            from .buffer import RolloutBuffer
            buf = RolloutBuffer(OBS_DIM, OBS_DIM * self.n_agents, ACTION_SPACE, capacity=len(rew) + 16)
            buf.obs[:len(rew)] = obs
            buf.global_state[:len(rew)] = gstate
            buf.agent_id[:len(rew)] = aid
            buf.action[:len(rew)] = act
            buf.logp[:len(rew)] = logp
            buf.reward[:len(rew)] = rew
            buf.mask[:len(rew)] = mask
            buf.value[:len(rew)] = val
            buf.done[:len(rew)] = done
            buf.ptr = len(rew)
            buf.size = len(rew)
            buf.compute_returns_advantages(last_value=last_val, gamma=0.99, lam=0.95)
            batch = buf.get_tensors(mappo.device)

            metrics = mappo.update(batch, n_epochs=n_epochs)
            collected += len(rew)
            update_count += 1

            # push new weights to workers
            self.broadcast_weights(mappo.ac.state_dict())

            # ---- TensorBoard ----
            if tb_writer is not None:
                last = metrics[-1] if metrics else {}
                tb_writer.add_scalar(f"{tag}/policy_loss", last.get("policy_loss", 0), update_count)
                tb_writer.add_scalar(f"{tag}/value_loss", last.get("value_loss", 0), update_count)
                tb_writer.add_scalar(f"{tag}/entropy", last.get("entropy", 0), update_count)
                tb_writer.add_scalar(f"{tag}/approx_kl", last.get("approx_kl", 0), update_count)
                tb_writer.add_scalar(f"{tag}/mean_reward_per_step", float(rew.mean()), update_count)
                if ep_rewards:
                    tb_writer.add_scalar(f"{tag}/mean_episode_reward", float(np.mean(ep_rewards)), update_count)
                    tb_writer.add_scalar(f"{tag}/mean_episode_length", float(np.mean(ep_lengths)), update_count)
                    ep_rewards, ep_lengths = [], []
                tb_writer.add_scalar(f"{tag}/collected_steps", collected, update_count)

            # ---- self-play evaluation ----
            if eval_fn is not None and update_count % eval_every == 0:
                try:
                    eval_stats = eval_fn()
                    if tb_writer is not None and isinstance(eval_stats, dict):
                        for k, v in eval_stats.items():
                            if isinstance(v, (int, float)):
                                tb_writer.add_scalar(f"{tag}/eval/{k}", float(v), update_count)
                except Exception as e:
                    print(f"[eval] skipped: {e}")

            if log_fn:
                last = metrics[-1] if metrics else {}
                log_fn(update_count, collected, last)

            if save_path and update_count % save_every == 0:
                mappo.save(save_path)

        if save_path:
            mappo.save(save_path)
        self.stop()
