"""MAPPO (team mode, CTDE) / PPO (standard mode, self-play) trainer for Splendor.

Run on your Windows box with an RTX 5070:

    # 2v2 team mode -> MAPPO (centralized critic, team reward)
    python -m rl.mappo.main_train --mode team --timesteps 2000000 \
        --workers 7 --save checkpoints/mappo_team.pt --tb-runs runs/team

    # standard 4-player -> PPO (parameter-shared self-play, individual reward)
    python -m rl.mappo.main_train --mode standard --timesteps 2000000 \
        --workers 7 --save checkpoints/ppo_std.pt --tb-runs runs/standard

Recording (TensorBoard):
    tensorboard --logdir runs

Logged curves (prefix = team/ or standard/):
    policy_loss, value_loss, entropy, approx_kl, mean_reward_per_step,
    mean_episode_reward, mean_episode_length, collected_steps,
    eval/win_rate_seat0 (standard) or eval/win_rate_team0/team1 (team),
    eval/mean_score, eval/mean_length.

Self-play: during collection all 4 seats use the current (shared) policy, so the
agent is always playing against itself (IPPO-style self-play). Periodically we
evaluate the current policy as seat 0 against random opponents to track win rate.
"""

import argparse
import os
import sys
import time

# Defensive headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.mappo.algo import MAPPO
from rl.mappo.collector import DistributedCollector
from rl.mappo.encoders import ACTION_SPACE, OBS_DIM
from rl.mappo.selfplay import SelfPlayArena


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_argparser():
    p = argparse.ArgumentParser(description="Splendor MAPPO/PPO (self-play) trainer")
    p.add_argument("--mode", choices=["team", "standard"], default="team",
                   help="team -> MAPPO/CTDE; standard -> PPO/self-play")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=None, help="rollout workers (default cpu_count-1)")
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--target-score", type=int, default=None, help="override win score")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--rollouts-per-update", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save", type=str, default=None, help="checkpoint path")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--tb-runs", type=str, default="runs", help="TensorBoard log root")
    p.add_argument("--eval-every", type=int, default=10, help="eval + TB log frequency (updates)")
    p.add_argument("--eval-games", type=int, default=32, help="games per self-play eval")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", type=str, default=default_device())
    return p


def main():
    args = build_argparser().parse_args()

    team_mode = (args.mode == "team")
    target_score = args.target_score if args.target_score else (30 if team_mode else 15)
    if args.save is None:
        args.save = f"checkpoints/{'mappo_team' if team_mode else 'ppo_standard'}.pt"
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    os.makedirs(args.tb_runs, exist_ok=True)

    device = torch.device(args.device)
    print(f"[TRAIN] mode={args.mode} team_mode={team_mode} device={device} "
          f"workers={args.workers or 'auto'} agents={args.n_agents} "
          f"target={target_score} obs_dim={OBS_DIM} action_dim={ACTION_SPACE}")

    mappo = MAPPO(
        obs_dim=OBS_DIM, action_dim=ACTION_SPACE, n_agents=args.n_agents,
        global_dim=OBS_DIM, device=device,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        entropy_coef=args.entropy_coef, clip_eps=args.clip_eps,
    )

    collector = DistributedCollector(
        n_workers=args.workers, n_agents=args.n_agents, team_mode=team_mode,
        target_score=target_score, horizon=args.horizon, seed=args.seed,
        device=str(device),
    )

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    tag = "team" if team_mode else "standard"
    writer = SummaryWriter(log_dir=os.path.join(args.tb_runs, tag))

    # Self-play arena: current policy as seat0 vs random opponents
    arena = SelfPlayArena(team_mode=team_mode, target_score=target_score,
                          n_games=args.eval_games, seed=args.seed + 99)

    def eval_fn():
        sd = {k: v.cpu().clone() for k, v in mappo.ac.state_dict().items()}
        return arena.evaluate(sd, random_opponents=True)

    start = time.time()
    step_count = [0]
    update_count = [0]

    def log_fn(upd, collected, metrics):
        step_count[0] = collected
        update_count[0] = upd
        if upd % args.log_every == 0:
            el = time.time() - start
            print(f"[{tag} upd {upd:04d}] steps={collected:>8d} "
                  f"pol={metrics.get('policy_loss',0):.4f} "
                  f"val={metrics.get('value_loss',0):.4f} "
                  f"ent={metrics.get('entropy',0):.4f} "
                  f"kl={metrics.get('approx_kl',0):.4f} t={el:.1f}s")

    try:
        collector.train_loop(
            mappo, total_timesteps=args.timesteps,
            rollouts_per_update=args.rollouts_per_update, n_epochs=args.epochs,
            log_fn=log_fn, save_path=args.save, save_every=args.save_every,
            tb_writer=writer, eval_fn=eval_fn, eval_every=args.eval_every, tag=tag,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] interrupted; saving checkpoint")
        mappo.save(args.save)
        collector.stop()
        writer.close()
        sys.exit(0)

    writer.close()
    print(f"[TRAIN] done. steps={step_count[0]} updates={update_count[0]} "
          f"elapsed={time.time()-start:.1f}s -> {args.save}")


if __name__ == "__main__":
    main()
