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
from rl.mappo.history import HistoryManager
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
    p.add_argument("--minibatch-size", type=int, default=1024,
                   help="GPU samples per PPO minibatch")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save", type=str, default=None, help="checkpoint path")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--history-dir", type=str, default="checkpoints/history",
                   help="root directory for timed historical checkpoints")
    p.add_argument("--history-every-seconds", type=float, default=600.0,
                   help="wall-clock seconds between historical snapshots")
    p.add_argument("--history-every-steps", type=int, default=0,
                   help="collected-step interval between snapshots (0 = disabled; "
                        "use this to build the opponent pool deterministically)")
    p.add_argument("--history-opponent-prob", type=float, default=0.5,
                   help="probability of an episode using historical opponents")
    p.add_argument("--history-pool-size", type=int, default=32,
                   help="newest historical models available to rollout workers")
    p.add_argument("--elo-games-per-match", type=int, default=4,
                   help="side-balanced games per historical Elo pairing")
    p.add_argument("--elo-convergence-window", type=int, default=6,
                   help="snapshots required for Elo convergence; 0 disables auto-stop")
    p.add_argument("--elo-convergence-threshold", type=float, default=25.0,
                   help="max newest-model Elo range across the window")
    p.add_argument("--elo-pool-size", type=int, default=40,
                   help="newest checkpoints entering the Elo round robin (older keep last rating)")
    p.add_argument("--tb-runs", type=str, default="runs", help="TensorBoard log root")
    p.add_argument("--run-name", type=str, default=None,
                   help="TB sub-directory name; default = mode name")
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

    # TensorBoard: each run gets its own sub-directory so curves do not get
    # polluted by old event files from previous runs.
    from torch.utils.tensorboard import SummaryWriter
    tag = args.run_name if args.run_name else ("team" if team_mode else "standard")
    tb_logdir = os.path.join(args.tb_runs, tag)
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    history = HistoryManager(
        directory=os.path.join(args.history_dir, args.mode),
        team_mode=team_mode,
        target_score=target_score,
        snapshot_every_seconds=args.history_every_seconds,
        snapshot_every_steps=args.history_every_steps,
        opponent_pool_size=args.history_pool_size,
        elo_games_per_match=args.elo_games_per_match,
        elo_pool_size=args.elo_pool_size,
        convergence_window=args.elo_convergence_window,
        convergence_threshold=args.elo_convergence_threshold,
        seed=args.seed,
    )

    device = torch.device(args.device)
    print(f"[TRAIN] mode={args.mode} team_mode={team_mode} device={device} "
          f"workers={args.workers or 'auto'} agents={args.n_agents} "
          f"target={target_score} obs_dim={OBS_DIM} action_dim={ACTION_SPACE} "
          f"tb_logdir={tb_logdir}")

    mappo = MAPPO(
        obs_dim=OBS_DIM, action_dim=ACTION_SPACE, n_agents=args.n_agents,
        global_dim=OBS_DIM, device=device,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        entropy_coef=args.entropy_coef, clip_eps=args.clip_eps,
    )

    collector = DistributedCollector(
        n_workers=args.workers, n_agents=args.n_agents, team_mode=team_mode,
        target_score=target_score, horizon=args.horizon, seed=args.seed,
        device=str(device), history_opponent_probability=args.history_opponent_prob,
    )

    # Self-play arena: current policy as seat0 vs random/pool opponents, plus
    # an internal symmetric self-play pass (every seat = current policy).
    arena = SelfPlayArena(team_mode=team_mode, target_score=target_score,
                          n_games=args.eval_games, seed=args.seed + 99)

    def eval_fn():
        sd = {k: v.cpu().clone() for k, v in mappo.ac.state_dict().items()}
        opponent_pool = history.opponent_pool()
        return arena.evaluate(
            sd,
            opponent_pool=opponent_pool or None,
            random_opponents=not opponent_pool,
        )

    def on_update(state_dict, collected, _update_count):
        if history.maybe_snapshot(state_dict, collected):
            pool = history.opponent_pool()
            if history.converged:
                print(
                    f"[history] Elo converged: span={history.convergence_span():.2f} "
                    f"over {args.elo_convergence_window} snapshots; stopping training"
                )
            else:
                print(f"[history] saved checkpoint #{len(history.snapshot_paths())}; opponent pool={len(pool)}")
            # --- log Elo / convergence into TensorBoard ---
            elos = history.latest_elo_history
            pool_scores = history.latest_pool_scores
            if elos:
                writer.add_scalar(f"{tag}/elo_latest", elos[-1], len(elos))
                writer.add_scalar(f"{tag}/elo_span", history.convergence_span() or 0.0, len(elos))
            if pool_scores:
                writer.add_scalar(f"{tag}/pool_score", pool_scores[-1], len(pool_scores))
            # Elo bar chart (written to disk by HistoryManager as elo.png) -> TB image
            elo_png = history.elo_plot_path()
            if elo_png and os.path.exists(elo_png):
                try:
                    import io
                    from PIL import Image
                    img = Image.open(elo_png).convert("RGB")
                    import numpy as np
                    writer.add_image(f"{tag}/elo_bar", np.array(img).transpose(2, 0, 1),
                                     len(elos), dataformats="CHW")
                except Exception as e:  # pragma: no cover - optional viz
                    print(f"[history] elo image log skipped: {e}")
            return {"opponent_pool": pool, "stop": history.converged}
        return None

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
            minibatch_size=args.minibatch_size,
            opponent_pool=history.opponent_pool(), on_update=on_update,
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
