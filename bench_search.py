"""Search-throughput scaling benchmark (robust, no env wrapper, no torch).

The training bottleneck with a search opponent is the per-move alpha-beta cost
running inside worker subprocesses. This measures how many search moves/sec the
machine sustains as we add parallel workers (num_envs), and where adding more
workers stops helping (CPU saturation). That knee is the config that fills the
box. Net inference in the real loop is a small extra main-process cost, so the
optimal num_envs lands near the physical core count (8) or a bit above with HT.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import random
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from game_engine import XiangqiGame
from rl.search import SearchOpponent

CONFIGS = [
    ("depth=3 k=20", 20, 3, False),
    ("depth=3 k=16", 16, 3, False),
    ("depth=3 k=12", 12, 3, False),
    ("depth=3 k=8", 8, 3, False),
    ("depth=2 k=16", 16, 2, False),
    ("depth=2+quies k=16", 16, 2, True),
]
TASKS = 8
ROLLOUT_PER_TASK = 4  # each task plays this many search moves on one position


def _positions(n: int):
    out = []
    while len(out) < n:
        g = XiangqiGame()
        for _ in range(random.randint(0, 50)):
            mv = g.legal_moves()
            if not mv or g.winner is not None:
                break
            g.push(random.choice(mv))
        if g.winner is None:
            out.append(g)
    return out


def _search_seq(args):
    game, depth, quies = args
    opp = SearchOpponent(depth=depth, use_quiescence=quies)
    g = game.clone()
    for _ in range(ROLLOUT_PER_TASK):
        mv = opp.select_move(g)
        if mv is None:
            break
        g.push(mv)
        if g.winner is not None:
            break
    return True


def measure(k: int, depth: int, quies: bool) -> tuple[float, float]:
    positions = _positions(TASKS)
    args = [(p, depth, quies) for p in positions]
    cpu_samples: list[float] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.2))

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    ctx = mp.get_context("spawn")
    total_moves = TASKS * ROLLOUT_PER_TASK
    with cf.ProcessPoolExecutor(max_workers=k, mp_context=ctx) as ex:
        t0 = time.perf_counter()
        list(ex.map(_search_seq, args))
        dt = time.perf_counter() - t0
    stop.set()
    t.join(timeout=1)
    return total_moves / dt, float(np.mean(cpu_samples)) if cpu_samples else float("nan")


def main() -> None:
    print(f"physical cores = {psutil.cpu_count(logical=False)}, logical = {psutil.cpu_count()}", flush=True)
    print(f"{'config':<22}{'moves/s':>10}{'cpu%':>9}", flush=True)
    print("-" * 42, flush=True)
    for label, k, d, q in CONFIGS:
        print(f">>> {label} ...", flush=True)
        mps, cpu = measure(k, d, q)
        print(f"{label:<22}{mps:>10.1f}{cpu:>8.1f}%", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
