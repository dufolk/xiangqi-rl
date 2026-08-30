"""Spawn-safe subprocess vector environment for parallel Xiangqi rollouts."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Optional


def _worker(connection: Connection, max_steps: int, start_fen: Optional[str], opponent=None) -> None:
    from rl.environment import XiangqiEnv

    env = XiangqiEnv(max_steps=max_steps, start_fen=start_fen, opponent=opponent)
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(env.reset(seed=payload))
            elif command == "step":
                result = env.step(int(payload))
                connection.send((result.observation, result.reward, result.terminated, result.truncated, result.info))
            elif command == "close":
                connection.close()
                break
            else:
                raise ValueError(f"unknown vector environment command: {command}")
    except EOFError:
        connection.close()


class SubprocessVectorEnv:
    """Run independent Xiangqi environments in spawned worker processes.

    The learner stays in the main process, so batched policy inference remains
    on MPS/CUDA. Workers only perform Python rules evaluation and legal move
    generation in parallel.
    """

    def __init__(self, num_envs: int, max_steps: int = 400, start_fen: Optional[str] = None, opponent=None):
        if num_envs < 1:
            raise ValueError("num_envs must be at least one")
        self.num_envs = num_envs
        self._closed = False
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        for _ in range(num_envs):
            main_connection, worker_connection = context.Pipe()
            process = context.Process(target=_worker, args=(worker_connection, max_steps, start_fen, opponent), daemon=True)
            process.start()
            worker_connection.close()
            self._connections.append(main_connection)
            self._processes.append(process)

    def reset(self, seeds: Optional[list[int]] = None):
        if seeds is None:
            seeds = [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("seed count must equal num_envs")
        for connection, seed in zip(self._connections, seeds):
            connection.send(("reset", seed))
        return [connection.recv() for connection in self._connections]

    def reset_at(self, indices: list[int], seeds: Optional[list[int]] = None):
        if seeds is None:
            seeds = [None] * len(indices)
        if len(indices) != len(seeds):
            raise ValueError("index count must equal seed count")
        for index, seed in zip(indices, seeds):
            self._connections[index].send(("reset", seed))
        return [self._connections[index].recv() for index in indices]

    def step(self, actions: list[int]):
        if len(actions) != self.num_envs:
            raise ValueError("action count must equal num_envs")
        for connection, action in zip(self._connections, actions):
            connection.send(("step", int(action)))
        return [connection.recv() for connection in self._connections]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
