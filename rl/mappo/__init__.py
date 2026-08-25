"""MAPPO (CTDE) training framework for Splendor.

Components:
- encoders: fixed-length obs / fixed-size action-space encoding + legal mask
- envs: SplendorMAEnv (turn-based multi-agent wrapper over game_engine)
- networks: shared Actor + centralized Critic (CTDE)
- buffer: rollout buffer with GAE
- algo: MAPPO update (clip + value + entropy)
- collector: headless multiprocessing distributed rollout collection
- main_train: CLI training entry point
"""

from .algo import MAPPO
from .encoders import ACTION_SPACE, OBS_DIM, encode_observation, build_action_mask
from .envs import SplendorMAEnv

__all__ = ["MAPPO", "SplendorMAEnv", "OBS_DIM", "ACTION_SPACE",
           "encode_observation", "build_action_mask"]
