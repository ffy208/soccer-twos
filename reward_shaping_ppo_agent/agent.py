import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../ray_results/PPO_selfplay_rec/PPO_Soccer_eb80a_00000_0_2026-04-12_02-57-50/checkpoint_000900/checkpoint-900", #ray 1.4 retrain version
)
POLICY_NAME = "default"


class RewardShapingPPOAgent(AgentInterface):
    """PPO selfplay agent loaded from local checkpoint."""

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # # ====[edstem#104] PATCH for DummyEnv =====
        # # https://edstem.org/us/courses/92385/discussion/7884350
        # from soccer_twos.utils import DummyEnv
        # from utils import RLLibWrapper

        # obs_space = env.observation_space
        # act_space = env.action_space
        # tune.registry.register_env("DummyEnv", lambda *_: RLLibWrapper(DummyEnv(obs_space, act_space)))

        # # create a dummy env since it's required but we only care about the policy
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        # # ==== END PATCH =====
        config["env"] = "DummyEnv"
        #config["disable_env_checking"] = True # Add this to avoid check the DummyEnv --- IGNORE ---

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint

        # ===== PATCH for Ray 1.4 checkpoint loading =====
        agent.restore(CHECKPOINT_PATH)
        # # replace:  agent.restore(CHECKPOINT_PATH)
        # with open(CHECKPOINT_PATH, "rb") as f:
        #     checkpoint_data = pickle.load(f)
        # worker_state = pickle.loads(checkpoint_data["worker"])
        # weights = {
        #     pid: {
        #         k: v for k, v in state.items()
        #         if k != "_optimizer_variables"
        #     }
        #     for pid, state in worker_state["state"].items()
        # }
        # agent.workers.local_worker().set_weights(weights)
        # ===== END PATCH =====
        
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)


    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions
