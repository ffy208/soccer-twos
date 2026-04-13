"""
League-style selfplay training:
- opponent_1, opponent_2: updated when default policy is strong (selfplay)
- opponent_3: FIXED as ceia_baseline weights (never updates)

This forces the default policy to learn to beat ceia_baseline directly.
Achieves ~15% direct exposure to ceia_baseline every episode.
"""
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3

# Adjust paths if different on server
CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

# Our best checkpoint (72% vs ceia_baseline)
RESTORE_CHECKPOINT = (
    "./ray_results/PPO_selfplay_rec/"
    "PPO_Soccer_d156b_00000_0_2026-04-12_14-38-37/checkpoint_001000/checkpoint-1000"
)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"
    else:
        # opponent_3 = fixed ceia_baseline, faces default 15% of time
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.50, 0.20, 0.15, 0.15],
        )[0]


def _load_policy_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Load a single policy's weights from a Ray checkpoint (skip optimizer state)."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class LeagueSelfPlayCallback(DefaultCallbacks):
    """
    - First iteration: set opponent_3 = ceia_baseline (once only).
    - Subsequent iterations: update opponent_1/2 when default reward > threshold.
    - opponent_3 is NEVER updated (stays as ceia_baseline forever).
    """

    def __init__(self):
        super().__init__()
        self._ceia_initialized = False

    def on_train_result(self, **info):
        trainer = info["trainer"]

        # Step 1: initialize opponent_3 as ceia_baseline on first call
        if not self._ceia_initialized:
            print("=== Initializing opponent_3 = ceia_baseline ===")
            try:
                ceia_weights = _load_policy_weights(CEIA_CHECKPOINT, policy_name="default")
                trainer.set_weights({"opponent_3": ceia_weights})
                self._ceia_initialized = True
                print("=== opponent_3 initialized as ceia_baseline ===")
            except Exception as e:
                print(f"WARNING: Could not load ceia_baseline: {e}")
                self._ceia_initialized = True  # don't retry every iter

        # Step 2: selfplay opponent update (only opponent_1 and opponent_2)
        default_reward = info["result"].get("policy_reward_mean", {}).get("default", -999)
        if default_reward > 0.3:
            print(f"---- Updating selfplay opponents (default_reward={default_reward:.3f}) ----")
            trainer.set_weights(
                {
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                    # opponent_3 intentionally NOT updated
                }
            )


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_league_ceia",
        config={
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": LeagueSelfPlayCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),  # fixed ceia_baseline
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 30000000, "time_total_s": 86400},  # 24h
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=RESTORE_CHECKPOINT,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
