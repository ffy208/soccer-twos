"""
League-style selfplay training against ceia_baseline.

Team structure (2v2):
  Team 0: agent_id 0, 1 → always "default"  (your team, trained together)
  Team 1: agent_id 2, 3 → sampled from opponent pool  (never trained)

Opponent pool:
  opponent_1: rolling selfplay snapshot (most recent)
  opponent_2: rolling selfplay snapshot (older)
  opponent_3: FIXED ceia_baseline weights (never updated)

Opponent update rule:
  When policy_reward_mean/default > OPPONENT_UPDATE_THRESHOLD, shift the rolling snapshots:
    opponent_2 ← opponent_1
    opponent_1 ← default
  opponent_3 stays as ceia_baseline forever.
"""
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
OPPONENT_UPDATE_THRESHOLD = 0.08
OPPONENT_UPDATE_COOLDOWN = 20  # minimum iterations between updates

# ── Paths (relative to project root; adjust if needed) ───────────────────────
CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

RESTORE_CHECKPOINT = (
    "./ray_results/PPO_team/"
    "PPO_Soccer_17b64_00000_0_2026-04-14_16-18-26/checkpoint_004500/checkpoint-4500"
)
# ─────────────────────────────────────────────────────────────────────────────


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Team 0 (agents 0, 1) always use the policy being trained ("default").
    Team 1 (agents 2, 3) sample from the fixed opponent pool.
    """
    if agent_id == 0 or agent_id == 1:
        return "default"
    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        p=[0.30, 0.20, 0.50],  # 50% vs ceia_baseline, sustained pressure on fixed benchmark
    )


def _load_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Extract policy weights from a Ray checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class LeagueCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._ceia_initialized = False
        self._last_update_iter = -OPPONENT_UPDATE_COOLDOWN

    def on_train_result(self, **info):
        trainer = info["trainer"]

        # On the very first iteration: load ceia_baseline into opponent_3
        if not self._ceia_initialized:
            print("=== Loading ceia_baseline into opponent_3 ===")
            try:
                weights = _load_weights(CEIA_CHECKPOINT)
                trainer.set_weights({"opponent_3": weights})
                print("=== opponent_3 = ceia_baseline (fixed forever) ===")
            except Exception as e:
                print(f"WARNING: failed to load ceia_baseline: {e}")
            self._ceia_initialized = True

        # Shift rolling selfplay snapshots when default is winning enough
        # Cooldown prevents cluster updates from reducing opponent diversity
        current_iter = info["result"]["training_iteration"]
        default_reward = info["result"].get("policy_reward_mean", {}).get("default", -999)
        since_last = current_iter - self._last_update_iter

        if default_reward > OPPONENT_UPDATE_THRESHOLD and since_last >= OPPONENT_UPDATE_COOLDOWN:
            print(f"---- Promoting opponents (default_reward={default_reward:.3f}, gap={since_last}) ----")
            trainer.set_weights({
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
                # opponent_3 intentionally skipped
            })
            self._last_update_iter = current_iter


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    tmp = create_rllib_env()
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    analysis = tune.run(
        "PPO",
        name="PPO_team",
        config={
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": LeagueCallback,
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
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
        stop={
            "timesteps_total": 60000000,
            "time_total_s": 259200,   # 72 h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=RESTORE_CHECKPOINT,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_trial)
    print(best_ckpt)
    print("Done training")
