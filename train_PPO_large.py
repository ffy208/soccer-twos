"""
Fresh training with larger network [512, 512] against fixed ceia_baseline.

Key differences from train_PPO_team.py:
  - Network: [512, 512], vf_share_layers=False (stable separate value head)
  - Entropy: annealed 0.01 → 0 over 30M timesteps (exploit deterministic ceia)
  - Opponent: 100% fixed ceia_baseline, no self-play arms race
  - Larger train_batch_size for more stable gradient estimates
  - Starts from scratch (no restore checkpoint)
"""
import os
import pickle

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env  # RewardShapingWrapper already inside


NUM_ENVS_PER_WORKER = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

RESTORE_CHECKPOINT = (
    "./ray_results/PPO_large/"
    "PPO_Soccer_f2e20_00000_0_2026-04-22_12-54-38/checkpoint_002500/checkpoint-2500"
)
# ─────────────────────────────────────────────────────────────────────────────


def policy_mapping_fn(agent_id, *args, **kwargs):
    """Team 0 trains; Team 1 is always the fixed ceia opponent."""
    if agent_id == 0 or agent_id == 1:
        return "default"
    return "opponent"


def _load_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Extract policy weights from a Ray checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class CeiaFixedCallback(DefaultCallbacks):
    """Loads ceia weights into opponent once at startup, never updates them."""

    def __init__(self):
        super().__init__()
        self._initialized = False

    def on_train_result(self, **info):
        if self._initialized:
            return
        trainer = info["trainer"]
        print("=== Loading ceia_baseline into opponent (fixed forever) ===")
        try:
            weights = _load_weights(CEIA_CHECKPOINT)
            trainer.set_weights({"opponent": weights})
            print("=== opponent = ceia_baseline, training begins ===")
        except Exception as e:
            print(f"WARNING: failed to load ceia_baseline: {e}")
        self._initialized = True


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    tmp = create_rllib_env()
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    analysis = tune.run(
        "PPO",
        name="PPO_large",
        config={
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CeiaFixedCallback,
            "multiagent": {
                "policies": {
                    "default":  (None, obs_space, act_space, {}),
                    "opponent": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": {
                "vf_share_layers": False,       # separate value head, more stable
                "fcnet_hiddens": [512, 512],    # 2x capacity vs [256, 256]
                "fcnet_activation": "relu",
            },
            "clip_param": 0.2,          # standard PPO (default 0.3 too aggressive)
            "lambda": 0.95,             # GAE (default 1.0 = high variance MC)
            "lr": 3e-4,                 # Unity Soccer recommended
            "vf_loss_coeff": 1.0,       # separate VF head, no interference
            # Anneal entropy 0.005 → 0 over training
            "entropy_coeff_schedule": [
                [0,        0.005],
                [20000000, 0.001],
                [50000000, 0.0],
            ],
            "train_batch_size": 8000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 80000000,
            "time_total_s": 259200,         # 72 h
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
