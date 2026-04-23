# Reward Shaping PPO Agent (256)

**Agent name:** RewardShapingPPO_256

**Author(s):** Bo Feng (bfeng66@gatech.edu), Frank Yang

## Description

A PPO agent trained with league-style self-play against a fixed ceia_baseline opponent, using dense reward shaping to accelerate learning. The agent uses 6 shaped reward signals (approach, kick, offensive, defensive, time-step penalty, separation) on top of the sparse goal reward to guide exploration. Trained with Ray RLLib on the PACE-ICE cluster.

The network uses shared policy and value layers (`vf_share_layers=True`) with a [256, 256] MLP, trained for 16,700 iterations (~120M timesteps budget).

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Network | MLP [256, 256], shared policy/value |
| Learning rate | 3e-4 (fixed) |
| GAE lambda | 0.95 |
| Clip param | 0.2 |
| Entropy coeff | 0.005 |
| VF loss coeff | 0.5 |
| Rollout fragment length | 5000 |
| Workers | 7 × 3 envs |

## Checkpoint

`ray_results/PPO_team/PPO_Soccer_cb83c_00000_0_2026-04-23_11-33-37/checkpoint_016700/checkpoint-16700`
