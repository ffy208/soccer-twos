# Reward Shaping PPO Agent (512)

**Agent name:** RewardShapingPPO_512

**Author(s):** Bo Feng (bfeng66@gatech.edu), Frank Yang

## Description

A PPO agent trained with league-style self-play against a fixed ceia_baseline opponent, using dense reward shaping to accelerate learning. The agent uses 6 shaped reward signals (approach, kick, offensive, defensive, time-step penalty, separation) on top of the sparse goal reward to guide exploration. Trained with Ray RLLib on the PACE-ICE cluster.

The network uses separate policy and value layers (`vf_share_layers=False`) with a larger [512, 512] MLP, which allows the value function to train independently without gradient interference from the policy — beneficial in the noisy multi-agent reward setting.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Network | MLP [512, 512], separate policy/value |
| Learning rate | 3e-4 (fixed) |
| GAE lambda | 0.95 |
| Clip param | 0.2 |
| Entropy coeff | 0.005 |
| VF loss coeff | 1.0 |
| SGD iterations | 10 |
| Train batch size | 8000 |
| SGD minibatch size | 512 |
| Rollout fragment length | 5000 |
| Workers | 7 × 3 envs |

## Checkpoint

`ray_results/PPO_large/PPO_Soccer_dc747_00000_0_2026-04-23_02-30-03/checkpoint_003400/checkpoint-3400`
