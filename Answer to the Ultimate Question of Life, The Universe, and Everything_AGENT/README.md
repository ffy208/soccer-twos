# Reward Shaping PPO Agent

**Agent name:** RewardShapingPPO

**Author(s):** Bo Feng (bfeng66@gatech.edu), Frank Yang (frank.yang@gatech.edu)

## Description

A PPO agent trained with league-style self-play against a fixed baseline opponent, using dense reward shaping to accelerate learning. The agent uses 6 shaped reward signals (approach, kick, offensive, defensive, time-step penalty, separation) on top of the sparse goal reward to guide exploration. Trained with Ray RLLib on the PACE-ICE cluster.
