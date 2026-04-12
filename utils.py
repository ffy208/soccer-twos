from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShapingWrapper(gym.core.Wrapper, MultiAgentEnv):
    # Goal x-positions: team 0 (agents 0/1) defends TEAM0_GOAL_X,
    #                   team 1 (agents 2/3) defends TEAM1_GOAL_X.
    # Adjust if coordinate inspection reveals different values.
    TEAM0_GOAL_X = -13.0
    TEAM1_GOAL_X = 13.0

    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_vel = None

    def reset(self):
        self.prev_ball_vel = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped = self._shaping(info)
        combined = {aid: reward[aid] + shaped[aid] for aid in reward}
        return obs, combined, done, info

    def _shaping(self, info) -> dict:
        shaped = {aid: 0.0 for aid in info}

        # Extract ball state from first agent that has it
        ball_pos = ball_vel = None
        for aid in info:
            if "ball_info" in info[aid]:
                ball_pos = np.array(info[aid]["ball_info"]["position"])
                ball_vel = np.array(info[aid]["ball_info"]["velocity"])
                break

        if ball_pos is None:
            return shaped  # binary env not active, no extra info

        for aid in info:
            agent_info = info[aid]
            if "player_info" not in agent_info:
                continue

            player_pos = np.array(agent_info["player_info"]["position"])
            player_vel = np.array(agent_info["player_info"]["velocity"])
            own_goal_x = self.TEAM0_GOAL_X if aid < 2 else self.TEAM1_GOAL_X
            attack_dir = 1.0 if aid < 2 else -1.0

            # Signal 1: approach reward — velocity component pointing toward ball
            to_ball = ball_pos - player_pos
            dist = np.linalg.norm(to_ball) + 1e-6
            approach_reward = np.dot(player_vel, to_ball / dist)  # [-v_max, v_max]
            shaped[aid] += 0.0002 * approach_reward  # reduced 50x: game signal must dominate

            # Signal 2: kick reward — ball speed increased since last step
            if self.prev_ball_vel is not None:
                delta_speed = np.linalg.norm(ball_vel) - np.linalg.norm(self.prev_ball_vel)
                if delta_speed > 0 and dist < 1.5:  # only credit nearby agent
                    shaped[aid] += 0.001 * delta_speed  # reduced 50x

            # Signal 3: offensive reward — ball moving toward opponent goal
            shaped[aid] += 0.0004 * ball_vel[0] * attack_dir  # reduced 50x

            # Signal 4: defensive penalty — ball dangerously close to own goal
            danger = max(0.0, 1.0 - abs(ball_pos[0] - own_goal_x) / 5.0)
            shaped[aid] -= 0.001 * danger  # reduced 50x

            # Signal 5: time-step penalty — discourage stalling
            shaped[aid] -= 0.00002  # reduced 50x

        # Signal 6: separation reward — teammates should cover different zones
        for team_start in (0, 2):
            ids = [team_start, team_start + 1]
            if all("player_info" in info.get(i, {}) for i in ids):
                pos0 = np.array(info[ids[0]]["player_info"]["position"])
                pos1 = np.array(info[ids[1]]["player_info"]["position"])
                sep = min(float(np.linalg.norm(pos0 - pos1)), 5.0)
                for i in ids:
                    shaped[i] += 0.0001 * sep  # reduced 50x

        self.prev_ball_vel = ball_vel

        # Signal 7: passing reward — if ball is moving toward teammate, reward the passer
        # TODO: implement this later
        return shaped



def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    # return RLLibWrapper(env)
    return RewardShapingWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
