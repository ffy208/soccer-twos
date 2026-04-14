from typing import Dict
import numpy as np
from soccer_twos import AgentInterface
import gym

class RandomAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.action_space = env.action_space

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return {pid: self.action_space.sample() for pid in observation}
