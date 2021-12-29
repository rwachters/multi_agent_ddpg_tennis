from my_unity_environment import MyUnityEnvironment
from utilities import convert_to_tensor
import numpy as np
import torch
from typing import Tuple, Deque, Union


class TennisEnvironment(MyUnityEnvironment):
    def step(self, actions: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """ step forward in environment
                Params
                ======
                    actions: actions for each agent.
                """
        self.brain_info = self.env.step(actions)[self.brain_name]
        states = convert_to_tensor(self.brain_info.vector_observations)
        rewards = [(-0.1 if reward < 0 else reward) for reward in self.brain_info.rewards]
        rewards = convert_to_tensor(rewards)[:, None]  # add dimension at the end
        dones = convert_to_tensor(self.brain_info.local_done)[:, None]
        return states, rewards, dones
