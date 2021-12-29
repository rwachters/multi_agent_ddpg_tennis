import numpy as np
import torch
from unityagents import UnityEnvironment
from utilities import convert_to_tensor
from typing import Tuple


class MyUnityEnvironment:
    def __init__(self, file_name, no_graphics=False):
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.brain_info = self.env.reset(train_mode=True)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = len(self.get_states()[0])
        self.num_agents = len(self.brain_info.agents)

    def reset(self, train_mode: bool = False):
        self.brain_info = self.env.reset(train_mode=train_mode)[self.brain_name]

    def get_states(self):
        return convert_to_tensor(self.brain_info.vector_observations)

    def step(self, actions: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """ step forward in environment
        Params
        ======
            actions: actions for each agent.
        """
        self.brain_info = self.env.step(actions)[self.brain_name]
        states = convert_to_tensor(self.brain_info.vector_observations)
        rewards = convert_to_tensor(self.brain_info.rewards)[:, None] # add dimension at the end
        dones = convert_to_tensor(self.brain_info.local_done)[:, None]
        return states, rewards, dones

    def close(self):
        self.env.close()

