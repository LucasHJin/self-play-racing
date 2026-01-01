import numpy as np
import torch
import gymnasium as gym

# wraps multi agent env to train single agent
class SelfPlayWrapper(gym.Wrapper):
    def __init__(self, env, agent_idx=0):
        pass
    
    def set_opponent(self, opp_policy):
        pass
    
    def reset(self, **kwargs): # type: ignore
        pass
    
    def step(self, action): # type: ignore
        pass