import numpy as np
import torch
import gymnasium as gym
import copy
from ppo import PPO, Agent
from environment.multi_racing_env import MultiRacingEnv
from environment.wrappers import SelfPlayWrapper

class SelfPlayPPO(PPO):
    def __init__(self, env_fn, config, device="cuda"):
        pass
    
    def snapshot_agent(self):
        pass
    
    def select_opponent(self):
        pass
    
    def update_opponent_in_envs(self):
        pass
    
    def train(self):
        pass