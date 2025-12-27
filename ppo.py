"""
Needed functions/logic:

parse_args
make_env
Agent class
train
    collect_roolout
    compute_advantages
    ppo_update
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def parse_args():
    parser = argparse.ArgumentParser(prog="PPO implementation")
    
    # simplified
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    
    # optimizations (TO BE DONE LATER)
    
    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)
    
    return args

def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed) # type: ignore
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            Agent.layer_optimization(nn.Linear(np.array(envs.single_action_space.shape).prod(), 64)),
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(64, 64)),
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(64, envs.single_action_space.n), std=0.01), # output logits for all possible actions
        )
        
        self.critic = nn.Sequential(
            Agent.layer_optimization(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), # flatten into total input features
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(64, 64)),
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(64, 1), std=1.0),
        )
        
    # for advantage calcs
    def get_value(self, obs):
        return self.critic(obs)
        
    # for interacting with env + optimizing parameters
    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs) # which action to take, clip ratio, exploration, advantages
    
    @staticmethod
    def layer_optimization(layer, std=np.sqrt(2), bias=0.0):
        torch.nn.init.orthogonal_(layer.weight, std) # stabilize initial weights
        torch.nn.init.constant_(layer.bias, bias) # more predictable training
        return layer
        
    
if __name__ == "__main__":
    pass