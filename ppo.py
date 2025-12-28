"""
Needed functions/logic:

parse_args
make_env
Agent class
train
    collect_rollout
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
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    
    # optimizations (TO BE DONE LATER)
    
    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)
    args.minibatch_size = int(args.num_steps // args.num_minibatches)
    
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
        
def collect_rollout(agent, envs, args, obs, actions, logprobs, dones, rewards, values):
    init_obs, _ = envs.reset()
    # keep track of observations + if it's done to progress the rollout (remember to convert to gpu)
    next_obs = torch.from_numpy(init_obs).to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    reward_tensor = torch.empty(args.num_envs, dtype=torch.float32, device=device) # preallocate reward tensor 
    
    # no gradient update -> just collecting data
    with torch.no_grad():
        for step in range(args.num_steps):
            obs[step].copy_(next_obs) # for each step, you give it shape (envs, obs) - also use copy instead of =
            dones[step].copy_(next_done) # use done for bootstrapping
            
            action, logprob, _, value = agent.get_action_and_value(next_obs) # returns tensors
            actions[step].copy_(action)
            logprobs[step].copy_(logprob)
            values[step].copy_(value.flatten()) # (num_envs, 1) -> (num_envs,)
                
            # step through and repeat
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy()) # gym envs expect cpu
            
            reward_tensor.copy_(torch.from_numpy(reward).to(device).flatten())
            rewards[step].copy_(reward_tensor)
            
            next_obs.copy_(torch.from_numpy(next_obs).to(device))
            next_done.copy_(torch.from_numpy(np.logical_or(terminated, truncated)).to(device))
    
    return obs, actions, logprobs, dones, rewards, values, next_obs, next_done
        
def compute_advantages(args, rewards, dones, values, next_value, next_done, gamma):
    with torch.no_grad():
        returns = torch.zeros_like(rewards, device=device)
        for t in reversed(range(args.num_steps)): # need to reverse because we bootstrap based on future rewards
            if t == args.num_steps - 1:
                next_nonterminal = 1.0 - next_done 
                next_return = next_value # bootstrap extra value from critic
            else:
                next_nonterminal = 1.0 - dones[t+1]
                next_return = returns[t+1]
                
            returns[t] = rewards[t] + gamma * next_nonterminal * next_return # non terminal checks if we should add it in or already stopped
        advantages = returns - values # compute once at the end
        
        return advantages, returns
            
def ppo_update(agent, args, advantages, returns, logprobs, values, actions, obs, optimizer):
    # flatten batch data (num_steps * num_envs, [optional data]) -> no time data needed
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:])
    b_logprobs = logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_inds = np.arange(args.batch_size) # for shuffling + minibatch
    
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start_idx in range(0, args.batch_size, args.minibatch_size):
            end_idx = start_idx + args.minibatch_size
            mb_inds = b_inds[start_idx:end_idx]
            
            # pass action back in to get new probability + value using new policy
            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) # long because logprob needs int
            ratio = (new_logprob - b_logprobs[mb_inds]).exp()
            
            # policy loss
            # note -> signs in formula show you what to max + min (pytorch naturally minimizes)
            pg_loss1 = -b_advantages[mb_inds] * ratio
            pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean() # mean averages E[] into single scalar
            # value loss (minimize -> keep positive)
            new_value = new_value.flatten()
            v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
            #entropy loss
            e_loss = -entropy.mean()
            
            # total loss
            loss = pg_loss + args.ent_coef * e_loss + args.vf_coef * v_loss
            
            optimizer.zero_grad()
            loss.backwards()
            optimizer.step()
    
def train(agent, envs, args, optimizer):
    # predefine storage buffer -> for each step + env, store a specific piece of data (more efficient than lists)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    
    NUM_UPDATES = args.total_timesteps // args.batch_size
    
    for update in range(NUM_UPDATES):
        # 2 phase loop
            # collect experience with current policy -> rollout
            # use experience to update policy + value function (actor + critic) -> compute advantage, update ppo
        obs, actions, logprobs, dones, rewards, values, next_obs, next_done = collect_rollout(agent, envs, args, obs, actions, logprobs, dones, rewards, values)
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
        
        advantages, returns = compute_advantages(args, rewards, dones, values, next_value, next_done, args.gamma)
        
        ppo_update(agent, args, advantages, returns, logprobs, values, actions, obs, optimizer)
    
if __name__ == "__main__":
    args = parse_args()
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed) for _ in range(args.num_envs)])
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = Agent(envs).to(device) # obs + agent need to be on same device (pref gpu)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    train(agent, envs, args, optimizer)