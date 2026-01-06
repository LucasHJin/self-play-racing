import numpy as np
import torch
import gymnasium as gym

# wraps multi agent env to train single agent
class SelfPlayWrapper(gym.Wrapper):
    def __init__(self, env, agent_idx=0):
        super().__init__(env)
        self.agent_idx = agent_idx
        if agent_idx == 0:
            self.opponent_idx = 1
        else:
            self.opponent_idx = 0
        self.action_space = env.action_space[f"{agent_idx}"]
        self.observation_space = env.observation_space[f"{agent_idx}"]
        self.opponent_policy = None
        self.opponent_action_space = env.action_space[f"{self.opponent_idx}"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_obs_dict = None # last obs to use for opponent policy
    
    def set_opponent(self, opponent_policy):
        self.opponent_policy = opponent_policy
    
    def reset(self, **kwargs): # type: ignore
        obs_dict, info_dict = self.env.reset(**kwargs) # obs dict contains info for both agents
        self.last_obs_dict = obs_dict 
        
        return obs_dict[f"{self.agent_idx}"], info_dict[f"{self.agent_idx}"]
    
    def step(self, action): # type: ignore        
        if self.opponent_policy is None:
            # random action opponent early on (pool not yet filled)
            opponent_action = self.opponent_action_space.sample()
        else:
            # use frozen opponent policy
            opponent_obs = self.last_obs_dict[f"{self.opponent_idx}"] # type: ignore
            opponent_obs_tensor = torch.from_numpy(opponent_obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                opponent_action_tensor, _, _, _ = self.opponent_policy.get_action_and_value(opponent_obs_tensor)
                opponent_action = opponent_action_tensor.squeeze(0).cpu().numpy()
        
        # combine actions to step through multi env
        actions = {
            f"{self.agent_idx}": action,
            f"{self.opponent_idx}": opponent_action
        }
        obs_dict, reward_dict, done_dict, truncated, info_dict = self.env.step(actions)
        self.last_obs_dict = obs_dict
        
        # get agent 0 experience to return
        obs = obs_dict[f"{self.agent_idx}"]
        reward = reward_dict[f"{self.agent_idx}"] # type: ignore
        done = done_dict["__all__"] # type: ignore
        info = info_dict[f"{self.agent_idx}"]
        
        return obs, reward, done, truncated, info
    
    @property
    def speed_weight(self):
        return self.env.speed_weight # type: ignore

    @speed_weight.setter  
    def speed_weight(self, value):
        self.env.speed_weight = value # type: ignore