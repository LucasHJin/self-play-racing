import numpy as np
import gymnasium as gym
from track import Track
from car import Car

class RacingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.track = Track()
        self.car = Car(self.track)
        
        # [steering, throttle]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        # [vx, vy, angle, angular_velocity, progress]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        ) 
        
    def _get_obs(self):
        pass
    
    def _get_info(self):
        pass
    
    def reset(self, seed=None, options=None):
        pass
    
    def step(self, action):
        pass