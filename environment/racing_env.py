import numpy as np
import gymnasium as gym
from track import Track
from car import Car

# https://gymnasium.farama.org/introduction/create_custom_env/

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
        return np.array([
            self.car.vx,
            self.car.vy,
            self.car.angle,
            self.car.angular_velocity,
            self.car.progress
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            "position": (self.car.x, self.car.y),
            "speed": np.sqrt(self.car.vx ** 2 + self.car.vy ** 2),
            "progress": self.car.progress,
            "crashed": self.car.crashed
        }
        
    def reset(self, seed=None, options=None):
        pass
    
    def step(self, action):
        pass