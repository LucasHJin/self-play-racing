import numpy as np
import gymnasium as gym
from .track import Track
from .car import Car

# https://gymnasium.farama.org/introduction/create_custom_env/

class RacingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.track = Track()
        self.car = Car(self.track)
        
        # for step function
        self.steps = 0
        self.last_progress = 0.0
        self.checkpoints = {
            0.25: False,
            0.50: False,
            0.75: False,
        }
        
        # [steering, throttle]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        # [vx, vy, angle, angular_velocity, progress]
        self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
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
            "crashed": self.car.crashed,
            "finished": self.car.finished
        }
        
    def reset(self, seed=None, options=None): # type: ignore
        super().reset(seed=seed)
        
        self.car.reset()
        self.steps = 0
        self.last_progress = 0.0
        self.checkpoints = {
            0.25: False,
            0.50: False,
            0.75: False,
        }
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action): # type: ignore
        # perform action
        steering, throttle = action
        self.car.update(steering, throttle)
        self.steps += 1
        
        progress_delta = self.car.progress - self.last_progress
        if self.last_progress > 0.9 and self.car.progress < 0.1:
            progress_delta = (1.0 - self.last_progress) + self.car.progress
        elif self.last_progress < 0.1 and self.car.progress > 0.9:
            progress_delta = -((1.0 - self.car.progress) + self.last_progress)
        
        # reward
        reward = 0.0
        # main signal -> reward progress 
        reward = progress_delta * 250
        # checkpoints to ensure no initial reward hacking
        if (not self.checkpoints[0.25] and 0.25 <= self.car.progress < 0.35):
            self.checkpoints[0.25] = True
            reward += 15
        if (self.checkpoints[0.25] and 
            not self.checkpoints[0.50] and 
            0.50 <= self.car.progress < 0.60):
            self.checkpoints[0.50] = True
            reward += 15
        if (self.checkpoints[0.50] and 
            not self.checkpoints[0.75] and 
            0.75 <= self.car.progress < 0.85):
            self.checkpoints[0.75] = True
            reward += 15
        # finished track
        all_checkpoints_passed = all(self.checkpoints.values())
        if (all_checkpoints_passed and self.last_progress > 0.9 and self.car.progress < 0.1 and progress_delta > 0):
            self.car.finished = True
            reward += 100
            
        # get returns
        observation = self._get_obs()
        info = self._get_info()
        info["reward"] = reward
        info["progress_delta"] = progress_delta
        terminated = self.car.crashed or self.car.finished
        truncated = self.steps >= 2000
        
        # update progress
        self.last_progress = self.car.progress
        
        return observation, reward, terminated, truncated, info