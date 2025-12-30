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
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action): # type: ignore
        # perform action
        steering, throttle = action
        self.car.update(steering, throttle)
        self.steps += 1
        
        # reward shaping
        progress_delta = self.car.progress - self.last_progress
        speed = np.sqrt(self.car.vx**2 + self.car.vy**2)
        reward = 0.0
        reward += progress_delta * 20.0
        reward += speed * 0.2
        if self.last_progress > 0.9 and self.car.progress < 0.1:
            reward += 50.0
            self.car.finished = True
        if self.car.crashed:
            reward -= 100.0
            
        # get returns
        observation = self._get_obs()
        info = self._get_info()
        info["reward"] = reward
        info["progress_delta"] = progress_delta
        info["speed"] = speed
        terminated = self.car.crashed or self.car.finished
        truncated = self.steps >= 1000
        
        # update progress
        self.last_progress = self.car.progress
        
        return observation, reward, terminated, truncated, info