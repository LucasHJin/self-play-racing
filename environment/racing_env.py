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
        self.half_passed = False
        
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
        self.half_passed = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action): # type: ignore
        # perform action
        steering, throttle = action
        self.car.update(steering, throttle)
        self.steps += 1
        
        # update checkpoint status
        if not self.half_passed and self.car.progress >= 0.5:
            self.half_passed = True
        
        # reward shaping
        lap_completed = False
        progress_delta = self.car.progress - self.last_progress
        if self.half_passed and self.last_progress > 0.9 and self.car.progress < 0.1:
            self.car.finished = True
            lap_completed = True
            progress_delta = (1.0 - self.last_progress) + self.car.progress  # fix delta across wrap
            
        speed = np.sqrt(self.car.vx**2 + self.car.vy**2)
        reward = 0.0
        reward += progress_delta * 500.0
        if lap_completed:
            reward += 800.0
        if progress_delta > 0:
            reward += speed * 0.1
        elif progress_delta < 0:
            reward += progress_delta * 1000
        if self.car.crashed:
            reward -= 2000.0
        if speed < 5.0:
            reward -= 2.0
            
        # get returns
        observation = self._get_obs()
        info = self._get_info()
        info["reward"] = reward
        info["progress_delta"] = progress_delta
        info["speed"] = speed
        info["half_passed"] = self.half_passed
        terminated = self.car.crashed or self.car.finished
        truncated = self.steps >= 1000
        
        # update progress
        self.last_progress = self.car.progress
        
        return observation, reward, terminated, truncated, info