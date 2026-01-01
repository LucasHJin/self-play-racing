import numpy as np
import gymnasium as gym
from .multi_track import MultiTrack
from .multi_car import MultiCar

# https://gymnasium.farama.org/introduction/create_custom_env/

class MultiRacingEnv(gym.Env):
    def __init__(self, num_agents=2, num_sensors=11):
        super().__init__()
        
        self.track = MultiTrack()
        self.num_agents = num_agents
        self.cars = [MultiCar(self.track) for _ in range(num_agents)]
        self.num_sensors = num_sensors
        self.max_sensor_range = 50.0
        
        # for step function
        self.steps = 0
        self.agents_data = [{
            "last_progress": 0.0,
            "last_steering": 0.0,
            "checkpoints": {0.25: False, 0.50: False, 0.75: False},
            "finished_step": None,
        } for _ in range(num_agents)]
        
        # per agent -> [steering, throttle]
        self.action_space = gym.spaces.Dict({
            f"{i}": gym.spaces.Box(
                low=np.array([-1.0, 0.0]),
                high=np.array([1.0, 1.0]),
                shape=(2,),
                dtype=np.float32
            ) for i in range(num_agents)
        })
        # per agent -> [sensor_distances, v_forward, v_lateral, angular_velocity, steering, (relative_pos_x, relative_pos_y, relative_vel_x, relative_vel_y)]
        # ALL NORMALIZED -> [0, 1] or [-1, 1]
        obs_dim = num_sensors + 4 + (num_agents - 1) * 4
        self.observation_space = gym.spaces.Dict({
            f"{i}": gym.spaces.Box(
                low=np.float32(-1.0),
                high=np.float32(1.0),
                shape=(obs_dim,),
                dtype=np.float32
            ) for i in range(num_agents)
        })
        
    def get_sensor_readings(self, agent_idx):
        car = self.cars[agent_idx]
        sensor_angles = np.linspace(-np.pi/2, np.pi/2, self.num_sensors) # front cone
        distances = np.zeros(self.num_sensors, dtype=np.float32)
        origin = np.array([car.x, car.y])
        
        for i, relative_angle in enumerate(sensor_angles):
            world_angle = car.angle + relative_angle
            distances[i] = self.track.raycast_with_cars(origin, world_angle, self.cars, self.max_sensor_range)
        
        return distances / self.max_sensor_range # normalize 0-1
        
    def _get_obs(self, agent_idx):
        car = self.cars[agent_idx]
        sensor_readings = self.get_sensor_readings(agent_idx)
        
        # velocities in local car's frame
        cos_angle = np.cos(car.angle)
        sin_angle = np.sin(car.angle)
        v_forward = car.vx * cos_angle + car.vy * sin_angle
        v_lateral = -car.vx * sin_angle + car.vy * cos_angle
        
        # normalize to [-1, 1]
        v_forward = np.clip(v_forward / MultiCar.MAX_SPEED, -1.0, 1.0)
        v_lateral = np.clip(v_lateral / MultiCar.MAX_SPEED, -1.0, 1.0)
        angular_velocity = np.clip(car.angular_velocity / MultiCar.STEERING_SPEED, -1.0, 1.0)
        steering = self.agents_data[agent_idx]["last_steering"]
        
        opp_features = []
        for idx, other_car in enumerate(self.cars):
            if idx == agent_idx:
                continue
            
            # normalized relative position in local frame
            rel_x = other_car.x - car.x
            rel_y = other_car.y - car.y
            local_rel_x = rel_x * cos_angle + rel_y * sin_angle
            local_rel_y = -rel_x * sin_angle + rel_y * cos_angle
            local_rel_x = np.clip(local_rel_x / self.track.max_track_distance, -1.0, 1.0)
            local_rel_y = np.clip(local_rel_y / self.track.max_track_distance, -1.0, 1.0)
            
            # normalized relative velocity in local frame
            rel_vx = other_car.vx - car.vx
            rel_vy = other_car.vy - car.vy
            local_rel_vx = rel_vx * cos_angle + rel_vy * sin_angle
            local_rel_vy = -rel_vx * sin_angle + rel_vy * cos_angle
            local_rel_vx = np.clip(local_rel_vx / MultiCar.MAX_SPEED, -1.0, 1.0)
            local_rel_vy = np.clip(local_rel_vy / MultiCar.MAX_SPEED, -1.0, 1.0)
            
            opp_features.extend([local_rel_x, local_rel_y, local_rel_vx, local_rel_vy])
        
        obs = np.concatenate([
            sensor_readings,
            [v_forward, v_lateral, angular_velocity, steering],
            opp_features
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self, agent_idx):
        car = self.cars[agent_idx]
        
        return {
            "position": (car.x, car.y),
            "speed": np.sqrt(car.vx ** 2 + car.vy ** 2),
            "progress": car.progress,
            "crashed": car.crashed,
            "finished": car.finished,
        }
        
    def reset(self, seed=None, options=None): # type: ignore
        super().reset(seed=seed)
        
        for i, car in enumerate(self.cars):
            car.reset()
            if i > 0:
                car.y += i * (MultiCar.WIDTH + 0.5) # stagger starting position
        
        self.steps = 0
        for idx in range(len(self.cars)):
            self.agents_data[idx] = {
                "last_progress": 0.0,
                "last_steering": 0.0,
                "checkpoints": {0.25: False, 0.50: False, 0.75: False},
                "finished_step": None,
            }
            
        observations = {f"{i}": self._get_obs(i) for i in range(self.num_agents)}
        infos = {f"{i}": self._get_info(i) for i in range(self.num_agents)}
        
        return observations, infos
    
    def calc_reward(self, agent_idx):
        car = self.cars[agent_idx]
        data = self.agents_data[agent_idx]
        
        progress_delta = car.progress - data['last_progress']
        if data['last_progress'] > 0.9 and car.progress < 0.1:
            progress_delta = (1.0 - data['last_progress']) + car.progress
        elif data['last_progress'] < 0.1 and car.progress > 0.9:
            progress_delta = -((1.0 - car.progress) + data['last_progress'])
        
        reward = 0.0
        # main signal -> reward progress 
        reward += progress_delta * 250
        # checkpoints to ensure no initial reward hacking
        if (not data['checkpoints'][0.25] and 0.25 <= car.progress < 0.35):
            data['checkpoints'][0.25] = True
            reward += 15
        if (data['checkpoints'][0.25] and not data['checkpoints'][0.50] and 0.50 <= car.progress < 0.60):
            data['checkpoints'][0.50] = True
            reward += 15
        if (data['checkpoints'][0.50] and not data['checkpoints'][0.75] and 0.75 <= car.progress < 0.85):
            data['checkpoints'][0.75] = True
            reward += 15
        # finished track
        all_checkpoints_passed = all(data['checkpoints'].values())
        if (all_checkpoints_passed and data['last_progress'] > 0.9 and car.progress < 0.1 and progress_delta > 0):
            car.finished = True
            data['finished_step'] = self.steps
            reward += 100
        # crash penalty
        if car.crashed:
            reward -= 50
        
        return reward
    
    def place(self):
        scores = []
        for idx, car in enumerate(self.cars):
            score = ( # make sure no ties
                car.finished * 10000 + 
                car.progress * 100 + 
                (not car.crashed) * 10 + 
                (1.0 / (self.agents_data[idx]['finished_step'] or 10000))
            )
            scores.append((score, idx))
        scores.sort(reverse=True)
        
        for placement, (_, idx) in enumerate(scores):
            self.agents_data[idx]['placement'] = placement+1
    
    def step(self, actions): # type: ignore
        # perform action (clip just in case)
        for idx, car in enumerate(self.cars):
            steering = float(np.clip(actions[f"{idx}"][0], -1.0, 1.0)) 
            throttle = float(np.clip(actions[f"{idx}"][1],  0.0, 1.0))
            self.agents_data[idx]['last_steering'] = steering
            car.update(steering, throttle)
            
        # car collision check
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.cars[i].check_car_collision([self.cars[j]]):
                    self.cars[i].vx *= 0.85
                    self.cars[i].vy *= 0.85
                    self.cars[j].vx *= 0.85
                    self.cars[j].vy *= 0.85
                
        self.steps += 1
        
        rewards = {}
        observations = {}
        infos = {}
        
        for i in range(self.num_agents):
            rewards[f"{i}"] = self.calc_reward(i)
            observations[f"{i}"] = self._get_obs(i)
            infos[f"{i}"] = self._get_info(i)
        
        any_finished = any(car.finished for car in self.cars)
        all_crashed = all(car.crashed for car in self.cars)
        terminated = any_finished or all_crashed
        truncated = self.steps >= 3000
        
        if terminated or truncated:
            self.place()
            for i in range(self.num_agents):
                placement = self.agents_data[i]['placement']
                if placement == 1:
                    rewards[f"{i}"] += 200
                infos[f"{i}"]["reward"] = rewards[f"{i}"]
                infos[f"{i}"]["placement"] = placement
        else:
            for i in range(self.num_agents):
                infos[f"{i}"]["reward"] = rewards[f"{i}"]
                
        dones = {f"{i}": terminated for i in range(self.num_agents)}
        dones["__all__"] = terminated or truncated
        for i in range(self.num_agents):
            self.agents_data[i]['last_progress'] = self.cars[i].progress
        
        return observations, rewards, dones, truncated, infos