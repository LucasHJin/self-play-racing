import numpy as np
import torch
import gymnasium as gym
import copy
from .ppo import PPO, Agent
from environment.wrappers import SelfPlayWrapper

class SelfPlayPPO(PPO):
    def __init__(self, env_fn, config, device="cuda"):
        self.env_fn = env_fn
        # self play configs
        self.opponent_pool = []
        self.curr_opponent = None
        self.snapshot_freq = config["snapshot_freq"]
        self.pool_size = config["pool_size"]
        
        super().__init__(env_fn, config, device) 
    
    def _make_env(self, env_fn, seed, env_idx=0):
        def thunk():
            multi_env = self.env_fn(env_idx)
            env = SelfPlayWrapper(multi_env, 0) # initialize base ppo with wrapped env for 1 agent processing
            env.set_opponent(self.curr_opponent)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    def snapshot_agent(self):
        snapshot = Agent(self.envs.single_observation_space, self.envs.single_action_space).to(self.device) # new agent with same architecture
        snapshot.load_state_dict(copy.deepcopy(self.agent.state_dict())) # deep copy over weights (inner objects as well)
        # no training
        snapshot.eval() 
        for param in snapshot.parameters():
            param.requires_grad = False
        return snapshot
    
    def select_opponent(self):
        if not self.opponent_pool:
            return None
        
        return np.random.choice(self.opponent_pool)
    
    def update_opponent(self):
        self.curr_opponent = self.select_opponent()
        # make new envs for this rollout with new opponent
        self.envs.close() 
        self.envs = gym.vector.SyncVectorEnv([self._make_env(self.env_fn, self.config["seed"] + i, env_idx=i) for i in range(self.config["num_envs"])])
        
    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.opponent_pool = []
        for i, opp_state_dict in enumerate(checkpoint['opponent_pool']):
            opponent = Agent(self.envs.single_observation_space, self.envs.single_action_space).to(self.device)
            opponent.load_state_dict(opp_state_dict)
            opponent.eval()
            for param in opponent.parameters():
                param.requires_grad = False
            self.opponent_pool.append(opponent)
        training_info = checkpoint.get('training_info', {'steps': [], 'rewards': [], 'opponent_pool_size': []})
        
        return checkpoint['update'], checkpoint['global_step'], training_info
    
    def train(self, resume_from=None):
        import json # for saving training info
        
        c = self.config
        
        # type check -> make sure not none
        obs_shape = self.envs.single_observation_space.shape
        action_shape = self.envs.single_action_space.shape
        assert obs_shape is not None, "Observation space has no shape"
        assert action_shape is not None, "Action space has no shape"
        
        # predefine storage buffer -> for each step + env, store a specific piece of data (more efficient than lists)
        obs = torch.zeros((c["num_steps"], c["num_envs"]) + tuple(obs_shape), device=self.device)
        actions = torch.zeros((c["num_steps"], c["num_envs"]) + tuple(action_shape), device=self.device)
        logprobs = torch.zeros((c["num_steps"], c["num_envs"]), device=self.device)
        dones = torch.zeros((c["num_steps"], c["num_envs"]), device=self.device)
        rewards = torch.zeros((c["num_steps"], c["num_envs"]), device=self.device)
        values = torch.zeros((c["num_steps"], c["num_envs"]), device=self.device)
        
        init_obs, _ = self.envs.reset()
        # keep track of observations + if it's done to progress the rollout (remember to convert to gpu)
        next_obs = torch.from_numpy(init_obs).to(self.device)
        next_done = torch.zeros(c["num_envs"], dtype=torch.bool, device=self.device)
        
        NUM_UPDATES = c["total_timesteps"] // c["batch_size"]
        
        if resume_from:
            start_update, global_step, training_info = self.load_checkpoint(resume_from)
            start_update += 1 
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING from update {start_update}/{NUM_UPDATES}")
            print(f"Previous global step: {global_step}")
            print(f"Opponent pool size: {len(self.opponent_pool)}")
            print(f"{'='*60}\n")
        else:
            start_update = 0
            global_step = 0
            training_info = {
                'steps': [],
                'rewards': [],
                'opponent_pool_size': []
            }
        
        for update in range(start_update, NUM_UPDATES):
            # snapshot management
            if update > 0 and update % self.snapshot_freq == 0:
                print(f"\n[Self-Play] Creating snapshot at update {update}")
                snapshot = self.snapshot_agent()
                self.opponent_pool.append(snapshot)
                if len(self.opponent_pool) > self.pool_size: # too many opponents -> remove + cleanup
                    removed = self.opponent_pool.pop(0)
                    del removed
                    print("[Self-Play] Removed oldest opponent (pool at max size)")
            # choose opponent
            self.update_opponent()
            if self.curr_opponent is None:
                print("[Self-Play] Training vs random opponent (pool empty)")
            else:
                print("[Self-Play] Training vs opponent from pool")
            
            # lr annealing
            frac = max(0.0, 1.0 - update / NUM_UPDATES) # clamp just in case because of floats
            new_lr = frac * c["learning_rate"]
            self.optimizer.param_groups[0]["lr"] = new_lr
            
            # log std annealing
            start_log_std = -0.3
            end_log_std = -1.2
            current_log_std = frac * start_log_std + (1 - frac) * end_log_std
            self.agent.log_std.data.fill_(current_log_std)
            
            # 2 phase loop
                # collect experience with current policy -> rollout
                # use experience to update policy + value function (actor + critic) -> compute advantage, update ppo
            obs, actions, logprobs, dones, rewards, values, next_obs, next_done, episode_info = self.collect_rollout(obs, actions, logprobs, dones, rewards, values, next_obs, next_done)
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).flatten()
                
            advantages, returns = self.compute_advantages(rewards, dones, values, next_value, next_done)
            self.ppo_update(advantages, returns, values, logprobs, actions, obs)
            
            # logging 
            global_step += c["batch_size"]
            
            if update > 0 and update % 10 == 0:
                checkpoint = {
                    'update': update,
                    'global_step': global_step,
                    'agent_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'opponent_pool': [
                        opp.state_dict() for opp in self.opponent_pool
                    ],
                    'config': self.config,
                    'training_info': training_info,
                }
                torch.save(checkpoint, f"models/checkpoint_update_{update}.pth")
                print("Saved full checkpoint")
            
            if episode_info:
                mean_reward = np.mean([ep["reward"] for ep in episode_info])
                mean_length = np.mean([ep["length"] for ep in episode_info])
                training_info['steps'].append(global_step)
                training_info['rewards'].append(float(mean_reward))
                training_info['opponent_pool_size'].append(len(self.opponent_pool))
                print(f"Update {update+1}/{NUM_UPDATES} | Step {global_step} | Episodes: {len(episode_info)} | "
                      f"Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.2f} | Pool Size: {len(self.opponent_pool)}")
            else:
                print(f"Update {update+1}/{NUM_UPDATES} | Step {global_step} | No episodes completed this rollout")
                
        try:
            with open("data/training_info_self_play_3.json", 'w') as f:
                json.dump(training_info, f)
            print("\nTraining data saved to data/training_info_self_play.json")
        except Exception as e:
            print(f"Warning: Could not save data: {e}")
            
        self.envs.close()