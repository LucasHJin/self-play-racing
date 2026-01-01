import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

class Agent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        obs_dim = int(np.array(obs_space.shape).prod())
        action_dim = action_space.shape[0]
        
        # actor outputs mu + std for normal distribution
        self.actor_mu = nn.Sequential(
            Agent.layer_optimization(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(256, 256)),
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(256, action_dim), std=0.01), # output logits for all possible actions
            nn.Tanh(), # bound to (-1, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim)) # use log to always learn a positive value
        
        self.critic = nn.Sequential(
            Agent.layer_optimization(nn.Linear(obs_dim, 256)), # flatten into total input features
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(256, 256)),
            nn.Tanh(),
            Agent.layer_optimization(nn.Linear(256, 1), std=1.0),
        )
        
    # for advantage calcs
    def get_value(self, obs):
        return self.critic(obs)
        
    # for interacting with env + optimizing parameters
    def get_action_and_value(self, obs, action=None):
        # normal distribution instead of logits/prob
        mu = self.actor_mu(obs)
        std = torch.exp(self.log_std).expand_as(mu) # expand to match batch size
        dist = torch.distributions.Normal(mu, std)

        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(obs) # -1 -> add up all log probs into 1 value per obs
    
    @staticmethod
    def layer_optimization(layer, std=np.sqrt(2), bias=0.0):
        torch.nn.init.orthogonal_(layer.weight, std) # stabilize initial weights
        torch.nn.init.constant_(layer.bias, bias) # more predictable training
        return layer


class PPO:
    def __init__(self, env_fn, config, device="cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and config["cuda"] else "cpu")
        
        self.envs = gym.vector.SyncVectorEnv([self._make_env(env_fn, config["seed"] + i) for i in range(config["num_envs"])])
        
        # seed the values for reproducibility
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        
        # obs + agent need to be on same device (pref gpu)
        self.agent = Agent(
            self.envs.single_observation_space,
            self.envs.single_action_space
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["learning_rate"], eps=1e-5)
    
    def _make_env(self, env_fn, seed):
        def thunk():
            env = env_fn()
            env = gym.wrappers.RecordEpisodeStatistics(env)
            #env = gym.wrappers.NormalizeObservation(env)
            #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    def collect_rollout(self, obs, actions, logprobs, dones, rewards, values, next_obs, next_done):
        c = self.config
        reward_tensor = torch.empty(c["num_envs"], dtype=torch.float32, device=self.device) # preallocate reward tensor 
        episode_info = []
        
        # no gradient update -> just collecting data
        with torch.no_grad():
            for step in range(c["num_steps"]):
                obs[step].copy_(next_obs) # for each step, you give it shape (envs, obs) - also use copy instead of =
                dones[step].copy_(next_done) # use done for bootstrapping
                
                action, logprob, _, value = self.agent.get_action_and_value(next_obs) # returns tensors
                actions[step].copy_(action)
                logprobs[step].copy_(logprob)
                values[step].copy_(value.flatten()) # (num_envs, 1) -> (num_envs,)
                    
                # step through and repeat
                next_obs_np, reward_np, terminated, truncated, infos = self.envs.step(action.cpu().numpy()) # gym envs expect cpu
                
                reward_tensor.copy_(torch.from_numpy(reward_np).to(self.device).flatten())
                rewards[step].copy_(reward_tensor)
                
                next_obs.copy_(torch.from_numpy(next_obs_np).to(self.device))
                next_done.copy_(torch.from_numpy(np.logical_or(terminated, truncated)).to(self.device))
                
                # get finished episodes and add them to the info for this rollout
                if "episode" in infos:
                    episode_mask = infos.get("_episode", np.ones(c["num_envs"], dtype=bool))
                    for i in range(c["num_envs"]):
                        if episode_mask[i]:
                            episode_info.append({
                                "reward": infos["episode"]["r"][i],
                                "length": infos["episode"]["l"][i]
                            })
        
        return obs, actions, logprobs, dones, rewards, values, next_obs, next_done, episode_info
    
    def compute_advantages(self, rewards, dones, values, next_value, next_done):
        c = self.config
        with torch.no_grad():
            advantages = torch.zeros_like(rewards, device=self.device)
            running_adv = 0 # is a_t+1 when calculating a_t
            
            for t in reversed(range(c["num_steps"])): # need to reverse because we bootstrap based on future rewards
                if t == c["num_steps"] - 1:
                    next_nonterminal = 1.0 - next_done.to(dtype=torch.float32)
                    next_value_t = next_value
                else:
                    next_nonterminal = 1.0 - dones[t+1].to(dtype=torch.float32)
                    next_value_t = values[t+1]
                
                # TD error -> delta = reward + discount * existing value - current value
                delta = rewards[t] + (c["gamma"] * next_nonterminal * next_value_t) - values[t] # non terminal checks if we should add it in or already stopped
                # GAE -> advantage = change in value + discount * special factor * existing advantage (recursive) 
                advantages[t] = running_adv = delta + c["gamma"] * c["gae_lambda"] * next_nonterminal * running_adv
            returns = advantages + values # returns is target for value function
            
            return advantages, returns
    
    def ppo_update(self, advantages, returns, values, logprobs, actions, obs):
        c = self.config
        # flatten batch data (num_steps * num_envs, [optional data]) -> no time data needed
        b_obs = obs.reshape((-1,) + obs.shape[2:])
        b_actions = actions.reshape((-1,) + actions.shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(c["batch_size"]) # for shuffling + minibatch
        
        for epoch in range(c["update_epochs"]):
            np.random.shuffle(b_inds)
            for start_idx in range(0, c["batch_size"], c["minibatch_size"]):
                end_idx = start_idx + c["minibatch_size"]
                mb_inds = b_inds[start_idx:end_idx]
                
                # pass action back in to get new probability + value using new policy
                _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds]) 
                ratio = (new_logprobs - b_logprobs[mb_inds]).exp()
                
                # kl stopping
                with torch.no_grad():
                    approx_kl = (b_logprobs[mb_inds] - new_logprobs).mean()
                    if approx_kl > c["kl_target"]:
                        print(f"  Early stopping at epoch {epoch+1} due to KL divergence: {approx_kl:.4f}")
                        return
                
                # policy loss
                # note -> signs in formula show you what to max + min (pytorch naturally minimizes)
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) # normalize in minibatches because each mb has different values
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - c["clip_coef"], 1 + c["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # mean averages E[] into single scalar
                
                # value loss + clipping for critic (minimize -> keep positive)
                new_values = new_values.flatten()
                v_clip = b_values[mb_inds] + torch.clamp(new_values - b_values[mb_inds], -c["clip_coef"], c["clip_coef"]) # same clip coef as policy clip
                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_loss_clipped = (v_clip - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                #entropy loss
                e_loss = -entropy.mean()
                
                # total loss
                loss = pg_loss + c["ent_coef"] * e_loss + c["vf_coef"] * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), c["max_grad_norm"]) # clip if magnitude of vectors is too big
                self.optimizer.step()
    
    def train(self):
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
        global_step = 0
        
        training_info = {
            'steps': [],
            'rewards': []
        }
        
        for update in range(NUM_UPDATES):
            # lr annealing
            frac = max(0.0, 1.0 - update / NUM_UPDATES) # clamp just in case because of floats
            new_lr = frac * c["learning_rate"]
            self.optimizer.param_groups[0]["lr"] = new_lr
            
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
            if episode_info:
                mean_reward = np.mean([ep["reward"] for ep in episode_info])
                mean_length = np.mean([ep["length"] for ep in episode_info])
                training_info['steps'].append(global_step)
                training_info['rewards'].append(float(mean_reward))
                print(f"Update {update+1}/{NUM_UPDATES} | Step {global_step} | Episodes: {len(episode_info)} | "
                      f"Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.2f}")
            else:
                print(f"Update {update+1}/{NUM_UPDATES} | Step {global_step} | No episodes completed this rollout")
                
        try:
            with open("/cache/training_info_self_play.json", 'w') as f:
                json.dump(training_info, f)
            print("\nTraining data saved to /cache/training_info_self_play.json")
        except Exception as e:
            print(f"Warning: Could not save data: {e}")
    
    def save(self, path):
        torch.save(self.agent.state_dict(), path)
    
    def load(self, path):
        self.agent.load_state_dict(torch.load(path, map_location=self.device))