import random
import numpy as np
import torch
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from environment.racing_env import RacingEnv
from environment.multi_racing_env import MultiRacingEnv
from environment.track import gen_tracks
from agent.ppo import PPO
from agent.self_play_ppo import SelfPlayPPO
from configs.base_config import hyperparams_config as base_config
from configs.self_play_config import hyperparams_config
from utils.sb3_logger import TrainingLoggerCallback

def train_multi():
    # set seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    config = hyperparams_config()
    set_seed(config["seed"])
    
    print("Generating track pool")
    TRACK_POOL = gen_tracks(num_tracks=config["num_envs"], seed=config["seed"])
    TRACK_WIDTHS = [np.random.randint(6, 10) for _ in range(config["num_envs"])]
    TRACK_ASSIGNMENTS = [i for i in range(config["num_envs"])]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'='*60}")
    print("SELF PLAY PPO TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Num environments: {config['num_envs']}")
    print(f"Batch size: {config['batch_size']:,}")
    print(f"Snapshot frequency: {config['snapshot_freq']}")
    print(f"Pool size: {config['pool_size']}")
    print(f"Expected updates: {config['total_timesteps'] // config['batch_size']}")
    print(f"{'='*60}\n")
    
    # factory function -> separate vectorized envs
    def env_fn(env_idx):
        track_id = TRACK_ASSIGNMENTS[env_idx]
        return MultiRacingEnv(num_agents=2, num_sensors=11, track_pool=TRACK_POOL, track_id=track_id, track_width=TRACK_WIDTHS)
    trainer = SelfPlayPPO(env_fn, config, device=device)
    
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}\n")
    trainer.train()
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    # save final model
    final_path = "models/self_play_agent_3.pth"
    trainer.save(final_path)
    print(f"Final model saved to {final_path}")
    
def train_single():
    # set seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    config = base_config()
    set_seed(config["seed"])
    
    print("Generating track pool")
    TRACK_POOL = gen_tracks(num_tracks=config["num_envs"], seed=config["seed"])
    TRACK_WIDTHS = [np.random.randint(6, 10) for _ in range(config["num_envs"])]
    TRACK_ASSIGNMENTS = [i for i in range(config["num_envs"])]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'='*60}")
    print("PPO TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Num environments: {config['num_envs']}")
    print(f"Batch size: {config['batch_size']:,}")
    print(f"Expected updates: {config['total_timesteps'] // config['batch_size']}")
    print(f"{'='*60}\n")
    
    # factory function -> separate vectorized envs
    def env_fn(env_idx):
        track_id = TRACK_ASSIGNMENTS[env_idx]
        return RacingEnv(
            num_sensors=11,
            track_pool=TRACK_POOL,
            track_id=track_id,
            track_width=TRACK_WIDTHS[env_idx]
        )
    trainer = PPO(env_fn, config, device=device)
    
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}\n")
    trainer.train()
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    # save final model
    final_path = "models/single_agent_4.pth"
    trainer.save(final_path)
    print(f"Final model saved to {final_path}")
    
def train_single_baseline():
    # set seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    config = base_config()
    set_seed(config["seed"])
    
    print("Generating track pool")
    TRACK_POOL = gen_tracks(num_tracks=config["num_envs"], seed=config["seed"])
    TRACK_WIDTHS = [np.random.randint(6, 10) for _ in range(config["num_envs"])]
    
    print(f"{'='*60}")
    print("SB3 PPO BASELINE TRAINING")
    print(f"{'='*60}")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Num environments: {config['num_envs']}")
    print(f"Batch size: {config['batch_size']:,}")
    print(f"Minibatch size: {config['minibatch_size']:,}")
    print(f"Expected updates: {config['total_timesteps'] // config['batch_size']}")
    print(f"{'='*60}\n")
    
    def make_env(env_idx):
        def env_fn():
            env = RacingEnv(
                num_sensors=11,
                track_pool=TRACK_POOL,
                track_id=env_idx,
                track_width=TRACK_WIDTHS[env_idx]
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return env_fn
    
    env = SubprocVecEnv([make_env(i) for i in range(config["num_envs"])])
    
    # match configs to hyperparameters for custom env
    model = SB3_PPO("MlpPolicy", env, seed=config["seed"])
    """model = SB3_PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["num_steps"],
        batch_size=config["minibatch_size"], 
        n_epochs=config["update_epochs"],
        gamma=config["gamma"], 
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_coef"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        seed=config["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )"""
    
    logger_callback = TrainingLoggerCallback(save_path="data/training_info_sb3_general.json")
    
    print("Starting training\n")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=logger_callback,
        progress_bar=False
    )
    print("\nTraining Complete!")

    model.save("models/sb3_baseline_agent_general")
    env.close()
    
if __name__ == "__main__":
    train_multi()
    #train_single_baseline()
    #train_single()