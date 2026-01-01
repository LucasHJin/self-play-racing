import modal

app = modal.App("racing-ppo")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.9.1",
        "numpy==2.4.0",
        "gymnasium==1.2.3",
        "scipy==1.16.3",
    )
    .add_local_dir("environment", remote_path="/root/environment")
    .add_local_dir("agent", remote_path="/root/agent")
    .add_local_dir("configs", remote_path="/root/configs")
)

@app.function(
    image=image,
    gpu="L4",
    timeout=36000,
    volumes={"/cache": modal.Volume.from_name("racing-model-cache", create_if_missing=True)},
)
def train():
    import random
    import numpy as np
    import torch
    import sys
    
    sys.path.insert(0, "/root") # for importing local code
    
    from environment.racing_env import RacingEnv
    from agent.ppo import PPO
    from configs.base_config import hyperparams_config
    
    # set seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    config = hyperparams_config()
    set_seed(config["seed"])
    
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
    def env_fn():
        return RacingEnv()
    trainer = PPO(env_fn, config, device=device)
    
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}\n")
    trainer.train()
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    # save final model
    final_path = "/cache/single_agent_speed.pth"
    trainer.save(final_path)
    print(f"Final model saved to {final_path}")
    
    return {
        "model_path": final_path,
    }


@app.local_entrypoint()
def main():
    train.remote()