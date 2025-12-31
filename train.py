import random
import numpy as np
import torch

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

def train():
    config = hyperparams_config()
    set_seed(config["seed"])
    
    # factory function -> separate vectorized envs
    def env_fn():
        return RacingEnv()
    trainer = PPO(env_fn, config, device="cuda")
    
    print("Starting training")
    
    trainer.train()
    
    trainer.save("models/racing_agent.pth")
    print("\nTraining complete! Model saved to models/racing_agent.pth")

if __name__ == "__main__":
    train()