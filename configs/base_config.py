def get_ppo_config():
    return {
        # Training
        "total_timesteps": 500000,
        "num_envs": 4,
        "num_steps": 2048,
        "learning_rate": 3e-4,
        
        # PPO specific
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "update_epochs": 12,
        "num_minibatches": 32,
        "max_grad_norm": 0.5,
        
        # System
        "seed": 1,
        "cuda": True,
        "torch_deterministic": True,
    }

def get_racing_config():
    config = get_ppo_config()
    
    config.update({
        "total_timesteps": 1000000, 
        "num_envs": 8,
        "learning_rate": 1e-4,
    })
    
    return config