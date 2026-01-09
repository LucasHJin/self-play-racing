def hyperparams_config():
    config = {
        # training
        "total_timesteps": 3000000,
        "num_envs": 16,
        "num_steps": 2048,
        "learning_rate": 3e-4,
        
        # ppo specific
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "clip_coef": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "update_epochs": 10,
        "num_minibatches": 16,
        "max_grad_norm": 0.5,
        "kl_target": 0.015,
        
        # self play
        "snapshot_freq": 15,
        "pool_size": 5,
        
        # system
        "seed": 1,
        "cuda": True,
        "torch_deterministic": True,
    }
    
    config["batch_size"] = config["num_steps"] * config["num_envs"]
    config["minibatch_size"] = config["batch_size"] // config["num_minibatches"]
    
    return config