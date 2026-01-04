import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import json

def normalize(rewards):
    min_r = np.min(rewards)
    max_r = np.max(rewards)
    return (rewards - min_r) / (max_r - min_r)

def eval_training():
    ROOT = Path(__file__).resolve().parent.parent
    data = {
        "Baseline": json.load(open(ROOT / "data" / "single_agent.json")),
        "+ Speed": json.load(open(ROOT / "data" / "single_agent_speed.json")),
        "+ Time": json.load(open(ROOT / "data" / "single_agent_time.json")),
    }
    
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange']
    for (name, d), color in zip(data.items(), colors):
        normalized = normalize(d["rewards"])
        plt.plot(d["steps"], normalized, label=name, linewidth=2, color=color, alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Normalized Rewards")
    plt.title("Learning Speed Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_speed.png", dpi=300)
    plt.show()

def eval_single_agent(env, agent, device, max_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    info = {}
    
    # run episode
    for step in range(max_steps):
        # get action and step
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # if stop
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'progress': info['progress'],
        'finished': info['finished'],
        'crashed': info['crashed'],
        'speed': info['speed']
    }

def eval_multi_agent(env, agent, device, max_steps=3000):
    obs_dict, _ = env.reset()
    total_reward_0 = 0
    total_reward_1 = 0
    step = 0
    info_dict = {}
    
    # run episode
    for step in range(max_steps):
        # get action and step
        with torch.no_grad():
            obs_0_tensor = torch.FloatTensor(obs_dict["0"]).unsqueeze(0).to(device)
            action_0, _, _, _ = agent.get_action_and_value(obs_0_tensor)
            action_0 = action_0.cpu().numpy()[0]
            
            obs_1_tensor = torch.FloatTensor(obs_dict["1"]).unsqueeze(0).to(device)
            action_1, _, _, _ = agent.get_action_and_value(obs_1_tensor)
            action_1 = action_1.cpu().numpy()[0]
        
        actions = {
            "0": action_0,
            "1": action_1
        }
        
        obs_dict, reward_dict, done_dict, truncated, info_dict = env.step(actions)
        total_reward_0 += reward_dict["0"]
        total_reward_1 += reward_dict["1"]
        
        # if stop
        if done_dict["__all__"]:
            break
        
    if info_dict['0']['finished']:
        chosen_idx = '0'
        chosen_reward = total_reward_0
    elif info_dict['1']['finished']:
        chosen_idx = '1'
        chosen_reward = total_reward_1
    else:
        chosen_idx = '0'
        chosen_reward = total_reward_0
    
    
    return {
        'total_reward': chosen_reward,
        'progress': info_dict[chosen_idx]['progress'],
        'finished': info_dict[chosen_idx]['finished'],
        'crashed': info_dict[chosen_idx]['crashed'],
        'speed': info_dict[chosen_idx]['speed'],
        'placement': info_dict[chosen_idx].get('placement', None),
        'steps': step + 1
    }
    
