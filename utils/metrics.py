import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def normalize(vals):
    min_v = np.min(vals)
    max_v = np.max(vals)
    return (vals - min_v) / (max_v - min_v)

def eval_training(data, output_path):
    loaded_data = {}
    for name, filepath in data.items():
        with open(filepath, 'r') as f:
            loaded_data[name] = json.load(f)
    
    min_len = min(len(d["steps"]) for d in loaded_data.values())

    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'pink']

    for (name, d), color in zip(loaded_data.items(), colors):
        steps = d["steps"][:min_len]
        rewards = d["rewards"][:min_len]

        normalized = normalize(rewards)
        plt.plot(steps, normalized, label=name,
                 linewidth=2, color=color, alpha=0.6)

    plt.xlabel("Training Steps")
    plt.ylabel("Normalized Rewards")
    plt.title("Learning Speed Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def eval_single_agent(env, agent, device, max_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    info = {}
    prev_pos = None
    total_distance = 0.0
    
    # run episode
    for step in range(max_steps):
        # get action and step
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # distance travelled
        current_pos = info['position']
        if prev_pos is not None:
            step_distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
            total_distance += step_distance
        prev_pos = current_pos
        
        # if stop
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'progress': info['progress'],
        'finished': info['finished'],
        'crashed': info['crashed'],
        'speed': info['speed'],
        'total_distance': total_distance,
        'distance_per_step': total_distance / (step + 1) if step > 0 else 0,
    }

def eval_multi_agent(env, agent, device, max_steps=3000):
    obs_dict, _ = env.reset()
    total_reward_0 = 0
    total_reward_1 = 0
    step = 0
    info_dict = {}
    total_distance_0 = 0.0
    total_distance_1 = 0.0
    prev_pos_0 = None
    prev_pos_1 = None
    
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
        
        current_pos_0 = info_dict['0']['position']
        current_pos_1 = info_dict['1']['position']
        if prev_pos_0 is not None:
            step_distance_0 = np.sqrt((current_pos_0[0] - prev_pos_0[0])**2 + (current_pos_0[1] - prev_pos_0[1])**2)
            total_distance_0 += step_distance_0
        if prev_pos_1 is not None:
            step_distance_1 = np.sqrt((current_pos_1[0] - prev_pos_1[0])**2 + (current_pos_1[1] - prev_pos_1[1])**2)
            total_distance_1 += step_distance_1
        prev_pos_0 = current_pos_0
        prev_pos_1 = current_pos_1
        
        # if stop
        if done_dict["__all__"]:
            break
        
    if info_dict['0']['finished']:
        chosen_idx = '0'
        chosen_reward = total_reward_0
        chosen_distance = total_distance_0
    elif info_dict['1']['finished']:
        chosen_idx = '1'
        chosen_reward = total_reward_1
        chosen_distance = total_distance_1
    else:
        chosen_idx = '0'
        chosen_reward = total_reward_0
        chosen_distance = total_distance_0
    
    return {
        'total_reward': chosen_reward,
        'progress': info_dict[chosen_idx]['progress'],
        'finished': info_dict[chosen_idx]['finished'],
        'crashed': info_dict[chosen_idx]['crashed'],
        'speed': info_dict[chosen_idx]['speed'],
        'placement': info_dict[chosen_idx].get('placement', None),
        'steps': step + 1,
        'total_distance': chosen_distance,
        'distance_per_step': chosen_distance / (step + 1) if step > 0 else 0
    }
    
def eval_sb3_agent(env, model, max_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    info = {}
    prev_pos = None
    total_distance = 0.0
    
    for step in range(max_steps):
        action, states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        current_pos = info['position']
        if prev_pos is not None:
            step_distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
            total_distance += step_distance
        prev_pos = current_pos
        
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'progress': info['progress'],
        'finished': info['finished'],
        'crashed': info['crashed'],
        'speed': info['speed'],
        'total_distance': total_distance,
        'distance_per_step': total_distance / (step + 1) if step > 0 else 0
    }
    
def display_comparison(results_files, labels, output_path):
    all_results = []
    for file in results_files:
        with open(file, 'r') as f:
            all_results.append(json.load(f))
    
    categories = [
        'Success Rate', 
        'Avg Speed\n(normalized)', 
        'Avg Distance\n(normalized)',
        'Steps / Progress',
    ]
    
    # max values -> to normalize
    max_speed = max(r['avg_speed'] for r in all_results if r['avg_speed'] > 0)
    max_distance = max(r['avg_distance'] for r in all_results if r['avg_distance'] > 0)
    max_steps_per_progress = max(r['avg_steps_per_progress'] for r in all_results)
    
    data = []
    for result in all_results:
        data.append([
            result['success_rate'],
            result['avg_speed'] / max_speed if result['avg_speed'] > 0 else 0,
            result['avg_distance'] / max_distance if result['avg_distance'] > 0 else 0,
            result['avg_steps_per_progress'] / max_steps_per_progress,
        ])
    
    x = np.arange(len(categories))
    width = 0.8 / len(data) 
    
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ['blue', 'green', 'orange', 'pink']
    
    # plot bars for each agent
    for i, (agent_data, label) in enumerate(zip(data, labels)):
        offset = (i - len(data)/2 + 0.5) * width
        ax.bar(x + offset, agent_data, width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Agent Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.4, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison chart saved to {output_path}")