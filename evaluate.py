import torch
import numpy as np
import json
from stable_baselines3 import PPO as SB3_PPO
from utils.visualization import visualize_single_agent, visualize_multi_agent, visualize_sb3_agent, visualization_grid
from utils.metrics import eval_single_agent, eval_multi_agent, eval_sb3_agent, display_comparison, eval_training
from environment.track import gen_tracks
from environment.racing_env import RacingEnv
from environment.multi_racing_env import MultiRacingEnv
from agent.ppo import Agent

def evaluate_single_agent_overall(track_pool, track_widths, device, model, video_path, num_tracks=20, num_runs=10):
    # load model
    dummy_env = RacingEnv(num_sensors=11)
    agent = Agent(dummy_env.observation_space, dummy_env.action_space)
    agent.load_state_dict(torch.load(model, map_location=device))
    agent.to(device)
    agent.eval()
    
    # run evaluation
    all_metrics = []
    for track_idx in range(num_tracks):
        print(f"\nTrack {track_idx + 1}/{num_tracks}")
        
        for run_idx in range(num_runs):
            env = RacingEnv(
                num_sensors=11,
                track_pool=track_pool,
                track_id=track_idx,
                track_width=track_widths[run_idx]
            )
            """if run_idx == 0 and track_idx == 0:
                metrics = visualize_single_agent(env, agent, device, video_path)
            else:"""
            metrics = eval_single_agent(env, agent, device)
            all_metrics.append(metrics)
    
    # aggregate stats
    total_episodes = len(all_metrics)
    successful_runs = [m for m in all_metrics if m['finished']]
    
    # efficiency -> calc distance
    efficiency_all = []
    for m in all_metrics:
        if m['progress'] > 0.01:
            efficiency_all.append({
                'steps_per_progress': m['steps'] / m['progress'],
                'distance_per_progress': m['total_distance'] / m['progress'],
            })
    
    results = {
        'num_episodes': total_episodes,
        'num_successful': len(successful_runs),
        'success_rate': len(successful_runs) / total_episodes,
        'crash_rate': sum(m['crashed'] for m in all_metrics) / total_episodes,
        'avg_steps': np.mean([m['steps'] for m in successful_runs]) if successful_runs else 0,
        'avg_reward': np.mean([m['total_reward'] for m in successful_runs]) if successful_runs else 0,
        'avg_progress': np.mean([m['progress'] for m in successful_runs]) if successful_runs else 0,
        'avg_speed': np.mean([m['speed'] for m in successful_runs]) if successful_runs else 0,
        'avg_distance': np.mean([m['total_distance'] for m in successful_runs]) if successful_runs else 0,
        'avg_steps_per_progress': np.mean([e['steps_per_progress'] for e in efficiency_all]),
        
        'all_episodes': all_metrics
    }
    
    return results

def evaluate_multi_agent_overall(track_pool, track_widths, device, model, video_path, num_tracks=20, num_runs=10):
    # load model
    dummy_env = MultiRacingEnv(num_agents=2, num_sensors=11)
    agent = Agent(dummy_env.observation_space["0"], dummy_env.action_space["0"]) # type: ignore
    agent.load_state_dict(torch.load(model, map_location=device))
    agent.to(device)
    agent.eval()
    
    # run evaluation
    all_metrics = []
    for track_idx in range(num_tracks):
        print(f"\nTrack {track_idx + 1}/{num_tracks}")
        
        for run_idx in range(num_runs):
            env = MultiRacingEnv(
                num_agents=2,
                num_sensors=11,
                track_pool=track_pool,
                track_id=track_idx,
                track_width=track_widths[run_idx]
            )
            """if run_idx == 0 and track_idx == 0:
                metrics = visualize_multi_agent(env, agent, device, video_path)
            else:"""
            metrics = eval_multi_agent(env, agent, device)
            all_metrics.append(metrics)
    
    # aggregate stats
    total_episodes = len(all_metrics)
    successful_runs = [m for m in all_metrics if m['finished']]
    
    efficiency_all = []
    for m in all_metrics:
        if m['progress'] > 0.01:
            efficiency_all.append({
                'steps_per_progress': m['steps'] / m['progress'],
                'distance_per_progress': m['total_distance'] / m['progress'],
            })
    
    results = {
        'num_episodes': total_episodes,
        'num_successful': len(successful_runs),
        'success_rate': len(successful_runs) / total_episodes,
        'crash_rate': sum(m['crashed'] for m in all_metrics) / total_episodes,
        'avg_steps': np.mean([m['steps'] for m in successful_runs]) if successful_runs else 0,
        'avg_reward': np.mean([m['total_reward'] for m in successful_runs]) if successful_runs else 0,
        'avg_progress': np.mean([m['progress'] for m in successful_runs]) if successful_runs else 0,
        'avg_speed': np.mean([m['speed'] for m in successful_runs]) if successful_runs else 0,
        'avg_distance': np.mean([m['total_distance'] for m in successful_runs]) if successful_runs else 0,
        'avg_steps_per_progress': np.mean([e['steps_per_progress'] for e in efficiency_all]),
        
        'all_episodes': all_metrics
    }

    return results

def evaluate_sb3_agent_overall(track_pool, track_widths, model, video_path, num_tracks=20, num_runs=10):
    agent = SB3_PPO.load(model)
    
    all_metrics = []
    for track_idx in range(num_tracks):
        print(f"\nTrack {track_idx + 1}/{num_tracks}")
        
        for run_idx in range(num_runs):
            env = RacingEnv(
                num_sensors=11,
                track_pool=track_pool,
                track_id=track_idx,
                track_width=track_widths[run_idx]
            )
            """if run_idx == 0 and track_idx == 0:
                metrics = visualize_sb3_agent(env, agent, video_path)
            else:"""
            metrics = eval_sb3_agent(env, agent)
            all_metrics.append(metrics)
            
    # aggregate stats
    total_episodes = len(all_metrics)
    successful_runs = [m for m in all_metrics if m['finished']]
    
    efficiency_all = []
    for m in all_metrics:
        if m['progress'] > 0.01:
            efficiency_all.append({
                'steps_per_progress': m['steps'] / m['progress'],
                'distance_per_progress': m['total_distance'] / m['progress'],
            })
    
    results = {
        'num_episodes': total_episodes,
        'num_successful': len(successful_runs),
        'success_rate': len(successful_runs) / total_episodes,
        'crash_rate': sum(m['crashed'] for m in all_metrics) / total_episodes,
        'avg_steps': np.mean([m['steps'] for m in successful_runs]) if successful_runs else 0,
        'avg_reward': np.mean([m['total_reward'] for m in successful_runs]) if successful_runs else 0,
        'avg_progress': np.mean([m['progress'] for m in successful_runs]) if successful_runs else 0,
        'avg_speed': np.mean([m['speed'] for m in successful_runs]) if successful_runs else 0,
        'avg_distance': np.mean([m['total_distance'] for m in successful_runs]) if successful_runs else 0,
        'avg_steps_per_progress': np.mean([e['steps_per_progress'] for e in efficiency_all]),
        
        'all_episodes': all_metrics
    }

    return results
    
def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configs
    num_tracks = 40
    num_runs = 5
    seed = 42
    # generate tracks
    track_pool = gen_tracks(num_tracks=num_tracks, seed=seed)
    track_widths = [np.random.RandomState(seed + i).randint(4, 10) 
                    for i in range(num_tracks)]
    
    single_results = evaluate_single_agent_overall(track_pool, track_widths, device, "models/single_agent.pth", "videos/single.mp4", num_tracks, num_runs)
    multi_results = evaluate_multi_agent_overall(track_pool, track_widths, device, "models/self_play_agent.pth", "videos/self_play.mp4", num_tracks, num_runs)
    sb3_results = evaluate_sb3_agent_overall(track_pool, track_widths, "models/sb3_baseline_agent", "videos/sb3.mp4", num_tracks, num_runs)
    sb3_general_results = evaluate_sb3_agent_overall(track_pool, track_widths, "models/sb3_baseline_agent_general", "videos/sb3_general.mp4", num_tracks, num_runs)

    with open("data/eval_info_single.json", "w") as f:
        json.dump(single_results, f, indent=2)
    with open("data/eval_info_sb3.json", "w") as f:
        json.dump(sb3_results, f, indent=2)
    with open("data/eval_info_sb3_general.json", "w") as f:
        json.dump(sb3_general_results, f, indent=2)
    with open("data/eval_info_self_play.json", "w") as f:
        json.dump(multi_results, f, indent=2)
        
    """visualization_grid(
        video_paths=[
            "videos/single.mp4",
            "videos/sb3.mp4",
            "videos/sb3_general.mp4",
            "videos/self_play.mp4",
        ],
        model_names=[
            "Single",
            "SB3 Finetuned",
            "SB3 General",
            "Self-Play",
        ],
        output_path="static/racing_grid.mp4"
    )
    
    eval_training(
        data={
            "Single": "data/training_info_single.json",
            "SB3 Finetuned": "data/training_info_sb3.json",
            "SB3 General": "data/training_info_sb3_general.json",
            "Self-Play": "data/training_info_self_play.json",
        },
        output_path="static/training_eval.png"
    )"""
    
    display_comparison(
        results_files=[
            "data/eval_info_single.json",
            "data/eval_info_sb3.json",
            "data/eval_info_sb3_general.json",
            "data/eval_info_self_play.json",
        ],
        labels=[
            "Single",
            "SB3 Finetuned",
            "SB3 General",
            "Self-Play",
        ],
        output_path="static/eval_comparison.png"
    )
    

if __name__ == "__main__":
    eval()