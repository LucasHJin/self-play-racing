import torch
import numpy as np
import json
from pathlib import Path
from utils.visualization import visualize_single_agent, visualize_multi_agent
from utils.metrics import eval_single_agent, eval_multi_agent
from environment.track import gen_tracks
from environment.racing_env import RacingEnv
from environment.multi_racing_env import MultiRacingEnv
from agent.ppo import Agent

def evaluate_single_agent(track_pool, track_widths, device, model, num_tracks=20, num_runs=10):
    # load model
    dummy_env = RacingEnv()
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
                track_pool=track_pool,
                track_id=track_idx,
                track_width=track_widths
            )
            #if run_idx == 0 and track_idx == 0:
            #    metrics = visualize_single_agent(env, agent, device, "videos/single_agent_eval.mp4")
            #else:
            metrics = eval_single_agent(env, agent, device)
            all_metrics.append(metrics)
    
    # aggregate stats
    total_episodes = len(all_metrics)
    results = {
        'num_episodes': total_episodes,
        'success_rate': sum(m['finished'] for m in all_metrics) / total_episodes,
        'crash_rate': sum(m['crashed'] for m in all_metrics) / total_episodes,
        'avg_steps': np.mean([m['steps'] for m in all_metrics]),
        'avg_reward': np.mean([m['total_reward'] for m in all_metrics]),
        'avg_progress': np.mean([m['progress'] for m in all_metrics]),
        'avg_speed': np.mean([m['speed'] for m in all_metrics]),
        'all_episodes': all_metrics
    }
    
    # print summary
    print("\n" + "="*60)
    print("SINGLE-AGENT RESULTS")
    print("="*60)
    print(f"Success Rate:    {results['success_rate']:6.1%}")
    print(f"Crash Rate:      {results['crash_rate']:6.1%}")
    print(f"Avg Steps:       {results['avg_steps']:6.1f}")
    print(f"Avg Reward:      {results['avg_reward']:6.1f}")
    print(f"Avg Progress:    {results['avg_progress']:6.1%}")
    print(f"Avg Speed:       {results['avg_speed']:6.1f}")
    print("="*60)
    
    return results


def evaluate_multi_agent(track_pool, track_widths, device, model, num_tracks=20, num_runs=10):
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
                track_width=track_widths
            )
            if run_idx == 0 and track_idx == 0:
                metrics = visualize_multi_agent(env, agent, device, "videos/self_play_agent_eval.mp4")
            else:
                metrics = eval_multi_agent(env, agent, device)
            all_metrics.append(metrics)
    
    # aggregate stats
    total_episodes = len(all_metrics)
    results = {
        'num_episodes': total_episodes,
        'success_rate': sum(m['finished'] for m in all_metrics) / total_episodes,
        'crash_rate': sum(m['crashed'] for m in all_metrics) / total_episodes,
        'avg_steps': np.mean([m['steps'] for m in all_metrics]),
        'avg_reward': np.mean([m['total_reward'] for m in all_metrics]),
        'avg_progress': np.mean([m['progress'] for m in all_metrics]),
        'avg_speed': np.mean([m['speed'] for m in all_metrics]),
        'all_episodes': all_metrics
    }
    
    # print summary
    print("\n" + "="*60)
    print("MULTI-AGENT RESULTS")
    print("="*60)
    print(f"Success Rate:    {results['success_rate']:6.1%}")
    print(f"Crash Rate:      {results['crash_rate']:6.1%}")
    print(f"Avg Steps:       {results['avg_steps']:6.1f}")
    print(f"Avg Reward:      {results['avg_reward']:6.1f}")
    print(f"Avg Progress:    {results['avg_progress']:6.1%}")
    print(f"Avg Speed:       {results['avg_speed']:6.1f}")
    print("="*60)
    
    return results


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configs
    num_tracks = 1
    num_runs = 3
    seed = 42
    # generate tracks
    track_pool = gen_tracks(num_tracks=num_tracks, seed=seed)
    track_widths = [np.random.RandomState(seed + i).randint(4, 10) 
                    for i in range(num_tracks)]
    
    single_results = evaluate_single_agent(track_pool, track_widths, device, "models/single_agent.pth", num_tracks, num_runs)
    #multi_results = evaluate_multi_agent(track_pool, track_widths, device, "models/self_play_agent.pth", num_tracks, num_runs)

    Path("results").mkdir(exist_ok=True)
    with open("data/single_agent_results.json", "w") as f:
        json.dump(single_results, f, indent=2)
    #with open("data/multi_agent_results.json", "w") as f:
    #    json.dump(multi_results, f, indent=2)
    
    print("\nResults saved to data/ directory")

if __name__ == "__main__":
    eval()