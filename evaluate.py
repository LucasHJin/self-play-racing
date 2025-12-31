import torch
import numpy as np
import matplotlib.pyplot as plt
from environment.racing_env import RacingEnv
from agent.ppo import Agent

# change to pygame later on

def eval(model_path="models/racing_agent.pth", num_episodes=3):
    # setup
    env = RacingEnv()
    track = env.track
    agent = Agent(env.observation_space, env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load trained weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()
    
    print("Testing:")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        # visualization setup
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        waypoints_closed = np.vstack([track.waypoints, track.waypoints[0]])
        left_closed = np.vstack([track.left_boundary, track.left_boundary[0]])
        right_closed = np.vstack([track.right_boundary, track.right_boundary[0]])
        ax.plot(waypoints_closed[:, 0], waypoints_closed[:, 1], 'b-', linewidth=2)
        ax.plot(left_closed[:, 0], left_closed[:, 1], 'k-')
        ax.plot(right_closed[:, 0], right_closed[:, 1], 'k-')
        car_dot, = ax.plot([], [], 'ro', markersize=10)
        car_arrow = ax.arrow(0, 0, 0, 0, head_width=1, head_length=1, fc='red', ec='red')
        trail_line, = ax.plot([], [], 'r-', linewidth=1)
        ax.set_aspect('equal')
        
        path_x = []
        path_y = []
        total_reward = 0

        # run episodes
        for step in range(2000):
            # get action and step
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # update visualization
            car = env.car
            path_x.append(car.x)
            path_y.append(car.y)
            car_dot.set_data([car.x], [car.y])
            trail_line.set_data(path_x, path_y)
            car_arrow.remove()
            arrow_length = 3
            car_arrow = ax.arrow(
                car.x, car.y,
                arrow_length * np.cos(car.angle),
                arrow_length * np.sin(car.angle),
                head_width=1, head_length=1, fc='red', ec='red'
            )
            ax.set_title(f'Episode {episode+1} | Step {step} | Progress: {info["progress"]:.1%} | '
                        f'Speed: {info["speed"]:.1f} | Reward: {total_reward:.0f}')
            
            plt.pause(0.01) # use for animation
            
            # if stop
            if terminated or truncated:
                reason = "Finished" if info['finished'] else "Crashed" if info['crashed'] else "Time limit"
                print(f"Episode {episode+1}: {reason} | Steps: {step} | "
                      f"Reward: {total_reward:.1f} | Progress: {info['progress']:.1%}")
                plt.pause(3) 
                break
        
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    eval(num_episodes=3)