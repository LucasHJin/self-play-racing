import torch
import numpy as np
import pygame
from environment.multi_racing_env import MultiRacingEnv
from environment.multi_track import gen_tracks
from agent.ppo import Agent

def convert_coords(x, y, ox, oy, scale, screen_size):
    screen_x = int((x + ox) * scale)
    screen_y = int(screen_size - (y + oy) * scale)
    return screen_x, screen_y

def eval(model_path="models/self_play_agent.pth", num_episodes=3, seed=999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # generate track pool
    track_pool = gen_tracks(num_tracks=num_episodes, seed=seed)
    track_widths = [np.random.randint(4, 10) for _ in range(num_episodes)]
    
    # dummy env to get shapes
    dummy_env = MultiRacingEnv(num_agents=2, num_sensors=11)
    single_obs_space = dummy_env.observation_space["0"] # type: ignore
    single_action_space = dummy_env.action_space["0"] # type: ignore
    
    agent = Agent(single_obs_space, single_action_space)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()
    
    print("Testing Self Play:")
    
    for episode in range(num_episodes):
        # setup
        env = MultiRacingEnv(
            num_agents=2, 
            num_sensors=11,
            track_pool=track_pool,
            track_id=episode,
            track_width=track_widths
        )
        track = env.track
        
        # pygame setup (redo for each env)
        pygame.init()
        all_points = np.vstack([track.left_boundary, track.right_boundary]) # track bounds for scaling
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        padding = 10
        track_width = max_x - min_x
        track_height = max_y - min_y
        screen_size = 800
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption(f"Self-Play Racing - Episode {episode + 1}")
        clock = pygame.time.Clock()
        scale = min(screen_size / (track_width + 2*padding), screen_size / (track_height + 2*padding))
        offset_x = -min_x + padding
        offset_y = -min_y + padding
        left_points = [convert_coords(p[0], p[1], offset_x, offset_y, scale, screen_size) for p in track.left_boundary]
        right_points = [convert_coords(p[0], p[1], offset_x, offset_y, scale, screen_size) for p in track.right_boundary]
        track_polygon = left_points + right_points[::-1] + [left_points[0]]
        start_left = convert_coords(
            track.waypoints[0][0] + track.normals[0][0] * track.track_width,
            track.waypoints[0][1] + track.normals[0][1] * track.track_width,
            offset_x,
            offset_y,
            scale,
            screen_size
        )
        start_right = convert_coords(
            track.waypoints[0][0] - track.normals[0][0] * track.track_width,
            track.waypoints[0][1] - track.normals[0][1] * track.track_width,
            offset_x,
            offset_y,
            scale,
            screen_size
        )
        
        obs_dict, _ = env.reset()
        
        path_points_0 = []
        path_points_1 = []
        total_reward_0 = 0
        total_reward_1 = 0
        running = True
        
        # run episodes
        for step in range(3000):
            for event in pygame.event.get(): # handle quit
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
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
            
            # update visualization
            screen.fill((50, 50, 50))
            pygame.draw.polygon(screen, (180, 180, 180), track_polygon)
            pygame.draw.lines(screen, (0, 0, 0), True, left_points, 4)
            pygame.draw.lines(screen, (0, 0, 0), True, right_points, 4)
            pygame.draw.line(screen, (0, 255, 0), start_left, start_right, 5)
            
            # draw cars
            car_0 = env.cars[0]
            path_points_0.append(convert_coords(car_0.x, car_0.y, offset_x, offset_y, scale, screen_size))
            if len(path_points_0) > 1:
                pygame.draw.lines(screen, (255, 100, 100), False, path_points_0, 3)
            corners_0 = car_0.get_corners()
            car_0_screen_points = [convert_coords(c[0], c[1], offset_x, offset_y, scale, screen_size) for c in corners_0]
            pygame.draw.polygon(screen, (255, 0, 0), car_0_screen_points)
            pygame.draw.polygon(screen, (150, 0, 0), car_0_screen_points, 2)
            
            car_1 = env.cars[1]
            path_points_1.append(convert_coords(car_1.x, car_1.y, offset_x, offset_y, scale, screen_size))
            if len(path_points_1) > 1:
                pygame.draw.lines(screen, (100, 100, 255), False, path_points_1, 3) 
            corners_1 = car_1.get_corners()
            car_1_screen_points = [convert_coords(c[0], c[1], offset_x, offset_y, scale, screen_size) for c in corners_1]
            pygame.draw.polygon(screen, (0, 0, 255), car_1_screen_points)
            pygame.draw.polygon(screen, (0, 0, 150), car_1_screen_points, 2)
            
            font = pygame.font.Font(None, 30)
            info_text = [
                f"Episode: {episode + 1}/{num_episodes} - Track {episode}",
                f"Track Width: {track.track_width:.1f}",
                f"Step: {step}",
                f"Car 0 (Red)  - Progress: {info_dict['0']['progress']:.1%}, Speed: {info_dict['0']['speed']:.1f}, Reward: {total_reward_0:.0f}",
                f"Car 1 (Blue) - Progress: {info_dict['1']['progress']:.1%}, Speed: {info_dict['1']['speed']:.1f}, Reward: {total_reward_1:.0f}",
            ]
            
            text_offset = 10
            for text in info_text:
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (10, text_offset))
                text_offset += 25
                
            pygame.display.flip()
            clock.tick(60)
            
            # if stop
            if done_dict["__all__"]:
                if "placement" in info_dict["0"]:
                    placement_0 = info_dict["0"]["placement"]
                    winner = "Car 0 (Red)" if placement_0 == 1 else "Car 1 (Blue)"
                else:
                    winner = "Unknown"
                
                reason = "Finished" if any(car.finished for car in env.cars) else \
                         "All crashed" if all(car.crashed for car in env.cars) else \
                         "Time limit"
                
                print(f"\nEpisode {episode+1}: {reason}")
                print(f"  Car 0 (Red):  Reward: {total_reward_0:.1f} | Progress: {info_dict['0']['progress']:.1%}")
                print(f"  Car 1 (Blue): Reward: {total_reward_1:.1f} | Progress: {info_dict['1']['progress']:.1%}")
                print(f"  Winner: {winner}")
                
                pygame.time.wait(2000)
                break
    
    pygame.quit()

if __name__ == "__main__":
    eval(num_episodes=5, seed=999)