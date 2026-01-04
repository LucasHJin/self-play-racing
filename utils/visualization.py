import numpy as np
import pygame
import torch
import cv2

def convert_coords(x, y, ox, oy, scale, screen_size):
    screen_x = int((x + ox) * scale)
    screen_y = int(screen_size - (y + oy) * scale)
    return screen_x, screen_y


def setup_track_visualization(track, screen_size=800):
    # track bounds for scaling
    all_points = np.vstack([track.left_boundary, track.right_boundary])
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    padding = 10
    track_width = max_x - min_x
    track_height = max_y - min_y
    
    scale = min(
        screen_size / (track_width + 2*padding), 
        screen_size / (track_height + 2*padding)
    )
    offset_x = -min_x + padding
    offset_y = -min_y + padding
    
    left_points = [
        convert_coords(p[0], p[1], offset_x, offset_y, scale, screen_size) 
        for p in track.left_boundary
    ]
    right_points = [
        convert_coords(p[0], p[1], offset_x, offset_y, scale, screen_size) 
        for p in track.right_boundary
    ]
    track_polygon = left_points + right_points[::-1] + [left_points[0]]
    
    start_left = convert_coords(
        track.waypoints[0][0] + track.normals[0][0] * track.track_width,
        track.waypoints[0][1] + track.normals[0][1] * track.track_width,
        offset_x, offset_y, scale, screen_size
    )
    start_right = convert_coords(
        track.waypoints[0][0] - track.normals[0][0] * track.track_width,
        track.waypoints[0][1] - track.normals[0][1] * track.track_width,
        offset_x, offset_y, scale, screen_size
    )
    
    return {
        'scale': scale,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'screen_size': screen_size,
        'left_points': left_points,
        'right_points': right_points,
        'track_polygon': track_polygon,
        'start_left': start_left,
        'start_right': start_right
    }


def visualize_single_agent(env, agent, device, video_path, max_steps=2000):
    # setup
    track = env.track
    
    # pygame setup
    pygame.init()
    screen_size = 800
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()
    
    viz = setup_track_visualization(track, screen_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    video_writer = cv2.VideoWriter(video_path, fourcc, 60, (screen_size, screen_size))
    
    obs, _ = env.reset()
    path_points = []
    total_reward = 0
    running = True
    step = 0
    info = {}
    
    # run episode
    for step in range(max_steps):
        for event in pygame.event.get():  # handle quit
            if event.type == pygame.QUIT:
                running = False
                break
        
        if not running:
            break
        
        # get action and step
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # update visualization
        screen.fill((50, 50, 50))
        pygame.draw.polygon(screen, (180, 180, 180), viz['track_polygon'])
        pygame.draw.lines(screen, (0, 0, 0), True, viz['left_points'], 4)
        pygame.draw.lines(screen, (0, 0, 0), True, viz['right_points'], 4)
        pygame.draw.line(screen, (0, 255, 0), viz['start_left'], viz['start_right'], 5)
        car = env.car
        path_points.append(convert_coords(
            car.x, car.y, viz['offset_x'], viz['offset_y'], viz['scale'], screen_size
        ))
        if len(path_points) > 1:
            pygame.draw.lines(screen, (255, 100, 100), False, path_points, 3)
        corners = car.get_corners()
        car_screen_points = [
            convert_coords(c[0], c[1], viz['offset_x'], viz['offset_y'], viz['scale'], screen_size) 
            for c in corners
        ]
        pygame.draw.polygon(screen, (255, 0, 0), car_screen_points) 
        pygame.draw.polygon(screen, (150, 0, 0), car_screen_points, 2)
        
        font = pygame.font.Font(None, 30)
        info_text = [
            f"Step: {step}",
            f"Progress: {info['progress']:.1%}",
            f"Speed: {info['speed']:.1f}",
            f"Reward: {total_reward:.0f}"
        ]
        text_offset = 10
        for text in info_text:
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, text_offset))
            text_offset += 25
        
        pygame.display.flip()
        
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        
        clock.tick(60)
        
        # if stop
        if terminated or truncated:
            reason = "Finished" if info['finished'] else "Crashed" if info['crashed'] else "Time limit"
            print(f"{reason} | Steps: {step} | Reward: {total_reward:.1f} | Progress: {info['progress']:.1%}")
            pygame.time.wait(3000)
            break
    
    video_writer.release()
    pygame.quit()
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'progress': info['progress'],
        'finished': info['finished'],
        'crashed': info['crashed'],
        'speed': info['speed']
    }


def visualize_multi_agent(env, agent, device, video_path, max_steps=3000):
    # setup
    track = env.track
    
    # pygame setup (redo for each env)
    pygame.init()
    screen_size = 800
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Multi-Agent Racing")
    clock = pygame.time.Clock()
    
    viz = setup_track_visualization(track, screen_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    video_writer = cv2.VideoWriter(video_path, fourcc, 60, (screen_size, screen_size))
    
    obs_dict, _ = env.reset()
    path_points_0 = []
    path_points_1 = []
    total_reward_0 = 0
    total_reward_1 = 0
    running = True
    step = 0
    info_dict = {}
    
    # run episode
    for step in range(max_steps):
        for event in pygame.event.get():  # handle quit
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
        pygame.draw.polygon(screen, (180, 180, 180), viz['track_polygon'])
        pygame.draw.lines(screen, (0, 0, 0), True, viz['left_points'], 4)
        pygame.draw.lines(screen, (0, 0, 0), True, viz['right_points'], 4)
        pygame.draw.line(screen, (0, 255, 0), viz['start_left'], viz['start_right'], 5)
        
        # draw cars
        car_0 = env.cars[0]
        path_points_0.append(convert_coords(
            car_0.x, car_0.y, viz['offset_x'], viz['offset_y'], viz['scale'], screen_size
        ))
        if len(path_points_0) > 1:
            pygame.draw.lines(screen, (255, 100, 100), False, path_points_0, 3)
        corners_0 = car_0.get_corners()
        car_0_screen_points = [
            convert_coords(c[0], c[1], viz['offset_x'], viz['offset_y'], viz['scale'], screen_size) 
            for c in corners_0
        ]
        pygame.draw.polygon(screen, (255, 0, 0), car_0_screen_points)
        pygame.draw.polygon(screen, (150, 0, 0), car_0_screen_points, 2)
        car_1 = env.cars[1]
        path_points_1.append(convert_coords(
            car_1.x, car_1.y, viz['offset_x'], viz['offset_y'], viz['scale'], screen_size
        ))
        if len(path_points_1) > 1:
            pygame.draw.lines(screen, (100, 100, 255), False, path_points_1, 3)
        corners_1 = car_1.get_corners()
        car_1_screen_points = [
            convert_coords(c[0], c[1], viz['offset_x'], viz['offset_y'], viz['scale'], screen_size) 
            for c in corners_1
        ]
        pygame.draw.polygon(screen, (0, 0, 255), car_1_screen_points)
        pygame.draw.polygon(screen, (0, 0, 150), car_1_screen_points, 2)
        
        font = pygame.font.Font(None, 30)
        info_text = [
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
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
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
            
            print(f"\n{reason}")
            print(f"  Car 0 (Red):  Reward: {total_reward_0:.1f} | Progress: {info_dict['0']['progress']:.1%}")
            print(f"  Car 1 (Blue): Reward: {total_reward_1:.1f} | Progress: {info_dict['1']['progress']:.1%}")
            print(f"  Winner: {winner}")
            
            pygame.time.wait(2000)
            break
    
    video_writer.release()
    pygame.quit()
    
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