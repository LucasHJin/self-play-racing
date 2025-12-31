import numpy as np
from scipy.interpolate import CubicSpline

class Track:
    TRACK_WIDTH = 5.0
    
    def __init__(self):
        self.control_points = np.array([
            [0, 0],
            [50, 0],
            [70, 20],
            [60, 40],
            [70, 50],
            [50, 70],
            [20, 70],
            [10, 50],
            [10, 20],
            [0, 10],
        ])
        self.waypoints = self.gen_waypoints() # points to approximate location on track
        self.normals = self.calc_normals()
        self.left_boundary = self.waypoints + self.normals * Track.TRACK_WIDTH
        self.right_boundary = self.waypoints - self.normals * Track.TRACK_WIDTH
        self.left_segments = self.gen_segments(self.left_boundary)
        self.right_segments = self.gen_segments(self.right_boundary)
        
    def gen_waypoints(self, factor=40):
        # close points loop
        points = np.vstack((self.control_points, self.control_points[0]))
        
        # parametric value for creating curves for points (euclidean distance) -> add on 0 for starting point
        t = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))))
        cs_x = CubicSpline(t, points[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, points[:, 1], bc_type='periodic')
        
        # generate waypoints
        num_waypoints = len(self.control_points) * factor
        t_waypoints = np.linspace(0, t[-1], num_waypoints, endpoint=False)
        t_x = cs_x(t_waypoints)
        t_y = cs_y(t_waypoints)
        waypoints = np.column_stack((t_x, t_y))
        return waypoints
    
    def calc_normals(self):
        tangents = np.diff(self.waypoints, axis=0, append=[self.waypoints[0]]) # vectors pointing between waypoints
        tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_lengths = np.where(tangent_lengths==0, 1, tangent_lengths) # get rid of div by 0
        tangents = tangents / tangent_lengths
        
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) # reverse to get normals
        return normals
    
    def gen_segments(self, boundary):
        segments = []
        for i in range(len(boundary)):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % len(boundary)] # close off loop
            segments.append((p1, p2))
        return segments
        
    def closest_waypoint_idx(self, x, y):
        idx = np.sum((self.waypoints - np.array((x, y))) ** 2, axis=1).argmin()
        return int(idx)
    
    def get_start_pos(self):
        dx = self.waypoints[1, 0]-self.waypoints[0, 0]
        dy = self.waypoints[1, 1]-self.waypoints[0, 1]
        return (self.waypoints[0, 0], self.waypoints[0, 1], np.arctan2(dy, dx)) # x, y, angle
    
    def calc_progress(self, x, y):
        curr_idx = self.closest_waypoint_idx(x, y)
        return curr_idx / len(self.waypoints)
    
    def check_collision(self, corners):
        for corner in corners:
            idx = self.closest_waypoint_idx(corner[0], corner[1])
            normal = self.normals[idx] # normal is normalized already 
            pos_vector = corner - self.waypoints[idx]
            dist = abs(np.dot(pos_vector, normal)) # project position onto normal vector
            if dist > Track.TRACK_WIDTH:
                return True
        return False
    
    def ray_segment_intersection(self, ray_origin, ray_dir, seg_start, seg_end):
        v1 = ray_origin - seg_start
        v2 = seg_end - seg_start
        v3 = np.array([-ray_dir[1], ray_dir[0]]) 
        
        denom = np.dot(v2, v3)
        if abs(denom) < 1e-10: 
            return None
        
        t = np.cross(v2, v1) / denom
        s = np.dot(v1, v3) / denom
        if t >= 0 and 0 <= s <= 1: 
            return t
        return None
    
    def raycast(self, origin, direction, max_dist=50.0):
        ray_dir = np.array([np.cos(direction), np.sin(direction)])
        
        min_dist = max_dist
        
        for seg_start, seg_end in self.left_segments:
            dist = self.ray_segment_intersection(origin, ray_dir, seg_start, seg_end)
            if dist is not None:
                min_dist = min(dist, min_dist)
        for seg_start, seg_end in self.right_segments:
            dist = self.ray_segment_intersection(origin, ray_dir, seg_start, seg_end)
            if dist is not None:
                min_dist = min(dist, min_dist)
        
        return min_dist