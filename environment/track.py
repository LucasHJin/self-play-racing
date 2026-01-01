import numpy as np
from scipy.interpolate import CubicSpline

class Track:
    TRACK_WIDTH = 4.0
    
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
        self.segment_cache = {} # segment cache for vectorized raycasting
        self.build_segment_cache()
        
    def gen_waypoints(self, factor=30):
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
    
    def build_segment_cache(self):
        left_starts = np.array([seg[0] for seg in self.left_segments])
        left_ends = np.array([seg[1] for seg in self.left_segments])
        right_starts = np.array([seg[0] for seg in self.right_segments])
        right_ends = np.array([seg[1] for seg in self.right_segments])
        
        # combine left and right [num_segments, 2]
        all_starts = np.vstack([left_starts, right_starts])
        all_ends = np.vstack([left_ends, right_ends])
        
        self.segment_cache = {
            'starts': all_starts,
            'ends': all_ends,
            'v2': all_ends - all_starts  # precompute segment vectors
        }
        
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
    
    def raycast(self, origin, direction, max_dist=50.0):
        ray_dir = np.array([np.cos(direction), np.sin(direction)])
        all_starts = self.segment_cache['starts']
        v1 = origin - all_starts 
        v2 = self.segment_cache['v2']
        v3 = np.array([-ray_dir[1], ray_dir[0]])
        dotp = np.sum(v2 * v3, axis=1) 
        
        # filter out parallel segments (where dotp around 0)
        valid = np.abs(dotp) > 1e-10
        if not np.any(valid):
            return max_dist
        
        # distance along ray -> (t = cross(v2, v1) / dotp)
        cross_products = v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]
        t = np.full(len(all_starts), max_dist)
        t[valid] = cross_products[valid] / dotp[valid]
        # position along segment -> (s = dot(v1, v3) / dotp)
        dot_products = np.sum(v1 * v3, axis=1)
        s = np.full(len(all_starts), -1.0)
        s[valid] = dot_products[valid] / dotp[valid]
        # valid intersections
        hit_mask = valid & (t >= 0) & (s >= 0) & (s <= 1)
        if not np.any(hit_mask):
            return max_dist
    
        return float(np.min(t[hit_mask])) # return min