import numpy as np
from scipy.interpolate import CubicSpline

def gen_random_track(num_points=15, base_radius=50, radius_variation=15, angle_jitter=0.2, smoothness=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # generate angles with some discrepancies in spacing
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    if angle_jitter > 0:
        angle_spacing = 2*np.pi / num_points
        angle_offsets = np.random.uniform(
            -angle_jitter * angle_spacing / 2,
            angle_jitter * angle_spacing / 2,
            num_points
        )
        angles = angles + angle_offsets
        angles = np.sort(angles % (2*np.pi)) # keep in 0-2π and sorted

    # generate random radius for how in/out they are from the circle + smoothing (how close to previous point)
    radii = np.zeros(num_points)
    for i in range(num_points):
        variation = np.random.uniform(-radius_variation, radius_variation)
        if smoothness > 0:
            prev_idx = (i - 1) % num_points
            if i == 0:
                radii[i] = base_radius + variation
            else:
                # smooth with previous point
                neighbor_avg = radii[prev_idx]
                radii[i] = (1 - smoothness) * (base_radius + variation) + (smoothness * neighbor_avg) # weighted average between new radius and previous radius
        else:
            radii[i] = base_radius + variation
    
    # smooth first point at the end
    if smoothness > 0:
        radii[0] = (radii[0] + radii[-1]) / 2
    
    # change to carteiian coords
    control_points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    return control_points

def gen_tracks(num_tracks=10, seed=None):
    tracks = []
    for _ in range(num_tracks):
        num_points = np.random.randint(10, 15)
        base_radius = np.random.randint(50, 80)
        radius_variation = np.random.randint(10, base_radius // 2 - 10)
        angle_jitter = np.random.uniform(0.2, 0.7)
        smoothness = np.random.uniform(0.2, 0.7)
        tracks.append(gen_random_track(num_points, base_radius, radius_variation, angle_jitter, smoothness, seed))
    return tracks

class Track:
    track_width = 4.0
    
    def __init__(self, control_points=None, track_width=None, track_pool=None, track_id=None):
        if track_pool is not None:
            if track_id is None:
                track_id = np.random.randint(0, len(track_pool))
            control_points = track_pool[track_id]
            if track_width is not None and isinstance(track_width, list):
                track_width = track_width[track_id]
                
        if control_points is None:
            self.control_points = np.array([
                [0, 0], [50, 0], [70, 20], [60, 40],
                [70, 50], [50, 70], [20, 70], [10, 50],
                [10, 20], [0, 10],
            ])
        else:
            self.control_points = control_points
        if track_width is not None:
            self.track_width = track_width
        else:
            self.track_width = 6.0
        self.waypoints = self.gen_waypoints() # points to approximate location on track
        self.track_bounds = {
            'min_x': self.waypoints[:, 0].min(),
            'max_x': self.waypoints[:, 0].max(),
            'min_y': self.waypoints[:, 1].min(),
            'max_y': self.waypoints[:, 1].max(),
        }
        self.max_track_distance = np.sqrt(
            (self.track_bounds['max_x'] - self.track_bounds['min_x'])**2 +
            (self.track_bounds['max_y'] - self.track_bounds['min_y'])**2
        )
        self.normals = self.calc_normals()
        self.left_boundary = self.waypoints + self.normals * self.track_width
        self.right_boundary = self.waypoints - self.normals * self.track_width
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
            if dist > self.track_width:
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