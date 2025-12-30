import numpy as np

class Car:
    MAX_SPEED = 30.0
    ACCELERATION = 10.0
    STEERING_SPEED = 3.0
    DRAG = 0.95 # friction in forward direction
    LATERAL_FRICTION = 0.85 # friction in sideways direction (drift)
    GRIP = 0.9 # how much power can be exerted laterally (also drift)
    
    def __init__(self, track):
        self.track = track
        self.reset()
    
    def reset(self):
        self.x, self.y, self.angle = self.track.get_start_pos()
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.progress = 0.0
        self.crashed = False
        self.finished = False
    
    def update(self, steering, throttle, dt=0.05):
        """
        steering -> [-1.0, 1.0] for full left/full right
        throttle -> [0.0, 1.0] for power amount
        dt -> timestep
        """
        angular_velocity = steering * Car.STEERING_SPEED
        self.angle = self.angle + (angular_velocity * dt)
        self.angle = self.angle % (2 * np.pi) # keep in [0, 2π]
        
        # compute velocities relative to the car
        v_forward = self.vx * np.cos(self.angle) + self.vy * np.sin(self.angle)
        v_lateral = self.vx * (-np.sin(self.angle)) + self.vy * np.cos(self.angle)
        accel_forward = throttle * Car.ACCELERATION
        v_forward = (v_forward + (accel_forward * dt)) * Car.DRAG
        v_lateral = v_lateral * Car.LATERAL_FRICTION * Car.GRIP
        
        # convert back to global
        self.vx = v_forward * np.cos(self.angle) - v_lateral * np.sin(self.angle)
        self.vy = v_forward * np.sin(self.angle) + v_lateral * np.cos(self.angle)
        
        # clamp speed
        speed = np.sqrt((self.vx ** 2) + (self.vy ** 2))
        if speed > Car.MAX_SPEED:
            scale = Car.MAX_SPEED / speed
            self.vx *= scale
            self.vy *= scale
            
        # final updates
        self.x = self.x + (self.vx * dt)
        self.y = self.y + (self.vy * dt)
        self.progress = self.track.calc_progress(self.x, self.y)
        self.crashed = self.track.check_collision(self.x, self.y)