

class Car:
    MAX_SPEED = 30.0
    ACCELERATION = 10.0
    STEERING_SPEED = 3.0
    DRAG = 0.95
    
    def __init__(self, track):
        self.track = track
        self.reset()
    
    def reset(self):
        self.x, self.y, self.angle = self.track.get_start_pos()
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.progress = 0.0
        self.checkpoints = 0
        self.crashed = False
        self.finished = False
    
    def update(self):
        pass