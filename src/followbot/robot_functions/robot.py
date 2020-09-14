import numpy as np
from numpy.linalg import norm
# from followbot.basics_geometry import Circle
from followbot.robot_functions.tracking import PedestrianDetection, MultiObjectTracking
from followbot.robot_functions.lidar2d import LiDAR2D


class MyRobot:
    def __init__(self):
        self.world = []  # pointer to world
        self.lidar = LiDAR2D(robot_ptr=self)
        self.ped_detector = PedestrianDetection(self.lidar.range_max, np.deg2rad(1/self.lidar.resolution))
        self.tracker = MultiObjectTracking()
        self.pos = [0, 0]
        self.orien = 0.0  # radian
        self.vel = [0, 0]
        self.angular_vel = 0
        self.radius = 0.4
        self.max_speed = 3.0

        self.leader_ped = []

        self.lidar_segments = []
        self.detected_peds = []
        self.tracks = []
        self.pom = []

    def follow(self, ped):
        vec_to_robot = np.array(ped.pos - self.pos)
        dist = norm(vec_to_robot)
        min_dist = 0.40
        speed = min(self.max_speed, norm(ped.vel) * 2)
        if dist < (min_dist + ped.radius + self.radius):
            speed *= max((dist - ped.radius - self.radius) / min_dist, 0)
        self.vel = vec_to_robot / norm(vec_to_robot) * speed

        delta_orien = np.arctan2(vec_to_robot[1], vec_to_robot[0]) - self.orien
        if delta_orien > +np.pi: delta_orien -= 2 * np.pi
        if delta_orien < -np.pi: delta_orien += 2 * np.pi

        self.angular_vel = np.sign(delta_orien) * self.max_speed  # rad/sec
        if abs(delta_orien) < np.pi / 30:
            self.angular_vel *= abs(delta_orien)/(np.pi/30)

    # TODO: should be refactored, and moved to work with ROS
    def step(self, dt):
        self.follow(self.leader_ped)
        self.pos += self.vel * dt
        self.orien += self.angular_vel * dt
        if self.orien >  np.pi: self.orien -= 2 * np.pi
        if self.orien < -np.pi: self.orien += 2 * np.pi

