import time
import numpy as np
from math import cos, sin, tan, tanh, acos

# from followbot.basics_geometry import Circle
from followbot.tracking import MultiObjectTracking
from followbot.lidar2d import LiDAR2D


class MyRobot:
    def __init__(self):
        self.world = []  # pointer to world
        self.lidar = LiDAR2D(robot_ptr=self)
        self.tracker = MultiObjectTracking(self.lidar.range_max)
        self.pos = [0, 0]
        self.orien = 0.0  # radian
        self.vel = [0, 0]
        self.angular_vel = 0
        self.radius = 0.4

        self.leader_ped = []

        self.lidar_segments = []
        self.detected_peds = []
        self.tracks = []
        self.pom = []

    def follow(self, ped):
        vec_to_robot = np.array(ped.pos - self.pos)
        dist = np.linalg.norm(vec_to_robot)
        if dist < (0.5 + ped.radius + self.radius):
            self.vel = np.zeros(2)
        else:
            self.vel = vec_to_robot * (np.linalg.norm(ped.vel) / dist)

        delta_orien = np.arctan2(vec_to_robot[1], vec_to_robot[0]) - self.orien
        # print(delta_orien / np.pi)
        if delta_orien >  np.pi: delta_orien -= 2 * np.pi
        if delta_orien < -np.pi: delta_orien += 2 * np.pi

        if abs(delta_orien) > np.pi / 15:
            self.angular_vel = np.sign(delta_orien) * 1.2  # rad/sec
        else:
            self.angular_vel *= 0.5

    # TODO: should be refactored, and moved to work with ROS
    def step(self, dt):
        # t0 = time.time()
        update_pom = False
        self.lidar.scan(self.world, update_pom, walkable_area=self.world.walkable)
        # self.detected_points, self.occupancy_gridmap = []
        # self.lidar.last_range_pnts , self.lidar.last_range_data, self.lidar.last_occupancy_gridmap

        # t1 = time.time()
        # print("scan time = {}".format(t1-t0))

        self.lidar_segments = self.tracker.segment_points(self.lidar.last_range_pnts, self.pos)
        # self.lidar_segments = self.tracker.segment_range(self.lidar.last_range_data, self.pos)
        self.detected_peds, walls = self.tracker.detect(self.lidar_segments, self.pos)
        self.tracks = self.tracker.track(self.detected_peds)

        if update_pom:
            self.pom = self.lidar.last_occupancy_gridmap.copy()
            for track in self.tracks:
                if track.coasted: continue
                px, py = track.position()
                u, v = self.world.mapping_to_grid(px, py)
                if u < self.pom.shape[0] and v < self.pom.shape[1]:
                    self.pom[u - 2:u + 2, v - 2:v + 2] = 1

        self.follow(self.leader_ped)
        self.pos += self.vel * dt
        self.orien += self.angular_vel * dt
        if self.orien >  np.pi: self.orien -= 2 * np.pi
        if self.orien < -np.pi: self.orien += 2 * np.pi

