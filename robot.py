import time
from math import cos, sin, tan, tanh

import numpy as np

from basics_2d import Circle
from tracking import MultiObjectTracking


class Robot:
    def __init__(self):
        self.lidar = Lidar1d(self)
        self.tracker = MultiObjectTracking(self.lidar.max_dist)
        self.pos = [0, 0]
        self.orien = 0.0  # radian
        self.vel = [0, 0]
        self.radius = 0.4
        self.leader_ped = []
        self.world = []
        self.range_data = []
        self.lidar_segments = []
        self.detections = []
        self.tracks = []

    def follow(self, loc):
        pass

    def step(self, dt):
        t0 = time.time()
        self.range_data = self.lidar.scan(self.world)
        t1 = time.time()
        # print("scan time = {}".format(t1-t0))

        self.lidar_segments = self.tracker.segment(self.range_data, self.pos)
        self.detections = self.tracker.detect(self.lidar_segments, self.pos)
        self.tracks = self.tracker.track(self.detections)
        print(len(self.tracks))

        self.follow(self.leader_ped)


class FollowBot(Robot):
    def __init__(self):
        super(FollowBot, self).__init__()


class Lidar1d:
    def __init__(self, robot):
        self.fov = 270          # degree        => SICK TiM571
        self.resolution = 3     # per degree    => SICK TiM571
        self.max_dist = 25      # in m          => SICK TiM571
        self.data_type = np.float32
        self.robot_ptr = robot

        # pre-compute ray angles
        self.angles = []
        self.rays = []
        for angle in np.arange(-self.fov / 2, self.fov / 2, 1 / self.resolution):
            alpha = angle * np.pi / 180
            ray_end = [np.cos(alpha) * self.max_dist, np.sin(alpha) * self.max_dist]
            self.rays.append([[0, 0], ray_end])
        self.rays = np.stack(self.rays)

    def scan(self, world):
        cur_rays = self.rays.copy()
        cur_rays[:, 0, :] = cur_rays[:, 0, :] + self.robot_ptr.pos

        orien = self.robot_ptr.orien
        rot_matrix = np.array([[cos(orien), -sin(orien)],
                               [sin(orien), cos(orien)]])
        cur_rays[:, 1, :] = np.matmul(rot_matrix, cur_rays[:, 1, :].transpose()).transpose() + self.robot_ptr.pos

        all_intersects = [cur_rays[:, 1]]
        for obj in world.objects:
            results, intersect_pts_ = obj.intersect_many(cur_rays)
            is_not_nan = 1 - np.any(np.isnan(intersect_pts_), axis=1)
            results = np.bitwise_and(results, is_not_nan)
            intersect_pts = np.stack([res * intersect_pts_[ii] + (1-res) * self.max_dist for ii, res in enumerate(results)])
            all_intersects.append(intersect_pts)

        for ped in world.crowds:
            results, intersect_pts_ = Circle(ped.pos, ped.radius).intersect_many(cur_rays)
            is_not_nan = 1 - np.any(np.isnan(intersect_pts_), axis=1)
            results = np.bitwise_and(results, is_not_nan)
            intersect_pts = np.stack([intersect_pts_[ii] * r + 100000 * (1 - r) for ii, r in enumerate(results)])
            all_intersects.append(intersect_pts)

        all_intersects = np.stack(all_intersects)
        dists = all_intersects - cur_rays[0, 0]
        dists = np.linalg.norm(dists, axis=2)
        min_ind = np.argmin(dists, axis=0)
        s = np.stack([all_intersects[ind, ii] for ii, ind in enumerate(min_ind)])

        return s