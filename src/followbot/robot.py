import time
import numpy as np
from math import cos, sin, tan, tanh, acos

from followbot.basics_2d import Circle
from followbot.tracking import MultiObjectTracking


class Robot:
    def __init__(self):
        self.world = []  # pointer to world
        self.lidar = Lidar1d(self)
        self.tracker = MultiObjectTracking(self.lidar.max_dist)
        self.pos = [0, 0]
        self.orien = 0.0  # radian
        self.vel = [0, 0]
        self.angular_vel = 0
        self.radius = 0.4

        self.leader_ped = []
        self.range_data = []
        self.occupancy_grid = np.empty(1)
        self.lidar_segments = []
        self.detections = []
        self.tracks = []

    def follow(self, ped):
        vec_to_robot = np.array(ped.pos - self.pos)
        dist = np.linalg.norm(vec_to_robot)
        if dist < (0.5 + ped.radius + self.radius):
            self.vel = np.zeros(2)
        else:
            self.vel = vec_to_robot * (np.linalg.norm(ped.vel) / dist)

        delta_orien = np.arctan2(vec_to_robot[1], vec_to_robot[0]) - self.orien
        print(delta_orien / np.pi)
        if delta_orien >  np.pi: delta_orien -= 2 * np.pi
        if delta_orien < -np.pi: delta_orien += 2 * np.pi

        if abs(delta_orien) > np.pi / 15:
            self.angular_vel = np.sign(delta_orien) * 1.2  # rad/sec
        else:
            self.angular_vel *= 0.5

    def step(self, dt):
        t0 = time.time()
        self.range_data, self.occupancy_grid = self.lidar.get_scan(self.world)
        t1 = time.time()
        # print("scan time = {}".format(t1-t0))

        self.lidar_segments = self.tracker.segment(self.range_data, self.pos)
        self.detections, walls = self.tracker.detect(self.lidar_segments, self.pos)

        self.tracks = self.tracker.track(self.detections)
        for track in self.tracks:
            if track.coasted: continue
            px, py = track.position()
            u, v = self.world.mapping_to_grid(px, py)
            if u < self.occupancy_grid.shape[0] and v < self.occupancy_grid.shape[1]:
                self.occupancy_grid[u-2:u+2, v-2:v+2] = 1

        self.follow(self.leader_ped)
        self.pos += self.vel * dt
        self.orien += self.angular_vel * dt
        if self.orien >  np.pi: self.orien -= 2 * np.pi
        if self.orien < -np.pi: self.orien += 2 * np.pi


class FollowBot(Robot):
    def __init__(self):
        super(FollowBot, self).__init__()


class Lidar1d:
    def __init__(self, robot):
        self.fov = 270          # degree        => SICK TiM571
        self.resolution = 1     # per degree    => SICK TiM571
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

    # TODO: move it to basics_2d and rename it to LiDAR.py
    def get_scan(self, world):
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
            dist = np.linalg.norm(ped.pos - self.robot_ptr.pos) - ped.radius
            if dist > self.max_dist:
                continue
            results, intersect_pts_ = Circle(ped.pos, ped.radius).intersect_many(cur_rays)
            is_not_nan = 1 - np.any(np.isnan(intersect_pts_), axis=1)
            results = np.bitwise_and(results, is_not_nan)
            intersect_pts = np.stack([intersect_pts_[ii] * r + 100000 * (1 - r) for ii, r in enumerate(results)])
            all_intersects.append(intersect_pts)

        all_intersects = np.stack(all_intersects)
        dists = all_intersects - cur_rays[0, 0]
        dists = np.linalg.norm(dists, axis=2)
        min_ind = np.argmin(dists, axis=0)
        scan = np.stack([all_intersects[ind, ii] for ii, ind in enumerate(min_ind)])

        # Occupancy Grid Map
        grid = np.ones_like(world.walkable, dtype=np.float) * 0.5
        for ii in range(len(cur_rays)):
            ray_i = cur_rays[ii]
            scan_i = scan[ii]
            white_line = [self.robot_ptr.pos, scan_i]
            line_len = np.linalg.norm(white_line[1] - white_line[0])

            for z in np.arange(0, line_len/self.max_dist, 0.002):
                px, py = z * ray_i[1] + (1-z) * ray_i[0]
                u, v = world.mapping_to_grid(px, py)
                grid[u, v] = 0

        return scan, grid

