import numpy as np
from math import cos, sin


class LiDAR2D:
    def __init__(self, robot_ptr):
        # specification: default values from SICK TiM571 datasheet
        self.range_min = 0.05   # meter
        self.range_max = 8      # meter (actually this is the range typical of the sensor)
        # self.range_max = 25   #

        self.fov = 340          # in degree
        self.resolution = 2     # per degree
        self.angle_min_radian = lambda: - self.fov / 2 * np.pi / 180.
        self.angle_max_radian = lambda: + self.fov / 2 * np.pi / 180.
        self.angle_increment_radian = lambda: (1 / self.resolution) * np.pi / 180

        self.scanning_freq = 15.  # Hz
        self.time_increment = lambda: 1. / self.scanning_freq

        self.systematic_error_abs = 0.060   # m
        self.statistical_error = 0.020      # m

        self.data_type = np.float32
        self.robot_ptr = robot_ptr

        # scan results
        class Data:
            def __init__(self):
                self.last_rotated_rays = []
                self.last_points = []
                self.last_range_data = []
                self.last_intensities = []
                self.last_occupancy_gridmap = []
        self.data = Data()

    def scan(self, world):
        angles = np.deg2rad(np.arange(-self.fov / 2, self.fov / 2, 1 / self.resolution))
        rays_end = (np.array([np.cos(angles), np.sin(angles)]) * self.range_max).T
        rays_origin = np.zeros_like(rays_end)
        rays = np.stack([rays_origin, rays_end], axis=1)

        self.data.last_rotated_rays = rays

        # Rotate (to robot face)
        rot_matrix = np.array([[cos(self.robot_ptr.orien), -sin(self.robot_ptr.orien)],
                               [sin(self.robot_ptr.orien), cos(self.robot_ptr.orien)]])
        self.data.last_rotated_rays[:, 1, :] = np.matmul(rot_matrix, self.data.last_rotated_rays[:, 1, :].transpose()).transpose()

        # Translate (to robot position)
        self.data.last_rotated_rays[:, 0, :] = self.data.last_rotated_rays[:, 0, :] + self.robot_ptr.pos
        self.data.last_rotated_rays[:, 1, :] = self.data.last_rotated_rays[:, 1, :] + self.robot_ptr.pos

        # scan the obstacles
        all_intersects = [self.data.last_rotated_rays[:, 1]]
        for obs in world.obstacles:
            results, intersect_pts_ = obs.intersect_many(self.data.last_rotated_rays)
            is_not_nan = 1 - np.any(np.isnan(intersect_pts_), axis=1)
            results = np.bitwise_and(results, is_not_nan)
            intersect_pts = np.stack([res * intersect_pts_[ii] + (1-res) * (self.range_max + self.robot_ptr.pos)
                                      for ii, res in enumerate(results)])
            all_intersects.append(intersect_pts)

        # scan the pedestrians
        for kk, ped in enumerate(world.crowds):
            dist = np.linalg.norm(ped.pos - self.robot_ptr.pos) - ped.radius - self.robot_ptr.radius
            if dist > self.range_max:
                continue
            ped_geometry = ped.geometry()
            results, intersect_pts_ = ped_geometry.intersect_many(self.data.last_rotated_rays)
            is_not_nan = 1 - np.any(np.isnan(intersect_pts_), axis=1)
            results = np.bitwise_and(results, is_not_nan)
            intersect_pts = np.stack([res * intersect_pts_[ii] + (1-res) * (self.range_max + self.robot_ptr.pos)
                                      for ii, res in enumerate(results)])
            all_intersects.append(intersect_pts)

        # combine the intersection points
        all_intersects = np.stack(all_intersects)
        dists = all_intersects - self.data.last_rotated_rays[0, 0]
        dists = np.linalg.norm(dists, axis=2)
        min_ind = np.argmin(dists, axis=0)
        self.data.last_points = np.stack([all_intersects[ind, ii] for ii, ind in enumerate(min_ind)])
        self.data.last_range_data = np.sqrt(np.power(self.data.last_points[:, 0] - self.robot_ptr.pos[0], 2) +
                                            np.power(self.data.last_points[:, 1] - self.robot_ptr.pos[1], 2))

        # Todo: set lidar intensities
        self.data.last_intensities = np.zeros_like(self.data.last_range_data)

