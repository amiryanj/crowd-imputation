import numpy as np
from math import cos, sin, tan, tanh, acos
from followbot.basic_geometry import Circle


class LiDAR2D:
    def __init__(self, robot_ptr):
        # specification: default values from SICK TiM571 datasheet
        self.range_min = 0.05   # meter
        self.range_max = 8      # meter (actually this is the range typical of the sensor)
        # self.range_max = 25   #

        self.fov = 340          # in degree
        self.resolution = 3     # per degree
        self.angle_min_radian = lambda: - self.fov / 2 * np.pi / 180.
        self.angle_max_radian = lambda: + self.fov / 2 * np.pi / 180.
        self.angle_increment_radian = lambda: (1 / self.resolution) * np.pi / 180

        self.scanning_freq = 15.  # Hz
        self.time_increment = lambda: 1. / self.scanning_freq

        self.systematic_error_abs = 0.060   # m
        self.statistical_error = 0.020      # m

        self.data_type = np.float32
        self.robot_ptr = robot_ptr

        # pre-computed ray angles
        self.angles = []
        self.rays = []
        for angle in np.arange(-self.fov / 2, self.fov / 2, 1 / self.resolution):
            alpha = angle * np.pi / 180
            ray_end = [np.cos(alpha) * self.range_max, np.sin(alpha) * self.range_max]
            self.rays.append([[0, 0], ray_end])
        self.rays = np.stack(self.rays)

        # scan results
        self.last_range_pnts = []
        self.last_range_data = []
        self.last_intensities = []
        self.last_occupancy_gridmap = []

    def scan(self, world, update_gridmap=False, walkable_area=[]):
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
            intersect_pts = np.stack([res * intersect_pts_[ii] + (1-res) * (self.range_max + self.robot_ptr.pos) for ii, res in enumerate(results)])
            all_intersects.append(intersect_pts)

        for kk, ped in enumerate(world.crowds):
            dist = np.linalg.norm(ped.pos - self.robot_ptr.pos) - ped.radius - self.robot_ptr.radius
            if dist > self.range_max:
                continue
            ped_geometry = ped.geometry()
            results, intersect_pts_ = ped_geometry.intersect_many(cur_rays)
            # if kk == 0:
            #     print(intersect_pts_)
            is_not_nan = 1 - np.any(np.isnan(intersect_pts_), axis=1)
            results = np.bitwise_and(results, is_not_nan)
            intersect_pts = np.stack([res * intersect_pts_[ii] + (1-res) * (self.range_max + self.robot_ptr.pos) for ii, res in enumerate(results)])
            all_intersects.append(intersect_pts)

        all_intersects = np.stack(all_intersects)
        dists = all_intersects - cur_rays[0, 0]
        dists = np.linalg.norm(dists, axis=2)
        min_ind = np.argmin(dists, axis=0)
        self.last_range_pnts = np.stack([all_intersects[ind, ii] for ii, ind in enumerate(min_ind)])
        self.last_range_data = np.sqrt(np.power(self.last_range_pnts[:, 0] - self.robot_ptr.pos[0], 2)
                                     + np.power(self.last_range_pnts[:, 1] - self.robot_ptr.pos[1], 2))

        self.last_intensities = np.zeros_like(self.last_range_data)

        if update_gridmap:
            self.last_occupancy_gridmap = np.ones_like(walkable_area, dtype=np.float) * 0.5
            for ii in range(len(cur_rays)):
                ray_i = cur_rays[ii]
                scan_i = self.last_range_pnts[ii]
                white_line = [self.robot_ptr.pos, scan_i]
                line_len = np.linalg.norm(white_line[1] - white_line[0])

                for z in np.arange(0, line_len/self.range_max, 0.05):
                    px, py = z * ray_i[1] + (1-z) * ray_i[0]
                    u, v = world.mapping_to_grid(px, py)
                    self.last_occupancy_gridmap[u, v] = 0


