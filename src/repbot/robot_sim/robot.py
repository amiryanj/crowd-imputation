import time

import numpy as np

from repbot.robot_sim.robot_world import RobotWorld
from repbot.crowd_tracking.human_detection import PedestrianDetection
from repbot.crowd_tracking.multi_obj_tracking import MultiObjectTracking
from repbot.crowd_tracking.lidar2d import LiDAR2D
# from repbot.basics_geometry import Circle
from repbot.util.mapped_array import MappedArray


# compute robot velocity according to the distance to the goal
# Keep pref speed as long as it's not within a  of the goal
def trapezoidal_motion_profile(loc, goal, pref_speed, deceleration_dist=1.0):
    """
    compute and return the velocity vector for the robot, according to trapezoidal motion profile
    like this: /‾‾‾‾‾‾‾\
    :param loc: current location of the robot
    :param goal: goal coordinate of the robot
    :param pref_speed: preferred speed of the robot ( m / s)
    :param deceleration_dist: (in meter) the distance to the goal after which robot starts to reduce speed
    :return: the velocity vector for the robot
    """
    vector_to_goal = goal - loc
    dist_to_goal = np.linalg.norm(vector_to_goal)

    if dist_to_goal > deceleration_dist:
        robot_vel = pref_speed * vector_to_goal / dist_to_goal
    else:
        robot_vel = pref_speed * vector_to_goal / deceleration_dist
    return robot_vel


class MyRobot:
    def __init__(self, worldPtr, prefSpeed, sensorFps, numHypothesisWorlds=1, mapResolution=4):
        self.real_world = worldPtr  # pointer to world

        # robot dynamic properties
        self.pos = np.array([0, 0], float)
        self.orien = 0.0  # radian
        self.vel = np.array([0, 0], float)
        self.angular_vel = 0

        # robot static properties
        self.radius = 0.3
        self.pref_speed = prefSpeed
        self.max_speed = 1.3

        # child objects that do some function
        self.lidar = LiDAR2D(robot_ptr=self)
        self.ped_detector = PedestrianDetection(self.lidar.range_max, np.deg2rad(1 / self.lidar.resolution))
        self.tracker = MultiObjectTracking(sensorFps)

        # Robot Goal: can be static/dynamic object: e.g. fixed point / leader person
        self.goal = [0, 0]

        # the sensory data + processed variables
        # ====================================
        self.mapped_array_resolution = mapResolution  # per meters
        self.occupancy_map = MappedArray(min_x=self.real_world.world_dim[0][0], max_x=self.real_world.world_dim[0][1],
                                         min_y=self.real_world.world_dim[1][0], max_y=self.real_world.world_dim[1][1],
                                         resolution=self.mapped_array_resolution,  # per meter
                                         n_channels=1, dtype=np.float)
        # at each location, it shows a categorical/cont. value that represent a different type of flow
        self.crowd_territory_map = MappedArray(min_x=self.real_world.world_dim[0][0], max_x=self.real_world.world_dim[0][1],
                                          min_y=self.real_world.world_dim[1][0], max_y=self.real_world.world_dim[1][1],
                                          resolution=self.mapped_array_resolution,  # per meter
                                          n_channels=1, dtype=np.float)

        # blind spot map is 1 everywhere robot can not see by its lidar
        self.blind_spot_map = MappedArray(min_x=self.real_world.world_dim[0][0], max_x=self.real_world.world_dim[0][1],
                                          min_y=self.real_world.world_dim[1][0], max_y=self.real_world.world_dim[1][1],
                                          resolution=self.mapped_array_resolution,  # per meter
                                          n_channels=1, dtype=np.float)
        walkable_map = MappedArray(min_x=self.real_world.world_dim[0][0], max_x=self.real_world.world_dim[0][1],
                                   min_y=self.real_world.world_dim[1][0], max_y=self.real_world.world_dim[1][1],
                                   resolution=self.mapped_array_resolution,  # per meter
                                   n_channels=1, dtype=np.float)
        walkable_map.fill(1)

        self.lidar_clusters = []
        self.detected_peds = []
        self.tracks = []
        self.hypothesis_worlds = [RobotWorld() for _ in range(numHypothesisWorlds)]
        for hypo_world in self.hypothesis_worlds:
            hypo_world.walkable_map = walkable_map

        # ====================================

    def init(self, init_pos):
        self.real_world.set_robot_position(0, [init_pos[0], init_pos[1]])

    # TODO: work with ROS
    def step(self, dt, lidar_enabled):
        # post-process of robot movement process
        self.pos += self.vel * dt
        self.orien += self.angular_vel * dt
        if self.orien > np.pi: self.orien -= 2 * np.pi
        if self.orien < -np.pi: self.orien += 2 * np.pi

        if lidar_enabled:
            start = time.time()
            self.lidar.scan(self.real_world)
            end = time.time()
            # print("scan time:", end - start)
            for hw in self.hypothesis_worlds:
                hw.update(self.lidar.data.last_range_data, self.tracker._tracks)
