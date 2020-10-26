import numpy as np

from followbot.robot_functions.robot_world import RobotWorld
from followbot.robot_functions.tracking import PedestrianDetection, MultiObjectTracking
from followbot.robot_functions.lidar2d import LiDAR2D
# from followbot.basics_geometry import Circle


class MyRobot:
    def __init__(self, numBeliefWorlds=1):
        self.real_world = []  # pointer to world

        # robot dynamic properties
        self.pos = np.array([0, 0], float)
        self.orien = 0.0  # radian
        self.vel = np.array([0, 0], float)
        self.angular_vel = 0

        # robot static properties
        self.radius = 0.4
        self.pref_speed = 1.2
        self.max_speed = 2.0

        # child objects that do some function
        self.lidar = LiDAR2D(robot_ptr=self)
        self.ped_detector = PedestrianDetection(self.lidar.range_max, np.deg2rad(1/self.lidar.resolution))
        self.tracker = MultiObjectTracking()

        # Robot Goal: can be static/dynamic object: e.g. fixed point / leader person
        self.goal = [0, 0]

        # the sensory data + processed variables
        # ====================================
        self.POM = np.empty(shape=(0, 0))
        self.crowd_flow_map = np.empty(shape=(0, 0))
        self.lidar_segments = []
        self.detected_peds = []
        self.tracks = []
        self.belief_worlds = [RobotWorld()] * numBeliefWorlds
        # ====================================


    def init(self, init_pos):
        self.real_world.set_robot_position(0, [init_pos[0], init_pos[1]])

    # default method for robot to update the velocity
    def update_next_vel(self, dt):
        vector_to_goal = self.goal - self.pos
        dist_to_goal = np.linalg.norm(vector_to_goal)
        if dist_to_goal > 0.5:
            self.vel = self.pref_speed * vector_to_goal / dist_to_goal
        else:
            self.vel = 0.4 * vector_to_goal / dist_to_goal

    def step(self, dt):

        update_pom = False  # Fixme
        self.lidar.scan(self.real_world, update_pom, walkable_area=self.real_world.walkable)

        if update_pom:
            pom_new = self.lidar.last_occupancy_gridmap.copy()
            seen_area_indices = np.where(pom_new != 0)
            self.POM[:] = self.POM[:] * 0.4 + 0.5
            self.POM[seen_area_indices] = 0
            for track in self.tracks:
                if track.coasted: continue
                px, py = track.position()
                u, v = self.mapping_to_grid(px, py)
                if 0 <= u < self.POM.shape[0] and 0 <= v < self.POM.shape[1]:
                    self.POM[u - 2:u + 2, v - 2:v + 2] = 1

        self.update_next_vel(dt)
        # FixMe: Here is the post-process of step process of the robot
        #  call it at the end of overridden function
        # TODO: work with ROS
        self.pos += self.vel * dt
        if self.pos[0] > self.real_world.world_dim[0][1]:
            self.pos[0] = self.real_world.world_dim[0][0]
        self.orien += self.angular_vel * dt
        if self.orien >  np.pi: self.orien -= 2 * np.pi
        if self.orien < -np.pi: self.orien += 2 * np.pi

        for w in self.belief_worlds:
            w.update(self.lidar.last_range_data, self.tracker.tracks)

