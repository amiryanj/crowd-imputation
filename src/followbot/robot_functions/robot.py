import numpy as np

# from followbot.basics_geometry import Circle
from followbot.robot_functions.tracking import PedestrianDetection, MultiObjectTracking
from followbot.robot_functions.lidar2d import LiDAR2D


class MyRobot:
    def __init__(self):
        self.world = []  # pointer to world
        self.lidar = LiDAR2D(robot_ptr=self)
        self.ped_detector = PedestrianDetection(self.lidar.range_max, np.deg2rad(1/self.lidar.resolution))
        self.tracker = MultiObjectTracking()
        self.pos = np.array([0, 0], float)
        self.orien = 0.0  # radian
        self.vel = np.array([0, 0], float)
        self.angular_vel = 0
        self.radius = 0.4
        self.max_speed = 2.0

        # Todo: goal can be a dynamic object: e.g. a Leader person
        #       or a static point
        self.goal = [0, 0]  # will be used depending on the task

        self.lidar_segments = []
        self.detected_peds = []
        self.tracks = []
        self.pom = []

    def init(self, init_pos):
        self.world.set_robot_position(0, [init_pos[0], init_pos[1]])

    def update_next_vel(self, dt):
        vector_to_goal = self.goal - self.pos
        dist_to_goal = np.linalg.norm(vector_to_goal)
        if dist_to_goal > 0.5:
            self.vel = 1.2 * vector_to_goal / dist_to_goal
        else:
            self.vel = 0.4 * vector_to_goal / dist_to_goal

    def step(self, dt):
        self.update_next_vel(dt)
        # FixMe: Here is the post-process of step process of the robot
        #  call it at the end of overridden function
        # TODO: work with ROS
        self.pos += self.vel * dt
        self.orien += self.angular_vel * dt
        if self.orien >  np.pi: self.orien -= 2 * np.pi
        if self.orien < -np.pi: self.orien += 2 * np.pi

