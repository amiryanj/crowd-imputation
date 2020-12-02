# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy.linalg import norm

from followbot.robot_functions.robot import MyRobot


class RobotReplaceHuman(MyRobot):
    def __init__(self, robot_poss, robot_vels, worldPtr, numHypothesisWorlds=1):
        super(RobotReplaceHuman, self).__init__(worldPtr, 1, numHypothesisWorlds)
        self.replaced_ped_id = []  # will be used depending on the task
        self.robot_poss = robot_poss
        self.robot_vels = robot_vels

    def update_next_vel(self, dt):
        ped = self.replaced_ped_id

    def init(self):
        # set the Robot position just behind first ped on the x axis
        self.real_world.set_robot_position(0, self.robot_poss[0])

    def step(self, dt, lidar_enabled):
        frame_id = self.real_world.time
        self.vel = (self.robot_poss[frame_id] - self.robot_poss[frame_id - 1]) / dt
        self.angular_vel = 0
        self.orien = np.arctan2(self.robot_vels[self.real_world.time, 1], self.robot_vels[self.real_world.time, 0])
        super(RobotReplaceHuman, self).step(dt, lidar_enabled)

