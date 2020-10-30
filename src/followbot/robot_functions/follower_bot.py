# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy.linalg import norm

from followbot.robot_functions.robot import MyRobot


class FollowerBot(MyRobot):
    def __init__(self):
        super(FollowerBot, self).__init__()
        self.leader_ped = []  # will be used depending on the task

    def update_next_vel(self, dt):
        ped = self.leader_ped
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
            self.angular_vel *= abs(delta_orien) / (np.pi / 30)

    def init(self, leader_pos):
        # set the Robot position just behind first ped on the x axis
        self.real_world.set_robot_position(0, [leader_pos[0] - 1, leader_pos[1]])

    def set_leader_ped(self, leader_ped):
        self.leader_ped = leader_ped
        self.init(self.leader_ped.pos)

    def step(self, dt):
        super(FollowerBot, self).step(dt)
