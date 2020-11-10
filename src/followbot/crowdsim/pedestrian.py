import sys
from math import cos, sin, atan2
import numpy as np
import scipy

from followbot.crowdsim.gait_sim import MocapGaitSimulator
from followbot.util.basic_geometry import Circle, DoubleCircle

eps = sys.float_info.epsilon


class Pedestrian:
    def __init__(self, init_pos=[0, 0], init_vel=[0, 0], biped=False):
        self.id = -1  # to check if somebody is him!
        self.radius = 0.25
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.orien = lambda: atan2(self.vel[1], self.vel[0])
        self.vel_unitvec = lambda: (self.vel + eps) / (np.linalg.norm(self.vel) + eps)
        self.lateral_unitvec = lambda: (np.array([self.vel[1], -self.vel[0]]) + eps) / (np.linalg.norm(self.vel) + eps)
        self.trajectory = []
        self.biped = biped

        # FIXME: there are 2 options to represent pedestrian:
        #  1: using circular agents for simple detection
        #  2: using 2 legs for use with DROW

        if biped:
            self.mocap_walk = MocapGaitSimulator()
            self.leg_radius = 0.075
            self.leg_dist   = 0.25
            self.geometry = lambda: DoubleCircle(self.pos + self.lateral_unitvec() * self.leg_dist /2.,
                                                 self.pos - self.lateral_unitvec() * self.leg_dist /2.,
                                                 self.leg_radius)
        else:
            self.geometry = lambda: Circle(self.pos, self.radius)

        # crowd sim features
        self.pref_speed = 0
        self.sim_radius = 1
        # self.ccw = True

        # visualization features
        self.color = (255, 255, 255)  # for visualization

    def step(self, dt):
        self.mocap_walk.step(dt)
        self_orien = self.orien() + np.pi/2
        rot_mat = np.array([[cos(self_orien), -sin(self_orien)], [sin(self_orien), cos(self_orien)]])
        right_leg_pos = self.pos + np.matmul(rot_mat, self.mocap_walk.right_leg)
        left_leg_pos = self.pos + np.matmul(rot_mat, self.mocap_walk.left_leg)
        self.geometry = lambda: DoubleCircle(left_leg_pos, right_leg_pos, self.leg_radius)

if __name__ == "__main__":
    MocapGaitSimulator().test()
    print(__file__)
