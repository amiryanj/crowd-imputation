import sys
import math
import numpy as np
from followbot.basics_geometry import Circle, DoubleCircle, rotate
eps = sys.float_info.epsilon

class Pedestrian:
    def __init__(self, init_pos=[0, 0], init_vel=[0, 0]):
        self.id = -1  # to check if somebody is him!
        self.radius = 0.5
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.orien = lambda: math.atan2(self.vel[1], self.vel[0])
        self.vel_unitvec = lambda: (self.vel + eps) / (np.linalg.norm(self.vel) + eps)
        self.lateral_unitvec = lambda: (np.array([self.vel[1], -self.vel[0]]) + eps) / (np.linalg.norm(self.vel) + eps)
        self.trajectory = []

        # self.geometry = lambda: Circle(self.pos, self.radius)

        self.leg_radius = 0.10
        self.leg_dist   = 0.20
        self.geometry = lambda: DoubleCircle(self.pos + self.lateral_unitvec() * self.leg_dist,
                                             self.pos - self.lateral_unitvec() * self.leg_dist,
                                             self.leg_radius)

        # crowd sim features
        self.pref_speed = 0
        self.sim_radius = 1
        self.ccw = True

        # visualization features
        self.color = (255, 255, 255)  # for visualization


    # def step(self, dt=0.1):  # crowd sim alg
    #     for ped in self.world.crowds:
    #         if ped.id == self.id: continue
    #         dp = self.pos - ped.pos
    #         dv = self.vel - ped.vel
    #         dca = 0
    #         ttca = 0
