import sys
from math import cos, sin, atan2
import numpy as np
import scipy

from repbot.crowd_sim.gait_sim import MocapGaitSimulator
from repbot.util.basic_geometry import Circle, DoubleCircle

eps = sys.float_info.epsilon


class Pedestrian:
    def __init__(self, init_pos=[0, 0], init_vel=[0, 0], biped_mode=False, color=(0, 255, 0)):
        self.id = -1  # to check if somebody is him!
        self.radius = 0.25
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.orien = lambda: atan2(self.vel[1], self.vel[0])
        self.vel_unitvec = lambda: (self.vel + eps) / (np.linalg.norm(self.vel) + eps)
        self.lateral_unitvec = lambda: (np.array([self.vel[1], -self.vel[0]]) + eps) / (np.linalg.norm(self.vel) + eps)
        self.trajectory = []
        self.biped_mode = biped_mode  # boolean

        # FIXME: there are 2 options to represent pedestrian:
        #  1: using circular agents for simple detection
        #  2: using 2 legs for use with DROW

        if biped_mode:
            self.mocap_walk = MocapGaitSimulator()
            self.mocap_walk.progress_time = np.random.uniform(0, self.mocap_walk.period_duration)
            self.leg_radius = 0.06
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
        self.color = color  # for visualization

    def set_new_position(self, pos):
        self.pos = np.array(pos, np.float)
        # append the position to pedestrian trajectory
        # traj_ii = self.trajectory
        if len(self.trajectory) and np.linalg.norm(self.trajectory[-1] - np.array(pos)) > 2.0:
            self.trajectory.clear()
        if len(self.trajectory) >= 150:
            del self.trajectory[0]
        self.trajectory.append(np.array(pos))


    def step(self, dt):
        if self.biped_mode:
            self.mocap_walk.step(dt)
            self_orien = self.orien() + np.pi/2
            rot_mat = np.array([[cos(self_orien), -sin(self_orien)], [sin(self_orien), cos(self_orien)]])
            right_leg_pos = self.pos + np.matmul(rot_mat, self.mocap_walk.right_leg)
            left_leg_pos = self.pos + np.matmul(rot_mat, self.mocap_walk.left_leg)
            # print("legs distance = ", np.linalg.norm(left_leg_pos - right_leg_pos))
            self.geometry = lambda: DoubleCircle(left_leg_pos, right_leg_pos, self.leg_radius)
        else:
            pass


if __name__ == "__main__":
    MocapGaitSimulator().test()
    print(__file__)
