import numpy as np


class Pedestrian:
    def __init__(self, loc=[0, 0]):
        self.id = -1  # to check if somebody is him!
        self.radius = 0.5
        self.pos = np.array(loc)
        self.vel = np.array([0, 0])
        self.pref_speed = 0
        self.ccw = True
        self.color = (255,255,255)  # for visualization
        self.trajectory = []


    # def step(self, dt=0.1):  # crowd sim alg
    #     for ped in self.world.crowds:
    #         if ped.id == self.id: continue
    #         dp = self.pos - ped.pos
    #         dv = self.vel - ped.vel
    #         dca = 0
    #         ttca = 0
