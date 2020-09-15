# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import time

import numpy as np
from math import cos, sin, sqrt
import yaml
from poisson_distribution import PoissonDistribution

import followbot.crowdsim.crowdsim as crowdsim
from followbot.gui.display import *
from followbot.scenarios.scenario import Scenario
from followbot.scenarios.simulation_scenario import SimulationScenario
from followbot.simulator.world import World
from followbot.util.basic_geometry import Line


class CorridorCrowd(SimulationScenario):
    """
    -> Crowd standing in groups of (2, 3, 4) persons, in a corridor.
    -> The inter-group distance is bigger than intra-group distances
    -> The leader passes the corridor among crowd
    """
    def __init__(self, n_peds_=32, n_robots_=1, corridor_wid=5, corridor_len=18):
        super(CorridorCrowd, self).__init__()
        self.world = []
        self.display = []
        self.n_peds = n_peds_
        self.n_robots = n_robots_
        self.corridor_wid = corridor_wid
        self.corridor_len = corridor_len
        self.ped_radius = 0.25

    def setup_crowd_sim(self):
        self.world.sim = crowdsim.CrowdSim("powerlaw")
        self.world.sim.initSimulation(self.n_peds + 1)
        self.world.inertia_coeff = 0.1  # larger, more inertia, zero means no inertia

        for ii in range(self.n_peds):
            self.world.crowds[ii].pref_speed = 1.3
            self.world.crowds[ii].radius = 0.25
            self.world.sim.setAgentRadius(ii, self.world.crowds[ii].radius * 2)
            self.world.sim.setAgentSpeed(ii, self.world.crowds[ii].pref_speed)
            self.world.sim.setAgentNeighborDist(ii, 4)
            self.world.sim.setAgentTimeHorizon(ii, 2)

    def setup(self):
        ped_poss = []
        poisson_distrib = PoissonDistribution((self.corridor_len - self.ped_radius * 4,
                                               self.corridor_wid - self.ped_radius * 4),
                                              minDist=1.3, k=10)

        group_centers = poisson_distrib.create_samples() + self.ped_radius * 2
        group_sizes = np.random.random_integers(1, 1, len(group_centers))

        for gg, g_size in enumerate(group_sizes):
            px_g = group_centers[gg][0]
            py_g = group_centers[gg][1]
            rotation = np.random.uniform(0, np.pi)
            for ii in range(g_size):
                px_i = px_g + cos(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * sqrt(g_size)
                py_i = py_g + sin(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * sqrt(g_size)
                ped_poss.append([px_i, py_i])

        self.n_peds = len(ped_poss)  # the random algorithm may return a different number of agents than what is asked

        world_dim = [[0, self.corridor_len], [-self.corridor_len/2, self.corridor_len/2]]
        self.world = World(self.n_peds, self.n_robots, biped=False)
        self.setup_crowd_sim()

        self.display = Display(self.world, world_dim, (960, 960), 'Basic Corridor')

        # Two walls of the corridor
        self.world.add_object(Line([0, 0], [self.corridor_len, 0]))
        self.world.add_object(Line([0, self.corridor_wid], [self.corridor_len, self.corridor_wid]))

        pom_resolution = 4  # per meter
        self.world.walkable = np.ones((self.corridor_len * pom_resolution,
                                       self.corridor_wid * pom_resolution), dtype=bool)
        self.world.POM = np.ones_like(self.world.walkable)

        self.world.mapping_to_grid = lambda x, y: (int(x * pom_resolution), int(y * pom_resolution))

        for ped_ind in range(len(ped_poss)):
            self.world.set_ped_position(ped_ind, ped_poss[ped_ind])
            self.world.set_ped_velocity(ped_ind, [0, 0])
            if np.random.rand() > 0.5:
                self.world.set_ped_goal(ped_ind, [1000, self.corridor_wid / 2])
            else:
                self.world.set_ped_goal(ped_ind, [-1000, self.corridor_wid / 2])

            if ped_ind == 0: continue
            self.world.crowds[ped_ind].color = RED_COLOR
        # Set the goal of Leader agent
        self.world.set_ped_goal(0, [self.corridor_len, self.corridor_wid / 2])

        # set the Robot position just behind first ped
        ped0_pos = self.world.crowds[0].pos
        self.world.set_robot_position(0, [ped0_pos[0] - 1, ped0_pos[1]])
        self.world.set_robot_leader(0, 0)
        self.world.sim.setTime(0)
        self.display.update()

    def step(self, save=False):
        if not self.world.pause:
            dt = 0.05

            self.world.sim.doStep(dt)
            for ii in range(self.n_peds):
                p = self.world.sim.getCenterNext(ii)
                v = self.world.sim.getCenterVelocityNext(ii)
                # apply inertia
                v_new = np.array(v) * (1 - self.world.inertia_coeff) \
                        + self.world.crowds[ii].vel * self.world.inertia_coeff
                p_new = self.world.crowds[ii].pos + v_new * dt

                self.world.set_ped_position(ii, p_new)
                self.world.set_ped_velocity(ii, v_new)

            self.world.step_robot(dt)

        self.update_disply()
        super(CorridorCrowd, self).step()

