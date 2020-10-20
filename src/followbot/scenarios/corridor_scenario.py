# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import time

import numpy as np
from math import cos, sin, sqrt
import yaml
from poisson_distribution import PoissonDistribution

import followbot.crowdsim.crowdsim as crowdsim
from followbot.gui.visualizer import *
from followbot.scenarios.scenario import Scenario
from followbot.scenarios.simulation_scenario import SimulationScenario
from followbot.simulator.world import World
from followbot.util.basic_geometry import Line


class CorridorScenario(SimulationScenario):
    """
    -> Crowd standing in groups of (2, 3, 4) persons, in a corridor.
    -> The inter-group distance is bigger than intra-group distances
    -> The leader passes the corridor among crowd
    """
    def __init__(self, n_peds_=32, n_robots_=1, corridor_wid=5, corridor_len=18):
        super(CorridorScenario, self).__init__()
        self.corridor_wid = corridor_wid
        self.corridor_len = corridor_len
        self.biD_flow = False  # by default Uni-directional Flow

    def setup_crowd_sim(self):
        # self.world.sim = crowdsim.CrowdSim("powerlaw")
        self.world.sim.initSimulation(self.n_peds + 1)
        self.world.inertia_coeff = 0.1  # larger, more inertia, zero means no inertia

        for ii in range(self.n_peds):
            self.world.crowds[ii].pref_speed = np.random.uniform(1.2, 1.6)
            self.world.crowds[ii].radius = 0.25
            # self.world.sim.setAgentRadius(ii, self.world.crowds[ii].radius)
            # self.world.sim.setAgentSpeed(ii, self.world.crowds[ii].pref_speed)
            # self.world.sim.setAgentNeighborDist(ii, 4)
            # self.world.sim.setAgentTimeHorizon(ii, 2)

    def setup_agents(self):
        ped_poss = []
        ped_goals = []

        # # FixMe: Add the Leader agent
        # ped_poss.insert(0, [1.2, self.corridor_wid / 2])
        # ped_goals.insert(0, [self.corridor_len, self.corridor_wid / 2])

        poisson_distrib = PoissonDistribution((self.corridor_len - self.ped_radius * 4,
                                               self.corridor_wid - self.ped_radius * 4),
                                              minDist=1.6, k=10)

        group_centers = poisson_distrib.create_samples() + self.ped_radius * 2
        group_sizes = np.random.random_integers(1, 1, len(group_centers))

        for gg, g_size in enumerate(group_sizes):
            px_g = group_centers[gg][0]
            py_g = group_centers[gg][1]
            rotation = np.pi / 2  # np.random.uniform(0, np.pi)
            for ii in range(g_size):
                px_i = px_g + cos(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * sqrt(g_size)
                py_i = py_g + sin(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * sqrt(g_size)
                ped_poss.append([px_i, py_i])

                if self.biD_flow:
                    toss = np.random.rand() > 0.5  # FixMe
                else:
                    toss = True  # always the same direction

                if toss:
                    ped_goals.append([1000, self.corridor_wid / 2])
                else:
                    ped_goals.append([-1000, self.corridor_wid / 2])

        self.n_peds = len(ped_poss)
        return ped_poss, ped_goals

    def setup(self, biD_flow=True):
        self.biD_flow = biD_flow
        ped_poss, ped_goals = self.setup_agents()

        world_dim = [[0, self.corridor_len], [-self.corridor_len/2, self.corridor_len/2]]
        self.world = World(self.n_peds, self.n_robots, biped=False)
        self.setup_crowd_sim()
        self.visualizer = Visualizer(self.world, world_dim, (960, 960), 'Basic Corridor')

        # Two walls of the corridor
        # self.world.add_obstacle(Line([0, 0], [self.corridor_len, 0]))
        # self.world.add_obstacle(Line([0, self.corridor_wid], [self.corridor_len, self.corridor_wid]))
        self.world.obstacles = self.world.sim.obstacles

        # Probabilistic Occupancy Map, created and updated by Robot
        pom_resolution = 4  # per meter
        self.world.walkable = np.ones((self.corridor_len * pom_resolution,
                                       self.corridor_wid * pom_resolution), dtype=bool)
        self.world.POM = np.ones_like(self.world.walkable)
        self.world.mapping_to_grid = lambda x, y: (int(x * pom_resolution), int(y * pom_resolution))
        # ======================

        for ped_ind in range(len(ped_poss)):
            self.world.set_ped_position(ped_ind, ped_poss[ped_ind])
            self.world.set_ped_velocity(ped_ind, [0, 0])
            self.world.set_ped_goal(ped_ind, ped_goals[ped_ind])
            # if ped_ind == 0: continue
            self.world.crowds[ped_ind].color = RED_COLOR

        self.world.sim.setTime(0)
        self.visualizer.update()

    def step(self, dt, save=False):
        super(CorridorScenario, self).step(dt, save)

