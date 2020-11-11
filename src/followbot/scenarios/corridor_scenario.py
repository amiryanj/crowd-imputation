# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import time
import random

import yaml
import numpy as np
from math import cos, sin, sqrt
from poisson_distribution import PoissonDistribution

import followbot.crowdsim.crowdsim as crowdsim

from followbot.crowdsim.grouping_behavior import GroupingBehavior
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

    def __init__(self, n_peds_=32, n_robots_=1, biD_flow=False, group_size_choices=[1], corridor_wid=6,
                 corridor_len=30):
        super(CorridorScenario, self).__init__()
        self.corridor_wid = corridor_wid
        self.corridor_len = corridor_len
        self.biD_flow = False  # by default Uni-directional Flow
        self.group_size_choices = group_size_choices

    def setup_crowd_sim(self):
        # self.world.sim = crowdsim.CrowdSim("powerlaw")
        self.world.sim.initSimulation(0)
        self.world.inertia_coeff = 0.1  # larger, more inertia, zero means no inertia

        for ii in range(self.n_peds):
            self.world.crowds[ii].pref_speed = np.random.uniform(1.2, 1.3)
            self.world.crowds[ii].radius = 0.25
            self.world.sim.addAgent(x=-1, y=-1,
                                    radius=self.world.crowds[ii].radius,
                                    prefSpeed=self.world.crowds[ii].pref_speed,
                                    maxSpeed=1.6,
                                    maxAcceleration=5.0)
            # self.world.sim.setAgentRadius(ii, self.world.crowds[ii].radius)
            # self.world.sim.setAgentSpeed(ii, self.world.crowds[ii].pref_speed)
            # self.world.sim.setAgentNeighborDist(ii, 4)
            # self.world.sim.setAgentTimeHorizon(ii, 2)

    def setup_agents(self, working_area):
        ped_poss = []
        ped_goals = []

        # # FixMe: Add the Leader agent
        # ped_poss.insert(0, [1.2, self.corridor_wid / 2])
        # ped_goals.insert(0, [self.corridor_len, self.corridor_wid / 2])

        poisson_distrib = PoissonDistribution((working_area[0][1] - working_area[0][0] - self.ped_radius * 4,
                                               working_area[1][1] - working_area[1][0] - self.ped_radius * 4),
                                              minDist=2.6, k=5)

        group_centers = poisson_distrib.create_samples() + self.ped_radius * 2 \
                        + np.array([working_area[0][0], working_area[1][0]])

        # Todo: delete the groups out of the corridor bottleneck

        # setup group-mates
        n_groups = len(group_centers)
        group_sizes = [random.choice(self.group_size_choices) for _ in range(n_groups)]

        group_size_cum = np.cumsum(group_sizes) - group_sizes[0]
        group_ids = [[k + group_size_cum[ii] for k in range(group_sizes[ii])] for ii in range(n_groups)]

        for gg, g_size in enumerate(group_sizes):
            px_g = group_centers[gg][0]
            py_g = group_centers[gg][1]
            rotation = np.pi / 2  # np.random.uniform(0, np.pi)
            for ii in range(g_size):
                px_i = px_g + cos(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * sqrt(g_size)
                py_i = py_g + sin(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * sqrt(g_size)
                ped_poss.append([px_i, py_i])

                # direction = (2 * round(np.random.rand()) - 1) if self.biD_flow else 1
                direction = np.sign(py_g)
                ped_goals.append([1000 * direction, 0])

        self.n_peds = len(ped_poss)
        return ped_poss, ped_goals, group_ids

    def setup(self):
        world_dim = [[-self.corridor_len / 2, self.corridor_len / 2],
                     [-self.corridor_wid / 2, self.corridor_wid / 2]]
        ped_locs, ped_goals, self.group_ids = self.setup_agents(world_dim)

        self.world = World(self.n_peds, self.n_robots, world_dim=world_dim, biped=False)
        self.setup_crowd_sim()
        self.grouping_behavior_handler = GroupingBehavior(self.group_ids)

        # Two walls of the corridor
        # self.world.add_obstacle(Line([0, 0], [self.corridor_len, 0]))
        # self.world.add_obstacle(Line([0, self.corridor_wid], [self.corridor_len, self.corridor_wid]))
        self.world.obstacles = self.world.sim.obstacles

        # init
        # ======================
        for ped_ind in range(len(ped_locs)):
            self.world.set_ped_position(ped_ind, ped_locs[ped_ind])
            self.world.set_ped_velocity(ped_ind, [0, 0])
            self.world.set_ped_goal(ped_ind, ped_goals[ped_ind])
            if ped_goals[ped_ind][0] - ped_locs[ped_ind][0] > 0:
                self.world.crowds[ped_ind].color = RED_COLOR
            else:
                self.world.crowds[ped_ind].color = BLUE_COLOR

        self.world.set_time(0)

    def step(self, dt, save=False):
        super(CorridorScenario, self).step(dt, save)
