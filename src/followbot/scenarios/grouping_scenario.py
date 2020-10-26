# Author: Javad Amirian
# Email: amiryan.j@gmail.com


from poisson_distribution import PoissonDistribution

import numpy as np
from math import sin, cos, sqrt
from sklearn.metrics.pairwise import euclidean_distances

import followbot.crowdsim.crowdsim as crowdsim
from followbot.crowdsim.grouping_behavior import GroupingBehavior
from followbot.scenarios.corridor_scenario import CorridorScenario


class GroupScenario(CorridorScenario):
    """
    2D couples scenario
    """

    def __init__(self):
        super(GroupScenario, self).__init__()
        self.group_behavior_handler = GroupingBehavior()
        self.grouped_agents = []

    def setup_agents(self):
        ped_poss = []
        ped_goals = []

        # # FixMe: Add the Leader agent
        # ped_poss.insert(0, [1.2, self.corridor_wid / 2])
        # ped_goals.insert(0, [self.corridor_len, self.corridor_wid / 2])
        # self.grouped_agents.append([0])

        poisson_distrib = PoissonDistribution((self.corridor_len - self.ped_radius * 4,
                                               self.corridor_wid - self.ped_radius * 4),
                                              minDist=1.9, k=10)

        group_centers = poisson_distrib.create_samples() + self.ped_radius * 2
        group_sizes = np.random.random_integers(2, 2, len(group_centers))

        for gg, g_size in enumerate(group_sizes):
            self.grouped_agents.append([x for x in range(len(ped_poss), len(ped_poss) + g_size)])

            px_g = group_centers[gg][0]
            py_g = group_centers[gg][1]
            rotation = np.pi / 2  # np.random.uniform(0, np.pi)

            random_toss = (np.random.rand() > 0.5)
            # random_toss = 0
            for ii in range(g_size):
                px_i = px_g + cos(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * 1.2
                py_i = py_g + sin(rotation + (ii / g_size) * np.pi * 2) * self.ped_radius * 1.2
                ped_poss.append([px_i, py_i])
                if random_toss:
                    ped_goals.append([1000, self.corridor_wid / 2])  # right end of the corridor
                else:
                    ped_goals.append([-1000, self.corridor_wid / 2])  # left end of the corridor

        self.n_peds = len(ped_poss)
        return ped_poss, ped_goals

    def setup_crowd_sim(self):
        # self.world.sim = crowdsim.CrowdSim("rvo2")
        self.world.sim.initSimulation(self.n_peds + 1)
        self.world.inertia_coeff = 0.1  # larger, more inertia, zero means no inertia

        for ii in range(self.n_peds):
            self.world.crowds[ii].pref_speed = np.random.uniform(1., 1.1)
            self.world.crowds[ii].radius = 0.25
            # self.world.sim.setAgentRadius(ii, self.world.crowds[ii].radius)
            # self.world.sim.setAgentSpeed(ii, self.world.crowds[ii].pref_speed)
            # self.world.sim.setAgentNeighborDist(ii, 4)
            # self.world.sim.setAgentTimeHorizon(ii, 2)

    def setup(self):
        super(GroupScenario, self).setup()

    def step(self, dt, save=False):
        #  override original step() function to apply grouping behavior
        #  1. be close to the center-of-mass of your group (attraction force)
        #  2. be away from center-of-mass of the nearest group (repulsion force)
        if not self.world.pause:
            self.world.sim.doStep(dt)

            # grouping behavior
            # ==========================
            center_of_groups = {}
            for group_k in self.grouped_agents:
                center_of_group_k = np.zeros(2)
                for jj in group_k:
                    center_of_group_k += self.world.crowds[jj].pos / len(group_k)
                for jj in group_k:
                    center_of_groups[jj] = center_of_group_k

            DD = euclidean_distances(list(center_of_groups.values()))
            DD[DD < 1E-6] = np.Infinity
            nearest_group_inds = np.argmin(DD, axis=0)  # the index of closest group to each agent
            # ==========================

            for ii in range(self.n_peds):
                p = self.world.sim.getCenterNext(ii)
                v = self.world.sim.getCenterVelocityNext(ii)

                # grouping behavior
                v_group = np.zeros(2, float)
                vec_to_ego_group_center = center_of_groups[ii] - self.world.crowds[ii].pos
                if np.linalg.norm(vec_to_ego_group_center) > self.ped_radius * 1.3:
                    v_group = (vec_to_ego_group_center / (np.linalg.norm(vec_to_ego_group_center) + 1E-6)) * 0.5

                vec_to_nearest_group_center = center_of_groups[nearest_group_inds[ii]] - self.world.crowds[ii].pos
                if np.linalg.norm(vec_to_nearest_group_center) < self.ped_radius * 4:
                    v_group -= (vec_to_nearest_group_center / (
                            np.linalg.norm(vec_to_nearest_group_center) + 1E-6)) * 0.3

                # apply inertia
                v_new = np.array(v) * (1 - self.world.inertia_coeff) \
                        + self.world.crowds[ii].vel * self.world.inertia_coeff \
                        + v_group

                p_new = self.world.crowds[ii].pos + v_new * dt

                self.world.set_ped_position(ii, p_new)
                self.world.set_ped_velocity(ii, v_new)

            self.step_robots(dt)

        super(GroupScenario, self).step(save)
