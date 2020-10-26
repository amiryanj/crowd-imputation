# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy.linalg import norm
from math import sin, cos, sqrt
from sklearn.metrics.pairwise import euclidean_distances


def unit_vec(v):
    return v / ((v ** 2).sum() ** 0.5 + 1E-6)


# TODO
class GroupingBehavior:
    def __init__(self, groups):
        """:param groups: list of list of agent_ids in the same group"""
        self.grouped_agents = groups

        # parameters
        self.group_attraction_speed = 0.4
        self.group_repulsion_speed = 0.3
        self.max_distance_from_group = 1.2
        self.min_distance_to_other_group = 2

    def step(self, crowds):
        COM_of_groups = {}  # Center-of-mass point for each group of agents
        for group_k in self.grouped_agents:
            COM_of_group_k = np.zeros(2)
            for jj in group_k:
                COM_of_group_k += crowds[jj].pos / len(group_k)
            for jj in group_k:
                COM_of_groups[jj] = COM_of_group_k

        # find index of closest group to each agent, excluding their own group
        DD = euclidean_distances(list(COM_of_groups.values()))  # distance matrix of COMs
        DD[DD < 1E-6] = np.Infinity                 # remove self-distances
        nearest_group_inds = np.argmin(DD, axis=0)

        collective_motions = np.zeros((len(crowds), 2), float)
        # ==========================
        for ii in range(len(crowds)):
            p_i = crowds[ii].pos
            v_i = crowds[ii].vel

            # attraction to the group
            vec_to_ego_group_COM = COM_of_groups[ii] - crowds[ii].pos
            if norm(vec_to_ego_group_COM) - crowds[ii].radius > self.max_distance_from_group:
                collective_motions[ii] = unit_vec(vec_to_ego_group_COM) * self.group_attraction_speed

            # repulsion from other groups to the group
            vec_to_nearest_group_COM = COM_of_groups[nearest_group_inds[ii]] - p_i
            if norm(vec_to_nearest_group_COM) - crowds[ii].radius < self.min_distance_to_other_group:
                collective_motions[ii] -= unit_vec(vec_to_nearest_group_COM) * self.group_repulsion_speed
        return collective_motions

