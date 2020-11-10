# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from scipy.stats import rv_histogram
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt

from followbot.util.mapped_array import MappedArray


# TODO: This class will compute the distribution of pair-wise distances for each flow instance
#  and will return a random sample on demand
class PairwiseDistribution:
    def __init__(self):
        self.pairwise_distances = []

        # the pairwise distances in the pool are each assigned a weight that will get smaller as time goes
        # this permits the system to forget too old data
        self.pairwise_distance_weights = np.zeros((0), dtype=np.float64)
        self.weight_decay_factor = 0.4  # per second

        # histogram
        self.rho_edges = np.linspace(0, 8, 17)
        self.theta_edges = np.linspace(-np.pi, np.pi, 37)
        self.rho_bin_midpoints = self.rho_edges[:-1] + np.diff(self.rho_edges) / 2
        self.theta_bin_midpoints = self.theta_edges[:-1] + np.diff(self.theta_edges) / 2
        self.polar_hist = np.zeros((len(self.rho_edges) - 1, len(self.theta_edges) - 1))
        self.hist_cum = np.cumsum(np.ones(self.polar_hist.size) / self.polar_hist.size)

        # plot
        self.fig = None
        self.axes = []

    def add_frame(self, agents_loc, agents_flow_class, dt):
        new_pairs = []
        for ii in range(len(agents_loc)):
            for jj in range(len(agents_loc)):
                if ii == jj: continue
                if agents_flow_class[ii].id == agents_flow_class[jj].id:
                    d_i_j = agents_loc[ii] - agents_loc[jj]
                    theta = np.arctan2(d_i_j[1], d_i_j[0])
                    if -np.pi / 2 < theta <= np.pi / 2:
                        new_pairs.append(d_i_j)

        if len(self.pairwise_distance_weights):
            self.pairwise_distance_weights *= (1 - self.weight_decay_factor) ** dt  # (t - self.last_t)
            self.pairwise_distances = np.concatenate((self.pairwise_distances, np.array(new_pairs)), axis=0)
        else:
            self.pairwise_distances = np.array(new_pairs)
        self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(new_pairs)))

        non_decayed_distances = self.pairwise_distance_weights > 0.2
        self.pairwise_distances = self.pairwise_distances[non_decayed_distances]
        self.pairwise_distance_weights = self.pairwise_distance_weights[non_decayed_distances]

    def update_histogram(self, smooth=True):
        if not len(self.pairwise_distances):
            return  # no pairwise data is available

        self.polar_hist, _, _ = np.histogram2d(x=np.linalg.norm(self.pairwise_distances, axis=1),
                                               y=np.arctan2(self.pairwise_distances[:, 1],
                                                            self.pairwise_distances[:, 0]),
                                               bins=[self.rho_edges, self.theta_edges],
                                               weights=self.pairwise_distance_weights,
                                               density=True)

        self.polar_hist /= (np.sum(self.polar_hist) + 1E-6)
        self.hist_cum = np.cumsum(self.polar_hist.ravel())
        # self.hist_cum /= self.hist_cum[-1]
        if smooth:
            self.polar_hist = gaussian_filter(self.polar_hist, sigma=1)  # does not invalidate the distribution

    def get_sample(self, n=1):
        rand_num = np.random.random(n)
        value_bins = np.searchsorted(self.hist_cum, rand_num)

        rho_idx, theta_idx = np.unravel_index(value_bins, (len(self.rho_bin_midpoints), len(self.theta_bin_midpoints)))
        random_from_cdf = np.column_stack((self.rho_bin_midpoints[rho_idx],
                                           self.theta_bin_midpoints[theta_idx]))

        return random_from_cdf

    def likelihood(self, link):
        if link[0] < 0:
            link = -link.copy()
        rho = np.linalg.norm(link)
        theta = np.arctan2(link[1], link[0])
        rho_idx = np.searchsorted(self.rho_edges, rho)
        theta_idx = np.searchsorted(self.theta_edges, theta)
        if rho_idx < len(self.rho_edges) - 1:
            return self.polar_hist[rho_idx][theta_idx] * self.polar_hist.size * \
                   (1 + np.sqrt(rho))  # decrease the impact for longer links
        else:
            return 1  # very long links are not unlikely

    def synthesis(self, det_agent_locs,
                  walkable_map: MappedArray = None,
                  blind_spot_map: MappedArray = None,
                  crowd_flow_map: MappedArray = None):
        all_agents = list(det_agent_locs.copy())
        synthetic_agents = []
        for i in range(50):
            random_anchor = all_agents[np.random.randint(0, len(all_agents))]
            random_displacement = self.get_sample(1).squeeze()
            suggested_loc = random_anchor + random_displacement

            #  check if the point falls inside the walkable_map area
            if walkable_map and not walkable_map.get(suggested_loc):
                continue

            #  check if it falls inside the blind spot area
            if blind_spot_map and not blind_spot_map.get(suggested_loc):
                continue

            #  and also check if it falls in the same flow class area
            if crowd_flow_map.get(random_anchor) != crowd_flow_map.get(suggested_loc):
                continue

            accept_suggested_loc = True
            for agent_i in all_agents:
                link_i = suggested_loc - agent_i
                link_i_likelihood = self.likelihood(link_i)  # multiplied by the size of array
                if link_i_likelihood < 0.5:
                    accept_suggested_loc = False  # REJECT
                    break
            if accept_suggested_loc:
                all_agents.append(suggested_loc)
                synthetic_agents.append(suggested_loc)
        return synthetic_agents

    def plot(self):
        # Debug: plot polar heatmap
        angular_hist = np.sum(self.polar_hist, axis=0)
        dist_hist = np.sum(self.polar_hist, axis=1)

        if not len(self.axes):
            self.fig, self.axes = plt.subplots(2, 2)
            self.axes[0, 0].remove()
            self.axes[0, 0] = self.fig.add_subplot(221, projection="polar")
            self.axes[1, 0].remove()
            self.axes[1, 1].remove()
            self.axes[1, 0] = self.fig.add_subplot(212)

        self.axes[0, 0].clear()
        polar_plot = self.axes[0, 0].pcolormesh(self.theta_edges, self.rho_edges, self.polar_hist, cmap='Blues')
        self.axes[0, 0].set_title("Polar Distance Dist.")

        self.axes[0, 1].clear()
        self.axes[0, 1].set_title("Bearing Angle Dist.")
        angle_axis = np.rad2deg(self.theta_edges[1:] + self.theta_edges[:-1]) * 0.5
        angle_plot = self.axes[0, 1].plot(angular_hist, angle_axis, 'r')
        self.axes[0, 1].fill_betweenx(angle_axis, 0, angular_hist)
        self.axes[0, 1].set_xlim([0, max(1, max(angular_hist) + 0.1)])
        self.axes[0, 1].set_ylim([-91, 91])
        self.axes[0, 1].set_yticks([-90, -45, 0, 45, 90])

        self.axes[1, 0].clear()
        pcf_plot = self.axes[1, 0].plot((self.rho_edges[1:] + self.rho_edges[:-1]) * 0.5, dist_hist)
        self.axes[1, 0].set_title("PCF")

        plt.pause(0.001)
