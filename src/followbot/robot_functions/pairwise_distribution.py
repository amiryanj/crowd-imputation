# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')


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
        self.x_edges = np.linspace(0, 8, 17)
        self.y_edges = np.linspace(-np.pi, np.pi, 37)
        self.x_bin_midpoints = self.x_edges[:-1] + np.diff(self.x_edges) / 2
        self.y_bin_midpoints = self.y_edges[:-1] + np.diff(self.y_edges) / 2
        self.polar_hist = np.zeros((len(self.x_edges)-1, len(self.y_edges)-1))
        self.hist_cum = None
        # plot
        self.fig = None
        self.axes = []

    def add_frame(self, agents_loc, agents_flow_class, dt):
        new_pair_distances = []
        for ii in range(len(agents_loc)):
            for jj in range(len(agents_loc)):
                if ii == jj: continue
                if agents_flow_class[ii].id == agents_flow_class[jj].id:
                    d_i_j = agents_loc[ii] - agents_loc[jj]
                    theta = np.arctan2(d_i_j[1], d_i_j[0])
                    if -np.pi / 2 < theta <= np.pi / 2:
                        new_pair_distances.append(d_i_j)

        if len(self.pairwise_distance_weights):
            self.pairwise_distance_weights *= (1 - self.weight_decay_factor) ** dt  # (t - self.last_t)
            self.pairwise_distances = np.concatenate((self.pairwise_distances, np.array(new_pair_distances)), axis=0)
        else:
            self.pairwise_distances = np.array(new_pair_distances)
        self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(new_pair_distances)))

        non_decayed_distances = self.pairwise_distance_weights > 0.2
        self.pairwise_distances = self.pairwise_distances[non_decayed_distances]
        self.pairwise_distance_weights = self.pairwise_distance_weights[non_decayed_distances]

    def update_histogram(self, smooth=True):
        self.polar_hist, _, _ = np.histogram2d(x=np.linalg.norm(self.pairwise_distances, axis=1),
                                               y=np.arctan2(self.pairwise_distances[:, 1],
                                                            self.pairwise_distances[:, 0]),
                                               bins=[self.x_edges, self.y_edges],
                                               weights=self.pairwise_distance_weights,
                                               density=True)

        self.hist_cum = np.cumsum(self.polar_hist.ravel())
        self.hist_cum /= self.hist_cum[-1]
        if smooth:
            self.polar_hist = gaussian_filter(self.polar_hist, sigma=1)

    def get_sample(self, n=1):
        r = np.random.random(n)
        value_bins = np.searchsorted(self.hist_cum, r)

        x_idx, y_idx = np.unravel_index(value_bins, (len(self.x_bin_midpoints), len(self.y_bin_midpoints)))
        random_from_cdf = np.column_stack((self.x_bin_midpoints[x_idx],
                                           self.y_bin_midpoints[y_idx]))

        return random_from_cdf

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
        polar_plot = self.axes[0, 0].pcolormesh(self.y_edges, self.x_edges, self.polar_hist, cmap='Blues')
        self.axes[0, 0].set_title("Polar Distance Dist.")

        self.axes[0, 1].clear()
        self.axes[0, 1].set_title("Bearing Angle Dist.")
        angle_axis = np.rad2deg(self.y_edges[1:] + self.y_edges[:-1]) * 0.5
        angle_plot = self.axes[0, 1].plot(angular_hist, angle_axis, 'r')
        self.axes[0, 1].fill_betweenx(angle_axis, 0, angular_hist)
        self.axes[0, 1].set_xlim([0, max(1, max(angular_hist) + 0.1)])
        self.axes[0, 1].set_ylim([-91, 91])
        self.axes[0, 1].set_yticks([-90, -45, 0, 45, 90])

        self.axes[1, 0].clear()
        pcf_plot = self.axes[1, 0].plot((self.x_edges[1:] + self.x_edges[:-1]) * 0.5, dist_hist)
        self.axes[1, 0].set_title("PCF")

        plt.pause(0.001)


