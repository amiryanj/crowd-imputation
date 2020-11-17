# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy.linalg import norm
from scipy.stats import rv_histogram
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from followbot.crowdsim.pedestrian import Pedestrian
from followbot.gui.visualizer import SKY_BLUE_COLOR
from followbot.robot_functions.flow_classifier import FlowClassifier
from followbot.util.mapped_array import MappedArray


def polar2cartesian(r, t, grid, x, y, order=3):
    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(Y, X)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_ir, new_it]), order=order, mode='wrap').reshape(new_r.shape).T


# TODO: This class will compute the distribution of pair-wise distances for each flow instance
#  and will return a random sample on demand
class PairwiseDistribution:
    def __init__(self):
        self.pairwise_distances = []

        # the pairwise distances in the pool are each assigned a weight that will get smaller as time goes
        # this permits the system to forget too old data
        self.pairwise_distance_weights = np.zeros(0, dtype=np.float64)
        self.weight_decay_factor = 0.25  # (fading memory control) per second

        # histogram
        max_distance = 4  # meter
        self.rho_edges = np.linspace(0, max_distance, 17)
        self.theta_edges = np.linspace(-np.pi, np.pi, 37)
        self.rho_bin_midpoints = self.rho_edges[:-1] + np.diff(self.rho_edges) / 2
        self.theta_bin_midpoints = self.theta_edges[:-1] + np.diff(self.theta_edges) / 2
        self.polar_link_pdf = np.zeros((len(self.rho_edges) - 1, len(self.theta_edges) - 1))
        self.hist_cum = np.cumsum(np.ones(self.polar_link_pdf.size) / self.polar_link_pdf.size)

        # cumulative distribution function of pairwise links in cartesian coord system
        self.cartesian_link_pdf_total = MappedArray(0, 10, 0, 10, 1)

        # plot
        self.fig = None
        self.axes = []

    def add_frame(self, agents_loc, agents_vel, agents_flow_class, dt):
        new_pairs = []
        for ii in range(len(agents_loc)):
            for jj in range(len(agents_loc)):
                if ii == jj: continue
                if agents_flow_class[ii].id == agents_flow_class[jj].id:
                    dp = agents_loc[jj] - agents_loc[ii]
                    # alpha = np.arctan2(dp[1], dp[0])
                    # alpha = np.arccos(np.dot(dp, agents_vel[ii]) / (norm(dp) * norm(agents_vel[ii]) + 1E-6))
                    alpha = np.arctan2(dp[1], dp[0]) - np.arctan2(agents_vel[ii][1], agents_vel[ii][0])
                    if alpha > np.pi: alpha -= 2 * np.pi

                    # if -np.pi / 2 < alpha <= np.pi / 2:  # FixMe
                    new_pairs.append(np.array([norm(dp), alpha]))

        if len(self.pairwise_distance_weights):
            self.pairwise_distance_weights *= (1 - self.weight_decay_factor) ** dt  # (t - self.last_t)
            self.pairwise_distances = np.concatenate((self.pairwise_distances, np.array(new_pairs).reshape((-1, 2))), axis=0)
        else:
            self.pairwise_distances = np.array(new_pairs)
        self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(new_pairs)))

        non_decayed_distances = self.pairwise_distance_weights > 0.2
        self.pairwise_distances = self.pairwise_distances[non_decayed_distances]
        self.pairwise_distance_weights = self.pairwise_distance_weights[non_decayed_distances]

    def update_histogram(self, smooth=True):
        if not len(self.pairwise_distances):
            return  # no pairwise data is available

        self.polar_link_pdf, _, _ = np.histogram2d(x=self.pairwise_distances[:, 0],  # r
                                                   y=self.pairwise_distances[:, 1],  # theta (bearing angle)
                                                   bins=[self.rho_edges, self.theta_edges],
                                                   weights=self.pairwise_distance_weights,
                                                   density=True)

        self.polar_link_pdf /= (np.sum(self.polar_link_pdf) + 1E-6)
        self.hist_cum = np.cumsum(self.polar_link_pdf.ravel())
        # self.hist_cum /= self.hist_cum[-1]
        if smooth:
            self.polar_link_pdf = gaussian_filter(self.polar_link_pdf, sigma=1)  # does not invalidate the distribution

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
            return self.polar_link_pdf[rho_idx][theta_idx] * self.polar_link_pdf.size * \
                   (1 + np.sqrt(rho))  # decrease the impact for longer links
        else:
            return 1  # very long links are not unlikely

    def synthesis(self, in_agent_locs, in_agent_vels,
                  walkable_map: MappedArray = None,
                  blind_spot_map: MappedArray = None,
                  crowd_flow_map: MappedArray = None):
        agent_states = np.concatenate([in_agent_locs, in_agent_vels], axis=1).tolist()

        # convert polar to cartesian using interpolation
        # the resolution should be similar to the one of mapped_arrays
        resol = blind_spot_map.resolution
        xx = np.linspace(-self.rho_edges[-1], self.rho_edges[-1], self.rho_edges[-1] * 2 * resol)
        yy = np.linspace(-self.rho_edges[-1], self.rho_edges[-1], self.rho_edges[-1] * 2 * resol)

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(self.p_link_polar.T, interpolation='nearest')
        #
        # plt.subplot(122)
        # plt.imshow(p_link_cartesian, interpolation='nearest')
        # plt.show()
        # exit(1)

        # accumulate all the
        self.cartesian_link_pdf_total = blind_spot_map.copy_constructor()
        self.cartesian_link_pdf_total.fill(0)
        # plt.figure()

        for ii, agent_i in enumerate(agent_states):
            pos_i = agent_i[0:2]
            vel_i = agent_i[2:4]
            orien_i = np.arctan2(vel_i[1], vel_i[0])
            n_shifts = int(round(orien_i / np.diff(self.theta_edges[:2])[0]))
            if n_shifts != 0:
                # rotate the polar link distribution it toward agent face
                rotated_polar_link_prob = np.roll(self.polar_link_pdf, -n_shifts, axis=1)
            else:
                rotated_polar_link_prob = self.polar_link_pdf
            p_link_cartesian = polar2cartesian(self.rho_edges, self.theta_edges, rotated_polar_link_prob, xx, yy, order=2)

            src_shape = p_link_cartesian.shape
            dst_shape = self.cartesian_link_pdf_total.data.shape
            try:
                ag_origin_u, ag_origin_v = blind_spot_map.map(pos_i[0], pos_i[1])
                offset = [ag_origin_u - src_shape[0] // 2, ag_origin_v - src_shape[1] // 2]
                crop = [[max(0, -offset[0]), max(0, offset[0]+src_shape[0] - dst_shape[0])],
                        [max(0, -offset[1]), max(0, offset[1]+src_shape[1] - dst_shape[1])]]
                self.cartesian_link_pdf_total.data[offset[0] + crop[0][0]:offset[0] - crop[0][1] + src_shape[0],
                offset[1] + crop[1][0]:offset[1] - crop[1][1] + src_shape[1]] \
                    = self.cartesian_link_pdf_total.data[offset[0] + crop[0][0]:offset[0] - crop[0][1] + src_shape[0],
                      offset[1] + crop[1][0]:offset[1] - crop[1][1] + src_shape[1]] \
                      + p_link_cartesian[crop[0][0]:src_shape[0] - crop[0][1],
                        crop[1][0]:src_shape[1] - crop[1][1]]
            except:
                print("Bug in link probability calculation")

            # if ii > 1: break

        # plt.pause(0.1)

        synthetic_agents = []
        n_tries = 12
        for i in range(n_tries):
            suggested_loc = self.cartesian_link_pdf_total.sample_random_pos()

        # for i in range(n_tries):
        #     random_anchor = agent_states[np.random.randint(0, len(agent_states))]
        #     random_displacement = self.get_sample(1).squeeze()
        #     suggested_loc = random_anchor[:2] + random_displacement

            #  check if the point falls inside the walkable_map area
            # if walkable_map and not walkable_map.get(suggested_loc):
            #     continue

            #  check if it falls inside the blind spot area
            if blind_spot_map and not blind_spot_map.get(suggested_loc):
                continue

            #  and also check if it falls in the same flow class area
            # if crowd_flow_map.get(random_anchor) != crowd_flow_map.get(suggested_loc):
            #     continue

            accept_suggested_loc = True
            for agent_i in agent_states:
                link_i = np.array(suggested_loc) - np.array(agent_i[:2])
                if np.linalg.norm(link_i) < 0.5:  # this violate the min distance between 2 agents
                    accept_suggested_loc = False  # REJECT
                    break
            #     link_i_likelihood = self.likelihood(link_i)  # multiplied by the size of array
            #     if link_i_likelihood < 0.5:
            #         accept_suggested_loc = False  # REJECT
            #         break
            if accept_suggested_loc:
                print(int(round(crowd_flow_map.get(suggested_loc))))
                suggested_vel = FlowClassifier().preset_flow_classes[int(round(crowd_flow_map.get(suggested_loc)))].velocity
                new_ped = Pedestrian(suggested_loc, suggested_vel, synthetic=True)
                new_ped.color = SKY_BLUE_COLOR
                synthetic_agents.append(new_ped)

                agent_states.append(suggested_loc)  # cuz: this agent should be considered when synthesising next agent
        return synthetic_agents

    def plot(self):
        # Debug: plot polar heatmap
        angular_hist = np.sum(self.polar_link_pdf, axis=0)
        dist_hist = np.sum(self.polar_link_pdf, axis=1)

        if not len(self.axes):
            self.fig, self.axes = plt.subplots(2, 2)
            self.axes[0, 0].remove()
            self.axes[0, 0] = self.fig.add_subplot(221, projection="polar")
            self.axes[1, 0].remove()
            self.axes[1, 1].remove()
            self.axes[1, 0] = self.fig.add_subplot(212)

        self.axes[0, 0].clear()
        polar_plot = self.axes[0, 0].pcolormesh(self.theta_edges, self.rho_edges, self.polar_link_pdf, cmap='Blues')
        self.axes[0, 0].set_ylabel("Dist. of Pairwise links", labelpad=40)

        self.axes[0, 1].clear()
        self.axes[0, 1].set_title("Link Angles")
        angle_axis = np.rad2deg(self.theta_edges[1:] + self.theta_edges[:-1]) * 0.5
        angle_plot = self.axes[0, 1].plot(angular_hist, angle_axis, 'r')
        self.axes[0, 1].fill_betweenx(angle_axis, 0, angular_hist)
        self.axes[0, 1].set_xlim([0, max(angular_hist)+ 0.1])
        self.axes[0, 1].set_ylim([-91, 91])
        self.axes[0, 1].set_yticks([-90, -45, 0, 45, 90])

        self.axes[1, 0].clear()
        pcf_plot = self.axes[1, 0].plot((self.rho_edges[1:] + self.rho_edges[:-1]) * 0.5, dist_hist)
        self.axes[1, 0].set_ylabel("PCF")
        self.axes[1, 0].grid()

        plt.pause(0.001)


if __name__ == "__main__":
    pass
    # test polar2cartesian()
    # Define original polar grid
    nr_ = 10
    nt_ = 10

    r_ = np.linspace(1, 100, nr_)
    t_ = np.linspace(0., np.pi, nt_)
    z_ = np.random.random((nr_, nt_))

    # Define new cartesian grid

    nx_ = 100
    ny_ = 200

    x_ = np.linspace(0., 100., nx_)
    y_ = np.linspace(-100., 100., ny_)

    # Interpolate polar grid to cartesian grid (nearest neighbor)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.imshow(polar2cartesian(r_, t_, z_, x_, y_, order=0), interpolation='nearest')

    # Interpolate polar grid to cartesian grid (cubic spline)

    ax = fig.add_subplot(212)
    ax.imshow(polar2cartesian(r_, t_, z_, x_, y_, order=3), interpolation='nearest')
    plt.show()
