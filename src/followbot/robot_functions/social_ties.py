# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from matplotlib import gridspec
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


def polar2cartesian(r, t, grid, x, y, cval, order=3):
    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(Y, X)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    map2cart = map_coordinates(grid, np.array([new_ir, new_it]), order=order, cval=cval, mode='constant').reshape(new_r.shape).T
    return map2cart


# Todo : Add Strong/Absent PDFs
class SocialTiePDF:
    """
        This class will compute the probability distribution of social ties
        and will return a random sample on demand
    """
    def __init__(self, max_distance=6, radial_resolution=4, angular_resolution=36):
        self.strong_ties = []
        self.absent_ties = []

        # the pairwise distances in the pool are each assigned a weight that will get smaller as time goes
        # this permits the system to forget too old data
        self.pairwise_distance_weights = np.zeros(0, dtype=np.float64)
        self.weight_decay_factor = 0.25  # (fading memory control) per second

        # histogram setup
        self.rho_edges = np.linspace(0, max_distance, max_distance * radial_resolution + 1)
        self.theta_edges = np.linspace(-np.pi, np.pi, angular_resolution + 1)
        self.rho_bin_midpoints = self.rho_edges[:-1] + np.diff(self.rho_edges) / 2
        self.theta_bin_midpoints = self.theta_edges[:-1] + np.diff(self.theta_edges) / 2
        pdf_num_elements = len(self.rho_bin_midpoints) * len(self.theta_bin_midpoints)

        self.strong_ties_pdf_polar = np.zeros((len(self.rho_bin_midpoints), len(self.theta_bin_midpoints)))
        self.strong_ties_pdf_polar_cum = np.cumsum(np.ones(pdf_num_elements) / pdf_num_elements)

        self.absent_ties_pdf_polar = np.zeros((len(self.rho_bin_midpoints), len(self.theta_bin_midpoints)))
        self.absent_ties_pdf_polar_cum = np.cumsum(np.ones(pdf_num_elements) / pdf_num_elements)

        # cumulative distribution function of pairwise links in cartesian coord system
        self.social_ties_cartesian_pdf_aggregated = np.zeros((1, 1))  # will be calculated later

        # debug/visualization
        self.fig = None
        self.axes = []

    def load_prior_pdfs_from_file(self, fname):
        raise Exception('Todo: Implement here!')

    def add_frame(self, agents_loc, agents_vel, agents_flow_class, dt):
        new_strong_ties = []
        new_absent_ties = []
        for ii in range(len(agents_loc)):
            for jj in range(len(agents_loc)):
                if ii == jj: continue
                dp = agents_loc[jj] - agents_loc[ii]
                alpha = np.arctan2(dp[1], dp[0]) - np.arctan2(agents_vel[ii][1], agents_vel[ii][0])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                # alpha = np.arctan2(dp[1], dp[0])
                # alpha = np.arccos(np.dot(dp, agents_vel[ii]) / (norm(dp) * norm(agents_vel[ii]) + 1E-6))
                if agents_flow_class[ii].id == agents_flow_class[jj].id:
                    new_strong_ties.append(np.array([norm(dp), alpha]))
                else:
                    new_absent_ties.append(np.array([norm(dp), alpha]))

        if len(self.strong_ties):
            self.strong_ties = np.concatenate((self.strong_ties, np.array(new_strong_ties).reshape((-1, 2))), axis=0)
        else:
            self.strong_ties = np.array(new_strong_ties)

        if len(self.absent_ties):
            self.absent_ties = np.concatenate((self.absent_ties, np.array(new_absent_ties).reshape((-1, 2))), axis=0)
        else:
            self.absent_ties = np.array(new_absent_ties)

        # self.pairwise_distance_weights *= (1 - self.weight_decay_factor) ** dt  # (t - self.last_t)
        # self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(new_strong_ties)))
        # non_decayed_distances = self.pairwise_distance_weights > 0.2
        # self.pairwise_distance_weights = self.pairwise_distance_weights[non_decayed_distances]
        # self.strong_ties = self.strong_ties[non_decayed_distances]

    def add_links(self, links_polar):
        self.strong_ties = np.concatenate((np.array(self.strong_ties).reshape(-1, 2),
                                           np.array(links_polar).reshape((-1, 2))), axis=0)
        self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(links_polar)))

    def update_pdf(self, smooth=True):
        if len(self.strong_ties):
            self.strong_ties_pdf_polar, _, _ = np.histogram2d(x=self.strong_ties[:, 0],  # r
                                                              y=self.strong_ties[:, 1],  # theta (bearing angle)
                                                              bins=[self.rho_edges, self.theta_edges],
                                                              # weights=self.pairwise_distance_weights,
                                                              density=True)
            self.strong_ties_pdf_polar /= (np.sum(self.strong_ties_pdf_polar) + 1E-6)  # normalize
            self.strong_ties_pdf_polar_cum = np.cumsum(self.strong_ties_pdf_polar.ravel())

            self.absent_ties_pdf_polar, _, _ = np.histogram2d(x=self.absent_ties[:, 0],  # r
                                                              y=self.absent_ties[:, 1],  # theta (bearing angle)
                                                              bins=[self.rho_edges, self.theta_edges],
                                                              # weights=self.pairwise_distance_weights,
                                                              density=True)
            self.absent_ties_pdf_polar /= (np.sum(self.absent_ties_pdf_polar) + 1E-6)
            self.absent_ties_pdf_polar_cum = np.cumsum(self.absent_ties_pdf_polar.ravel())

            # normalizing the pdfs to have a mean=1: to get rid of padding when extrapolating to the whole map
            self.strong_ties_pdf_polar /= np.mean(self.strong_ties_pdf_polar)
            self.absent_ties_pdf_polar /= np.mean(self.absent_ties_pdf_polar)

        if smooth:
            # note: smoothing does not invalidate the distribution (sum=1)
            self.strong_ties_pdf_polar = gaussian_filter(self.strong_ties_pdf_polar, sigma=1)
            self.absent_ties_pdf_polar = gaussian_filter(self.absent_ties_pdf_polar, sigma=1)

        # min dist compliance: the probability of being an agent closer to (0.5m) to another agent should be zero!
        m = 2  # fixme
        self.strong_ties_pdf_polar[:m, :] = 0
        self.absent_ties_pdf_polar[:m, :] = 0

    def get_sample(self, n=1):
        rand_num = np.random.random(n)
        value_bins = np.searchsorted(self.strong_ties_pdf_polar_cum, rand_num)

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
            return self.strong_ties_pdf_polar[rho_idx][theta_idx] * self.strong_ties_pdf_polar.size * \
                   (1 + np.sqrt(rho))  # decrease the impact for longer links
        else:
            return 1  # very long links are not unlikely



    def _aggregate_to_virtual_pdf_(self, pos_i, vel_i, crowd_flow_map: MappedArray):
        # convert polar to cartesian by interpolation
        # the resolution should be similar to the one of mapped_arrays
        resol = crowd_flow_map.resolution
        xx = np.linspace(-self.rho_edges[-1], self.rho_edges[-1], self.rho_edges[-1] * 2 * resol)
        yy = np.linspace(-self.rho_edges[-1], self.rho_edges[-1], self.rho_edges[-1] * 2 * resol)

        # Rotate the tie_pdf toward the face of each agent
        orien_i = np.arctan2(vel_i[1], vel_i[0])

        # the rotation is done by simply shifting the polar pdf
        n_shifts = int(round(orien_i / np.diff(self.theta_edges[:2])[0]))
        if n_shifts != 0:
            # rotate the polar link distribution it toward agent face
            rotated_strong_ties_pdf_polar = np.roll(self.strong_ties_pdf_polar, -n_shifts, axis=1)
            rotated_absent_ties_pdf_polar = np.roll(self.absent_ties_pdf_polar, -n_shifts, axis=1)
        else:
            rotated_strong_ties_pdf_polar = self.strong_ties_pdf_polar
            rotated_absent_ties_pdf_polar = self.absent_ties_pdf_polar

        # convert polar to cartesian by interpolation
        strong_ties_pdf_cartesian = polar2cartesian(self.rho_edges, self.theta_edges,
                                                    rotated_strong_ties_pdf_polar, xx, yy, cval=1, order=2)
        absent_ties_pdf_cartesian = polar2cartesian(self.rho_edges, self.theta_edges,
                                                    rotated_absent_ties_pdf_polar, xx, yy, cval=1, order=2)
        # after conversion there are some negative values!
        strong_ties_pdf_cartesian = np.clip(strong_ties_pdf_cartesian, a_min=0, a_max=1000)
        absent_ties_pdf_cartesian = np.clip(absent_ties_pdf_cartesian, a_min=0, a_max=1000)

        agent_flow_class_id = crowd_flow_map.get(pos_i)
        # this agent can have a strong link to a (virtual) agent in the areas with the same flow_class
        # and can have absent link to a (virtual) agent in the areas with a different flow_class
        flow_mate_area = np.zeros(crowd_flow_map.data.shape, np.uint8)
        flow_mate_area[crowd_flow_map.data == agent_flow_class_id] = 1

        # to multiply the current pdf (which is centered on agent[i]):
        # we need to translate and (maybe) crop the pdf
        src_shape = strong_ties_pdf_cartesian.shape
        dst_shape = self.social_ties_cartesian_pdf_aggregated.data.shape
        agent_origin_u, agent_origin_v = crowd_flow_map.map(pos_i[0], pos_i[1])
        offset = [agent_origin_u - src_shape[0] // 2, agent_origin_v - src_shape[1] // 2]
        crop = [[max(0, -offset[0]), max(0, offset[0] + src_shape[0] - dst_shape[0])],
                [max(0, -offset[1]), max(0, offset[1] + src_shape[1] - dst_shape[1])]]

        self.social_ties_cartesian_pdf_aggregated.data[offset[0] + crop[0][0]:offset[0] - crop[0][1] + src_shape[0],
        offset[1] + crop[1][0]:offset[1] - crop[1][1] + src_shape[1]] \
            = self.social_ties_cartesian_pdf_aggregated.data[
              offset[0] + crop[0][0]:offset[0] - crop[0][1] + src_shape[0],
              offset[1] + crop[1][0]:offset[1] - crop[1][1] + src_shape[1]] \
              * (strong_ties_pdf_cartesian[crop[0][0]:src_shape[0] - crop[0][1], crop[1][0]:src_shape[1] - crop[1][1]]
                 * flow_mate_area[offset[0] + crop[0][0]:offset[0] - crop[0][1] + src_shape[0],
                   offset[1] + crop[1][0]:offset[1] - crop[1][1] + src_shape[1]]

                 + absent_ties_pdf_cartesian[crop[0][0]:src_shape[0] - crop[0][1], crop[1][0]:src_shape[1] - crop[1][1]]
                 * (1 - flow_mate_area[offset[0] + crop[0][0]:offset[0] - crop[0][1] + src_shape[0],
                        offset[1] + crop[1][0]:offset[1] - crop[1][1] + src_shape[1]]))




    def synthesis(self, in_agent_locs, in_agent_vels,
                  walkable_map: MappedArray = None,
                  blind_spot_map: MappedArray = None,
                  crowd_flow_map: MappedArray = None):
        synthetic_agents = []
        all_agents = np.concatenate([in_agent_locs, in_agent_vels], axis=1).tolist()

        # plt.subplot(121)
        # plt.imshow(self.p_link_polar.T, interpolation='nearest')
        # plt.subplot(122)
        # plt.imshow(p_link_cartesian, interpolation='nearest')
        # plt.pause()

        # accumulate all the
        self.social_ties_cartesian_pdf_aggregated = blind_spot_map.copy_constructor()
        self.social_ties_cartesian_pdf_aggregated.fill(1)
        self.social_ties_cartesian_pdf_aggregated.data *= (1-blind_spot_map.data)

        for ii, agent_i in enumerate(all_agents):
            pos_i = agent_i[0:2]
            vel_i = agent_i[2:4]
            self._aggregate_to_virtual_pdf_(pos_i, vel_i, crowd_flow_map)
            # if ii > 1: break

        n_tries = 5
        for i in range(n_tries):
            suggested_loc = self.social_ties_cartesian_pdf_aggregated.sample_random_pos()
            accept_suggested_loc = True
            if accept_suggested_loc:
                print(int(round(crowd_flow_map.get(suggested_loc))))
                suggested_vel = FlowClassifier().preset_flow_classes[int(round(crowd_flow_map.get(suggested_loc)))].velocity
                new_ped = Pedestrian(suggested_loc, suggested_vel, synthetic=True)
                new_ped.color = SKY_BLUE_COLOR
                synthetic_agents.append(new_ped)

                all_agents.append(suggested_loc)  # cuz: this agent should be considered when synthesising next agent
                self._aggregate_to_virtual_pdf_(suggested_loc, suggested_vel, crowd_flow_map)
        return synthetic_agents


    def plot(self, title=""):
        # Debug: plot polar heatmap
        angular_hist = np.sum(self.strong_ties_pdf_polar, axis=0)
        dist_hist = np.sum(self.strong_ties_pdf_polar, axis=1)

        if not len(self.axes):
            self.fig = plt.figure()
            grid_spec = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1], wspace=0.05, hspace=0.20)
            # self.fig, self.axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [5, 1]})
            self.axes = [[None, None], None]
            # self.axes[0, 0].remove()
            # self.axes[0, 0] = self.fig.add_subplot(221, projection="polar")
            self.axes[0][0] = plt.subplot(grid_spec[0, 0], projection='polar')
            self.axes[0][1] = plt.subplot(grid_spec[0, 1])
            self.axes[1] = plt.subplot(grid_spec[1, :])
            # self.axes[1, 0].remove()
            # self.axes[1, 1].remove()
            # self.axes[1, 0] = self.fig.add_subplot(212)

        self.axes[0][0].clear()
        polar_plot = self.axes[0][0].pcolormesh(self.theta_edges, self.rho_edges, self.strong_ties_pdf_polar, vmin=0,
                                                cmap='YlGnBu')
        self.axes[0][0].set_ylabel("Polar Histogram of Links", labelpad=40)
        if len(title):
            self.axes[0][0].set_title(title, pad=20)

        self.axes[0][1].clear()
        self.axes[0][1].set_title("Angles pdf", fontsize=10)
        angle_plot = self.axes[0][1].plot(angular_hist, np.rad2deg(self.theta_bin_midpoints), 'r')
        self.axes[0][1].fill_betweenx(np.rad2deg(self.theta_bin_midpoints), 0, angular_hist)
        self.axes[0][1].set_xlim([0, max(angular_hist) * 1.1])
        self.axes[0][1].set_ylim([-151, 151])
        self.axes[0][1].set_yticks([-135, -90, -45, 0, 45, 90, 135])

        self.axes[1].clear()
        # pcf_plot = self.axes[1].plot(self.rho_bin_midpoints, dist_hist, 'r')
        # self.axes[1].fill_between(self.rho_bin_midpoints, 0, dist_hist)
        self.axes[1].bar(self.rho_bin_midpoints, dist_hist, width=np.diff(self.rho_edges)[0]*0.8)

        self.axes[1].set_title("Length pdf  ", loc='right', fontsize=10, pad=-14)
        self.axes[1].set_xlabel('$\it{m}$', labelpad=-5)
        # self.axes[1].grid()

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
