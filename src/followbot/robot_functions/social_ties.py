# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import os
import cv2
import numpy as np
from matplotlib import gridspec
from numpy.linalg import norm
from scipy.stats import rv_histogram
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from skimage.transform import resize

from followbot.crowdsim.pedestrian import Pedestrian
from followbot.gui.visualizer import SKY_BLUE_COLOR
from followbot.robot_functions.crowd_clustering import FlowClassifier
from followbot.util.mapped_array import MappedArray


def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1


def polar2cartesian(r, t, grid, x, y, cval, order=3):
    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X * X + Y * Y)
    new_t = np.arctan2(Y, X)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value="extrapolate")
    it = interp1d(t, np.arange(len(t)), fill_value="extrapolate")

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r) - 1
    new_ir[new_r.ravel() < r.min()] = 0

    map2cart = map_coordinates(grid, np.array([new_ir, new_it]), order=order, cval=cval, mode='nearest').reshape(
        new_r.shape).T
    return map2cart


# Todo : Add Strong/Absent PDFs
class SocialTiePDF:
    """
        This class will compute the probability distribution of social ties
        and will return a random sample on demand
    """

    def __init__(self, max_distance=4, radial_resolution=4, angular_resolution=36):
        self.num_prior_strong_ties = 0
        self.num_prior_absent_ties = 0
        self.padding_prob_val = 1.0
        self.strong_ties = []
        self.absent_ties = []

        self.radial_resolution = radial_resolution
        self.angular_resolution = angular_resolution

        # the pairwise distances in the pool are each assigned a weight that will get smaller as time goes
        # this permits the system to forget too old data
        self.pairwise_distance_weights = np.zeros(0, dtype=np.float64)
        self.weight_decay_factor = 0.25  # (fading memory control) per second

        # histogram setup
        self.rho_edges = np.linspace(0, max_distance, max_distance * radial_resolution + 1)
        self.theta_edges = np.linspace(-np.pi, np.pi, angular_resolution + 1)
        self.rho_bin_midpoints = self.rho_edges[:-1] + np.diff(self.rho_edges) / 2
        self.theta_bin_midpoints = self.theta_edges[:-1] + np.diff(self.theta_edges) / 2

        self.strong_ties_pdf_polar = np.zeros((len(self.rho_bin_midpoints), len(self.theta_bin_midpoints)))
        self.absent_ties_pdf_polar = np.zeros((len(self.rho_bin_midpoints), len(self.theta_bin_midpoints)))

        # cumulative distribution function of pairwise links in cartesian coord system
        self.social_ties_cartesian_pdf_aggregated = np.zeros((1, 1))  # will be calculated later

        # debug/visualization
        self.fig = None
        self.axes = []

    def load_prior_pdfs_from_file(self, fname):
        raise Exception('Todo: Implement here!')

    # No history: at the moment just working for Hermes
    def classify_ties(self, agents_loc, agents_vel):
        n = len(agents_loc)
        strong_ties = []
        absent_ties = []
        agent_flow_ids = np.zeros(n, dtype=np.uint8)
        if n == 0: return [], [], agent_flow_ids

        # simple classification of flows
        agent_flow_ids[agents_vel[:, 0] > 0.1] = 1
        agent_flow_ids[agents_vel[:, 0] < -0.1] = 2
        if n == 1: return [], [], agent_flow_ids

        oriens = np.arctan2(agents_vel[:, 1], agents_vel[:, 0])
        tiled_locs = np.tile(agents_loc, (n, 1, 1))
        D = tiled_locs - tiled_locs.transpose((1, 0, 2))
        tie_angles = np.arctan2(D[:, :, 1], D[:, :, 0])
        tie_lengths = np.linalg.norm(D, axis=2)

        max_tie_length = 4
        for ii in range(n):
            if norm(agents_vel[ii]) < 0.1:
                continue
            for jj in range(n):
                if ii == jj or tie_lengths[ii, jj] > max_tie_length:
                    continue
                rotated_tie_angle = tie_angles[ii, jj] - oriens[ii]
                rotated_tie_angle = (rotated_tie_angle + np.pi) % (2 * np.pi) - np.pi
                tie_polar = [tie_lengths[ii, jj], rotated_tie_angle]
                if agent_flow_ids[ii] == agent_flow_ids[jj]:
                    strong_ties.append(tie_polar)
                else:
                    absent_ties.append(tie_polar)

        return strong_ties, absent_ties, agent_flow_ids

    def add_frame(self, agents_loc, agents_vel, agents_flow_class, dt):
        new_strong_ties = []
        new_absent_ties = []
        for ii in range(len(agents_loc)):
            for jj in range(len(agents_loc)):
                if ii == jj: continue
                dp = agents_loc[jj] - agents_loc[ii]
                # alpha = np.arctan2(dp[1], dp[0])
                # alpha = np.arccos(np.dot(dp, agents_vel[ii]) / (norm(dp) * norm(agents_vel[ii]) + 1E-6))
                alpha = np.arctan2(dp[1], dp[0]) - np.arctan2(agents_vel[ii][1], agents_vel[ii][0])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                if agents_flow_class[ii] == agents_flow_class[jj]:
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

    def add_strong_ties(self, ties_polar):
        self.strong_ties = np.concatenate((np.array(self.strong_ties).reshape(-1, 2),
                                           np.array(ties_polar).reshape((-1, 2))), axis=0)
        # self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(links_polar)))

    def add_absent_ties(self, ties_polar):
        self.absent_ties = np.concatenate((np.array(self.absent_ties).reshape(-1, 2),
                                           np.array(ties_polar).reshape((-1, 2))), axis=0)
        # self.pairwise_distance_weights = np.append(self.pairwise_distance_weights, np.ones(len(links_polar)))

    def smooth_pdf(self, sigma=1):
        self.strong_ties_pdf_polar = gaussian_filter(self.strong_ties_pdf_polar, sigma=sigma)
        self.absent_ties_pdf_polar = gaussian_filter(self.absent_ties_pdf_polar, sigma=sigma)

    def update_pdf(self):
        if len(self.strong_ties):
            new_strong_ties_pdf_polar, _, _ = np.histogram2d(x=self.strong_ties[:, 0],  # r
                                                             y=self.strong_ties[:, 1],  # theta (bearing angle)
                                                             bins=[self.rho_edges, self.theta_edges],
                                                             # weights=self.pairwise_distance_weights,
                                                             density=True)

            # clear previous ties (that are already counted in the histogram)
            # and keep the total number of prior ties
            self.strong_ties_pdf_polar = self.strong_ties_pdf_polar * self.num_prior_strong_ties + \
                                         new_strong_ties_pdf_polar * len(self.strong_ties)

        if len(self.absent_ties):
            new_absent_ties_pdf_polar, _, _ = np.histogram2d(x=self.absent_ties[:, 0],  # r
                                                             y=self.absent_ties[:, 1],  # theta (bearing angle)
                                                             bins=[self.rho_edges, self.theta_edges],
                                                             # weights=self.pairwise_distance_weights,
                                                             density=True)

            # clear previous ties (that are already counted in the histogram)
            # and keep the total number of prior ties
            self.absent_ties_pdf_polar = self.absent_ties_pdf_polar * self.num_prior_absent_ties + \
                                         new_absent_ties_pdf_polar * len(self.absent_ties)

        self.num_prior_strong_ties += len(self.strong_ties)
        self.num_prior_absent_ties += len(self.absent_ties)
        self.strong_ties = []
        self.absent_ties = []
        # normalize
        self.strong_ties_pdf_polar /= (np.sum(self.strong_ties_pdf_polar) + 1E-6)
        self.absent_ties_pdf_polar /= (np.sum(self.absent_ties_pdf_polar) + 1E-6)
        # normalizing the pdfs to have a mean=1: to get rid of padding when extrapolating to the whole map
        self.strong_ties_pdf_polar /= (np.mean(self.strong_ties_pdf_polar) + 1E-6)
        self.absent_ties_pdf_polar /= (np.mean(self.absent_ties_pdf_polar) + 1E-6)

        # min dist compliance: the probability of being an agent closer to (0.5m) to another agent should be zero!
        m = 2  # fixme: depends on the resolution
        self.strong_ties_pdf_polar[:m, :] = 0
        self.absent_ties_pdf_polar[:m, :] = 0

    def _aggregate_to_virtual_pdf_(self, pos_i, vel_i, crowd_flow_map: MappedArray):
        # convert polar to cartesian by interpolation
        # the resolution should be similar to the one of mapped_arrays
        resol = crowd_flow_map.resolution
        xx = np.linspace(-self.rho_edges[-1], self.rho_edges[-1], int(round(self.rho_edges[-1] * 2 * resol)))
        yy = np.linspace(-self.rho_edges[-1], self.rho_edges[-1], int(round(self.rho_edges[-1] * 2 * resol)))

        # Rotate the tie_pdf toward the face of each agent
        orien_i = np.arctan2(vel_i[1], vel_i[0])

        # the rotation is done by simply shifting the polar pdf
        n_shifts = int(round(orien_i / np.diff(self.theta_edges[:2])[0]))
        if n_shifts != 0:
            # rotate the polar tie distribution it toward agent face
            rotated_strong_ties_pdf_polar = np.roll(self.strong_ties_pdf_polar, n_shifts, axis=1)
            rotated_absent_ties_pdf_polar = np.roll(self.absent_ties_pdf_polar, n_shifts, axis=1)
        else:
            rotated_strong_ties_pdf_polar = self.strong_ties_pdf_polar.copy()
            rotated_absent_ties_pdf_polar = self.absent_ties_pdf_polar.copy()

        # print("n_shifts = ", n_shifts)
        # =======================================
        # if n_shifts == 2:
        #     plt.figure()
        #     ax_1 = plt.subplot(2, 2, 1)
        #     ax_2 = plt.subplot(2, 2, 2)
        #     ax_1.imshow(self.strong_ties_pdf_polar)
        #     ax_2.imshow(rotated_strong_ties_pdf_polar)
        #     # ax_1 = plt.subplot(2, 2, 1, projection='polar')
        #     # ax_2 = plt.subplot(2, 2, 2, projection='polar')
        #     # ax_1.pcolormesh(self.theta_edges, self.rho_edges, self.strong_ties_pdf_polar, vmin=0, cmap='YlGnBu')
        #     # ax_2.pcolormesh(self.theta_edges, self.rho_edges, rotated_strong_ties_pdf_polar, vmin=0, cmap='YlGnBu')
        #
        #     cartesian_1 = polar2cartesian(self.rho_bin_midpoints, self.theta_bin_midpoints,
        #                                                  self.strong_ties_pdf_polar, xx, yy, cval=1, order=2)
        #     cartesian_2 = polar2cartesian(self.rho_bin_midpoints, self.theta_bin_midpoints,
        #                                                 rotated_strong_ties_pdf_polar, xx, yy, cval=1, order=2)
        #     ax_3 = plt.subplot(2, 2, 3)
        #     ax_3.imshow(np.flipud(cartesian_1.T))
        #     ax_4 = plt.subplot(2, 2, 4)
        #     ax_4.imshow(np.flipud(cartesian_2.T))
        #     plt.show()
        #  # =======================================


        # rotated_strong_ties_pdf_polar /= np.nanmedian(rotated_strong_ties_pdf_polar)
        # rotated_absent_ties_pdf_polar /= np.nanmedian(rotated_absent_ties_pdf_polar)

        # convert polar to cartesian by interpolation
        strong_ties_pdf_cartesian = polar2cartesian(self.rho_edges, self.theta_edges,
                                                    rotated_strong_ties_pdf_polar, xx, yy, cval=1, order=2)
        absent_ties_pdf_cartesian = polar2cartesian(self.rho_edges, self.theta_edges,
                                                    rotated_absent_ties_pdf_polar, xx, yy, cval=1, order=2)
        # after conversion there are some negative values => clip them to zero!
        strong_ties_pdf_cartesian = np.clip(strong_ties_pdf_cartesian, a_min=0, a_max=1000)
        absent_ties_pdf_cartesian = np.clip(absent_ties_pdf_cartesian, a_min=0, a_max=1000)

        strong_ties_pdf_cartesian = strong_ties_pdf_cartesian ** (1/4)
        absent_ties_pdf_cartesian = absent_ties_pdf_cartesian ** (1/4)

        agent_flow_class_id = crowd_flow_map.get(pos_i)
        # this agent can have a strong tie to a (virtual) agent in the areas with the same flow_class
        # and can have absent tie to a (virtual) agent in the areas with a different flow_class
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

    def project(self, in_agent_locs, in_agent_vels,
                walkable_map: MappedArray = None,
                blind_spot_map: MappedArray = None,
                crowd_flow_map: MappedArray = None):
        projected_agents = []
        all_agents = np.concatenate([in_agent_locs, in_agent_vels], axis=1).tolist()

        # accumulate all the
        self.social_ties_cartesian_pdf_aggregated = blind_spot_map.copy_constructor()
        self.social_ties_cartesian_pdf_aggregated.fill(self.padding_prob_val)
        self.social_ties_cartesian_pdf_aggregated.data *= blind_spot_map.data

        # Todo: smooth crowd_flow_map

        for ii, agent_i in enumerate(all_agents):
            pos_i = agent_i[0:2]
            vel_i = agent_i[2:4]
            self._aggregate_to_virtual_pdf_(pos_i, vel_i, crowd_flow_map)
            if ii >= 3: break  # debug

        n_tries = 0  # fixme
        for i in range(n_tries):
            accept_suggested_loc = True
            suggested_loc = self.social_ties_cartesian_pdf_aggregated.sample_random_pos()

            dist_to_existing_agents = norm(np.stack([suggested_loc[0]-np.array(all_agents)[:, 0],
                                                     suggested_loc[1]-np.array(all_agents)[:, 1]]), axis=0)
            if min(dist_to_existing_agents) < 0.7 or min(dist_to_existing_agents) > 6:
                accept_suggested_loc = False
            if accept_suggested_loc:
                # print(int(round(crowd_flow_map.get(suggested_loc))))
                suggested_vel = FlowClassifier().preset_flow_classes[
                    int(round(crowd_flow_map.get(suggested_loc)))].velocity
                new_ped = Pedestrian(suggested_loc, suggested_vel)
                new_ped.projected = True
                new_ped.color = SKY_BLUE_COLOR
                projected_agents.append(new_ped)

                all_agents.append(np.concatenate([suggested_loc, suggested_vel]))  # cuz: this agent should be considered when synthesising next agent
                self._aggregate_to_virtual_pdf_(suggested_loc, suggested_vel, crowd_flow_map)

        # self.fig2.imshow(np.flipud(self.social_ties_cartesian_pdf_aggregated.data.T))
        return projected_agents

    def plot(self, title=""):

        if not len(self.axes):  # First time initialization
            # _, self.fig2 = plt.subplots()
            self.fig = plt.figure(figsize=(8, 4))
            grid_spec = gridspec.GridSpec(2, 4, height_ratios=[3, 1], width_ratios=[1, 3, 3, 1], wspace=0.15,
                                          hspace=0.2)
            self.axes = [[None, None, None, None], [None, None]]
            # self.axes[0, 0].remove()
            # self.axes[0, 0] = self.fig.add_subplot(221, projection="polar")
            self.axes[0][0] = plt.subplot(grid_spec[0, 0])
            self.axes[0][1] = plt.subplot(grid_spec[0, 1], projection='polar')
            self.axes[0][2] = plt.subplot(grid_spec[0, 2], projection='polar')
            self.axes[0][3] = plt.subplot(grid_spec[0, 3])
            self.axes[1][0] = plt.subplot(grid_spec[1, :2])
            self.axes[1][1] = plt.subplot(grid_spec[1, 2:])
            # self.axes[1, 0].remove()
            # self.axes[1, 1].remove()
            # self.axes[1, 0] = self.fig.add_subplot(212)

        angle_y_ticks = [-135, -90, -45, 0, 45, 90, 135]

        # Strong Ties
        # ===================================
        strong_angular_pdf = np.nansum(self.strong_ties_pdf_polar, axis=0)
        strong_length_pdf = np.nansum(self.strong_ties_pdf_polar, axis=1)

        self.axes[0][1].clear()
        polar_plot = self.axes[0][1].pcolormesh(self.theta_edges, self.rho_edges, self.strong_ties_pdf_polar, vmin=0,
                                                cmap='YlGnBu')
        # self.axes[0][1].set_ylabel("Polar Histogram of Strong Ties", labelpad=40)

        # Marginal PDF: Tie Angles
        self.axes[0][0].clear()
        self.axes[0][0].set_title("Tie Angle PDF", fontsize=10)
        strong_angle_plot = self.axes[0][0].plot(strong_angular_pdf, np.rad2deg(self.theta_bin_midpoints), 'r')
        self.axes[0][0].fill_betweenx(np.rad2deg(self.theta_bin_midpoints), 0, strong_angular_pdf)
        self.axes[0][0].set_xlim([0, max(strong_angular_pdf) * 1.1])
        self.axes[0][0].set_ylim([-151, 151])
        self.axes[0][0].set_yticks(angle_y_ticks)
        self.axes[0][0].set_yticklabels(["$%d^\circ$" % deg for deg in angle_y_ticks])
        self.axes[0][0].invert_xaxis()

        # Marginal PDF: Tie Length
        self.axes[1][0].clear()
        # pcf_plot = self.axes[1][0].plot(self.rho_bin_midpoints, dist_hist, 'r')
        # self.axes[1][0].fill_between(self.rho_bin_midpoints, 0, dist_hist)
        self.axes[1][0].bar(self.rho_bin_midpoints, strong_length_pdf, width=np.diff(self.rho_edges)[0] * 0.8)

        self.axes[1][0].set_title("Tie Length PDF  ", loc='right', fontsize=10, pad=-14)
        # self.axes[1][0].set_xlabel('$\it{m}$', labelpad=-5)
        self.axes[1][0].set_xlabel('Strong Ties', labelpad=2)
        # self.axes[1][0].grid()

        # ===================================+++
        absent_angular_pdf = np.nansum(self.absent_ties_pdf_polar, axis=0)
        absent_length_pdf = np.nansum(self.absent_ties_pdf_polar, axis=1)

        # Absent Ties
        self.axes[0][2].clear()
        polar_plot = self.axes[0][2].pcolormesh(self.theta_edges, self.rho_edges, self.absent_ties_pdf_polar, vmin=0,
                                                cmap='YlOrRd')
        # self.axes[0][2].set_ylabel("Polar Histogram of Absent Ties", labelpad=40)
        # if len(title): self.axes[0][2].set_title(title, pad=20)

        # Marginal PDF: Tie Angles
        self.axes[0][3].clear()
        self.axes[0][3].set_title("Tie Angle PDF", fontsize=10)
        absent_angle_plot = self.axes[0][3].plot(absent_angular_pdf, np.rad2deg(self.theta_bin_midpoints), 'b')
        self.axes[0][3].fill_betweenx(np.rad2deg(self.theta_bin_midpoints), 0, absent_angular_pdf, color='darkred')
        self.axes[0][3].set_xlim([0, max(absent_angular_pdf) * 1.1])
        self.axes[0][3].set_ylim([-151, 151])
        self.axes[0][3].set_yticks(angle_y_ticks)
        self.axes[0][3].set_yticklabels(["$%d^\circ$" % deg for deg in angle_y_ticks])
        self.axes[0][3].yaxis.tick_right()

        # Marginal PDF: Tie Length
        self.axes[1][1].clear()
        self.axes[1][1].bar(self.rho_bin_midpoints, absent_length_pdf, width=np.diff(self.rho_edges)[0] * 0.8,
                            color='darkred')

        self.axes[1][1].set_title("Tie Length PDF  ", loc='right', fontsize=10, pad=-14)
        # self.axes[1][1].set_xlabel('$\it{m}$', labelpad=-5)
        self.axes[1][1].set_xlabel('Absent Ties', labelpad=2)
        self.axes[1][1].yaxis.tick_right()
        # ===================================+++

        if len(title):
            self.fig.suptitle(title)

        plt.pause(0.001)

    def save_pdf(self, fname):
        """save the distributions to file"""
        np.savez(fname,
                 strong_ties_pdf_polar=self.strong_ties_pdf_polar,
                 absent_ties_pdf_polar=self.absent_ties_pdf_polar,
                 num_prior_strong_ties=self.num_prior_strong_ties,
                 num_prior_absent_ties=self.num_prior_absent_ties,
                 rho_edges=self.rho_edges,
                 theta_edges=self.theta_edges)

    def load_pdf(self, fname):
        """load the distributions from file"""
        if not os.path.exists(fname):
            raise ValueError("Could not find the data file for prior social ties")
        npz_file = np.load(fname)
        self.num_prior_strong_ties = npz_file['num_prior_strong_ties']
        self.num_prior_absent_ties = npz_file['num_prior_absent_ties']

        src_strong_ties_pdf_polar = npz_file['strong_ties_pdf_polar']
        src_absent_ties_pdf_polar = npz_file['absent_ties_pdf_polar']
        if 'rho_edges' in npz_file:
            src_rho_edges = npz_file['rho_edges']
        else:
            max_distance = 5
            radial_resolution = 4
            src_rho_edges = np.linspace(0, max_distance, max_distance * radial_resolution + 1)

        if 'theta_edges' in npz_file:
            src_theta_edges = npz_file['theta_edges']
        else:
            angular_resolution = 36
            src_theta_edges = np.linspace(-np.pi, np.pi, angular_resolution + 1)


        if len(src_rho_edges) == len(self.rho_edges) and \
                np.all(np.isclose(src_rho_edges, self.rho_edges)) and \
                len(src_theta_edges) == len(self.theta_edges) and \
                np.all(np.isclose(src_theta_edges, self.theta_edges)):

            self.strong_ties_pdf_polar = src_strong_ties_pdf_polar
            self.absent_ties_pdf_polar = src_absent_ties_pdf_polar

        else:  # map the loaded histogram (src) to the histogram shape defined here (dst)
            src_rhos = (src_rho_edges[:-1] + src_rho_edges[1:]) / 2
            src_thetas = (src_theta_edges[:-1] + src_theta_edges[1:]) / 2
            dst_rhos, dst_thetas = np.meshgrid(self.rho_bin_midpoints, self.theta_bin_midpoints)

            interp_src_rhos = interp1d(src_rhos, np.arange(len(src_rhos)), bounds_error=False)
            interp_src_thetas = interp1d(src_thetas, np.arange(len(src_thetas)))

            new_interp_rhos = interp_src_rhos(dst_rhos.ravel())
            new_interp_thetas = interp_src_thetas(dst_thetas.ravel())

            new_interp_rhos[dst_rhos.ravel() < src_rhos.min()] = 0
            new_interp_rhos[dst_rhos.ravel() > src_rhos.max()] = src_rhos.max()

            self.strong_ties_pdf_polar = map_coordinates(src_strong_ties_pdf_polar,
                                                         np.array([new_interp_rhos, new_interp_thetas]),
                                                         order=2, cval=1, mode='reflect').reshape(dst_rhos.shape).T
            self.absent_ties_pdf_polar = map_coordinates(src_absent_ties_pdf_polar,
                                                         np.array([new_interp_rhos, new_interp_thetas]),
                                                         order=2, cval=1, mode='reflect').reshape(dst_rhos.shape).T


        self.strong_ties_pdf_polar = 5 * sigmoid(self.strong_ties_pdf_polar)
        self.absent_ties_pdf_polar = 5 * sigmoid(self.absent_ties_pdf_polar)

        # fade out the histograms (linear interpolation)
        fading_radius = 2.5
        p_val = self.padding_prob_val  # padding value
        if np.any(self.rho_edges > fading_radius):
            first_fading_index = (self.rho_bin_midpoints > fading_radius).nonzero()[0][0]
            last_index = len(self.rho_bin_midpoints) - 1
            intp_alpha = np.linspace(1, 0, last_index-first_fading_index + 1)
            for ii in range(len(intp_alpha)):
                self.strong_ties_pdf_polar[ii+first_fading_index:, :] = (1-intp_alpha[ii]) * p_val + \
                                                     intp_alpha[ii] * self.strong_ties_pdf_polar[first_fading_index - 1,:]
                self.absent_ties_pdf_polar[ii+first_fading_index:, :] = (1-intp_alpha[ii]) * p_val + \
                                                     intp_alpha[ii] * self.absent_ties_pdf_polar[first_fading_index - 1,:]




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

