import os
import random
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from pygame import Rect
from sklearn.mixture import gaussian_mixture
from sklearn import cluster, datasets, mixture
from bisect import bisect_left, bisect_right

from toolkit.core.trajdataset import TrajDataset
from toolkit.loaders.loader_hermes import load_bottleneck
from crowdrep_bot.crowd_imputation.dart_throwing import DartThrowing
from crowdrep_bot.crowd_imputation.pcf import PcfPattern, PcfReconstruction

import matplotlib
matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk


class LookUpTableSampler:
    """
        An alternative for GMM when the number of detections in a cell is not enough for gmm.fit()
    """
    def __init__(self):
        self.points = []

    def fit(self, X):
        X = np.array(X)
        self.points = X

    def sample(self, n_samples=1):
        ind = 0
        if len(self.points) > 1:
            ind = np.random.randint(0, len(self.points))
        return [self.points[ind] + np.random.randn(2) * 0.1], n_samples


class PCFCrowdSynthesizer:
    def __init__(self, active_vels=False):
        self.crowd = []
        self.synthetic_crowds = []
        self.data_rect = []
        self.dart_thrower = []
        self.pcf_sampler = PcfReconstruction()
        self.active_vels = active_vels

        self.pcf_range = np.arange(0.2, 5, 0.1)
        self.grid_size = (64, 64)  # for GMM grid / spatial heatmap
        self.vel_gmm_n_components = 3
        self.tpcf = PcfPattern()
        self.heatmap_grid = np.ones(self.grid_size, dtype=np.float)
        self.vel_pdf_grid = [[None for i in range(self.grid_size[1])]
                             for j in range(self.grid_size[0])]

        # a lambda function that return the grid cell index for a given location
        # this line is just for initialization of the function, it should be defined later
        self.map_to_grid_coord = lambda x, y: (-1, -1)

        self.synthesis_max_try = 5000
        self.n_agetns_histogram = np.empty(100)
        self.synthesis_max_pts = 100  # FIXME: calc from data

    # ===================================
    # ======== Extract Features =========
    # ===================================
    def extract_features(self, dataset: TrajDataset):
        self.compute_target_pcf(dataset)
        # self.compute_heatmap(data)
        # if self.active_vels:
        #     self.compute_vel_gmm(data)
        min_x, max_x = min(dataset.data["pos_x"]), max(dataset.data["pos_x"])
        min_y, max_y = min(dataset.data["pos_y"]), max(dataset.data["pos_y"])

        map_to_world_coord = lambda ind: [(ind % self.grid_size[1] + .5) / self.grid_size[0] * (max_x - min_x) + min_x,
                                          (ind // self.grid_size[1] + .5) / self.grid_size[1] * (max_y - min_y) + min_y]

        self.map_to_grid_coord = lambda x, y: [int(np.floor((x - min_x) / (max_x - min_x + 1E-6) * self.grid_size[0])),
                                               int(np.floor((y - min_y) / (max_y - min_y + 1E-6) * self.grid_size[1]))]
        self.pcf_sampler.set_pdf_grid(self.heatmap_grid, map_to_world_coord)

        self.data_rect = Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        self.dart_thrower = DartThrowing(self.data_rect, self.tpcf.pcf_values, self.tpcf.rad_values, k=30)

    def compute_target_pcf(self, dataset):
        all_pcfs = []
        frames = dataset.get_frames()
        self.n_agetns_histogram *= 0

        for t in tqdm(range(int(len(frames) * 0.7)), desc='Computing PCF of dataset'):
            frame_t = frames[t]
            points_t = frame_t[['pos_x', 'pos_y']].to_numpy()
            self.tpcf.update(points_t, self.pcf_range)
            pcf_t = self.tpcf.pcf_values
            all_pcfs.append(pcf_t)

            N = len(points_t)
            if N < len(self.n_agetns_histogram):
                self.n_agetns_histogram[N] += 1
        self.tpcf.pcf_values = np.array(all_pcfs).mean(axis=0)

    def compute_heatmap(self, dataset):
        all_points = dataset.get_all_points()
        all_points = np.array(all_points)

        # heatmap should be smoothed
        hist, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=self.grid_size)
        self.heatmap_grid = hist.T

        x_min, x_max = min(all_points[:, 0]), max(all_points[:, 0])
        y_min, y_max = min(all_points[:, 1]), max(all_points[:, 1])
        # FIXME: grid_size[0] and grid_size[1] may need to exchange thier place


    def compute_vel_gmm(self, dataset):
        all_trajs = dataset.get_all_trajs()
        all_vs = [traj_i[1:] - traj_i[:-1] for traj_i in all_trajs]
        all_pvs = []
        grid_vs = [ [[] for ii in range(self.grid_size[1])]
                    for jj in range(self.grid_size[0]) ]
        for ii in range(len(all_trajs)):
            pvs_i = np.concatenate([all_trajs[ii][:-1], all_vs[ii]], axis=1)
            all_pvs.extend(pvs_i)
        for pv in all_pvs:
            u, v = self.map_to_grid_coord(pv[0], pv[1])
            grid_vs[u][v].append(pv[2:])

        for u in range(self.grid_size[0]):
            for v in range(self.grid_size[1]):
                # print(grid_vs[u][v])
                # print('variance = ', var_uv)

                if len(grid_vs[u][v]) == 0:
                    grid_vs[u][v] = [np.array([0, 0], dtype=np.float)]

                var_uv = np.var(grid_vs[u][v], axis=0)
                if len(grid_vs[u][v]) > self.vel_gmm_n_components and np.linalg.norm(var_uv) > 1E-6:
                    self.vel_pdf_grid[u][v] = mixture.GaussianMixture(n_components=2, covariance_type='diag', max_iter=10)
                else:
                    self.vel_pdf_grid[u][v] = LookUpTableSampler()

                self.vel_pdf_grid[u][v].fit(grid_vs[u][v])

    # ===================================
    # ======== Synthesize Crowd =========
    # ===================================
    def pcf_synthesize(self, detections):
        final_points = detections.copy()
        target_pcf = PcfPattern()
        target_pcf.update(detections, self.pcf_range)
        target_pcf.pcf_values = np.maximum(self.tpcf.pcf_values, target_pcf.pcf_values)

        rnd_value = np.random.rand()
        weights = self.n_agetns_histogram.cumsum() / self.n_agetns_histogram.sum()
        max_n_points = bisect_left(weights, rnd_value)
        # max_n_points = self.synthesis_max_pts

        try_counter = 0
        while try_counter < self.synthesis_max_try:
            try_counter += 1

            temp_points = final_points.copy()
            p_new = self.draw_point()
            temp_points.append(p_new)

            if target_pcf.is_compatible(temp_points):
                final_points.append(p_new)

                # FIXME: add the velocity vector
                # v_new = self.draw_vel(p_new)
                # final_points[-1] = np.append(p_new, v_new)

            # FIXME: refinement does not work
            # else:
            #     temp_points.pop()
            #     p_new_new = target_pcf.check_and_refine(temp_points, p_new)
            #     temp_points.append(p_new_new)
            #     if target_pcf.compatible(temp_points):
            #         final_points.append(p_new_new)
            #         print('Refinement was successful!')

            if len(final_points) >= max_n_points:
                break

        return np.array(final_points)

    def assign_velocity(self, locs):
        vels = []
        for pt in locs:
            vel_i = self.draw_vel(pt)
            vels.append(vel_i)

        locs_and_vels = np.concatenate([locs, np.array(vels).reshape((-1, 2))], axis=1)
        return locs_and_vels

    def draw_point(self):
        return self.pcf_sampler.random_sample()

    def draw_vel(self, p):
        # TODO: use self.gaussian_mixture
        u, v = self.map_to_grid_coord(p[0], p[1])
        s_pnt, _ = self.vel_pdf_grid[u][v].sample()
        return s_pnt

    def analyze_and_plot(self, det_pnts, gt_pnts, n_configs, verbose=False):

        for kk in range(n_configs):
            pcf_pnts = self.pcf_synthesize(det_pnts.tolist())
            if self.active_vels:
                pcf_pnts = self.assign_velocity(pcf_pnts)
            dart_thrower_pnts = self.dart_thrower.create_samples(det_pnts)

            pcf_gt = PcfPattern()
            pcf_gt.update(gt_pnts, self.pcf_range)

            pcf_final = PcfPattern()
            pcf_final.update(pcf_pnts, self.pcf_range)

            pcf_start = PcfPattern()
            pcf_start.update(det_pnts, self.pcf_range)

            plt.figure()
            plt.subplot(4, 1, 4)
            plt.ylabel("PCF")
            plt.plot(self.tpcf.rad_values, self.tpcf.pcf_values, 'b', label='Target PCF')
            plt.plot(pcf_gt.rad_values, pcf_gt.pcf_values, 'g', label='Actual PCF[t]')
            plt.plot(pcf_start.rad_values, pcf_start.pcf_values, 'y', label='Initial (detections)')
            plt.plot(pcf_final.rad_values, pcf_final.pcf_values, 'r', label='Synthesize PCF')
            plt.legend()

            plt.subplot(4, 1, 1)
            # plt.imshow(np.flipud(self.heatmap_grid))
            # plt.title("Heatmap")

            plt.scatter(gt_pnts[:, 0], gt_pnts[:, 1], color='green', s=100, label='ground truth points')
            plt.xlim([self.data_rect.left, self.data_rect.right])
            plt.ylim([self.data_rect.top, self.data_rect.bottom])
            # plt.legend()
            plt.ylabel('Actual Locs[t]')

            plt.subplot(4, 1, 2)
            plt.scatter(det_pnts[:, 0], det_pnts[:, 1], color='blue', s=80, label='given points')
            plt.scatter(dart_thrower_pnts[:, 0], dart_thrower_pnts[:, 1], color='green', marker='x', label='dart-throwing')
            plt.xlim([self.data_rect.left, self.data_rect.right])
            plt.ylim([self.data_rect.top, self.data_rect.bottom])
            # plt.legend()
            plt.ylabel('Dart Throwing')

            plt.subplot(4, 1, 3)
            plt.scatter(det_pnts[:, 0], det_pnts[:, 1], color='blue', s=80, label='given points')
            plt.scatter(pcf_pnts[:, 0], pcf_pnts[:, 1], color='red', marker='x', label='Pcf reconstruction')
            if pcf_pnts.shape[1] > 3:
                for ii in range(len(pcf_pnts)):  # draw velocity arrows
                    plt.arrow(pcf_pnts[ii, 0], pcf_pnts[ii, 1], pcf_pnts[ii, 2] * 2, pcf_pnts[ii, 3] * 2)

            plt.xlim([self.data_rect.left, self.data_rect.right])
            plt.ylim([self.data_rect.top, self.data_rect.bottom])
            # plt.legend()
            plt.ylabel('PCF Reconst')

            # plt.ylim([0, 0.15])
            plt.show()


if __name__ == '__main__':
    # TODO: Algorithm
    #  rand_t <- Choose a random frame
    #  all_detections = D(rand_t)
    #  if len(all_detections) < 4: skip ...
    #  partial_detection <- a subset of all_detections
    #  for kk in range(n_configs):
    #      syn_crowds[kk] = crowd_syn.synthesize_init(partial_detection)

    # dataset = ParserETH('/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/obsmat.txt')
    # dataset = ParserGC('/home/cyrus/workspace2/OpenTraj/GC/Annotation', world_coord=True)
    # dataset = ParserHermes('/home/cyrus/workspace2/OpenTraj/HERMES/Corridor-1D/uo-050-180-180_combined_MB.txt')
    OPENTRAJ_ROOT = "/home/cyrus/workspace2/OpenTraj"
    annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-090-090.txt')
    dataset = load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4])

    crowd_syn = PCFCrowdSynthesizer()
    crowd_syn.extract_features(dataset)
    frames = dataset.get_frames()

    # select random agents to test
    gt_pts = []
    for kk in range(100):
        frame_t = random.choice(frames)
        gt_pts = frame_t[['pos_x', 'pos_y']].to_numpy()
        if len(gt_pts) >= 4:
            break

    n_gt_points = len(gt_pts)
    keep_k = max(3, n_gt_points // 5)
    anchor_det_ind = random.randint(0, len(gt_pts)-1)
    dist_from_anchor = np.linalg.norm(gt_pts - gt_pts[anchor_det_ind], axis=1)
    anchor_neighbor_inds = np.argsort(dist_from_anchor)
    partial_detection = gt_pts[anchor_neighbor_inds[:keep_k]]

    n_configs = 2
    crowd_syn.analyze_and_plot(partial_detection, gt_pts, n_configs)

    pcf_pnts = crowd_syn.pcf_synthesize(partial_detection.tolist())
    dart_thrower_pnts = crowd_syn.dart_thrower.create_samples(partial_detection)
