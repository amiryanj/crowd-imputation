# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial.distance import squareform, pdist

from followbot.util.mapped_array import MappedArray


class Evaluation:
    def __init__(self, show_heatmap=False):
        self.sigma = 0.5

        if show_heatmap:
            self.occ_map_fig = plt.figure(figsize=(7, 5))
            self.grid_spec = gridspec.GridSpec(2, 1)
            self.occ_map = [plt.subplot(self.grid_spec[0, 0]),
                            plt.subplot(self.grid_spec[1, 0])]
        else:
            self.occ_map_fig = None

    def calc_error(self, meshgrid, dst_peds, src_peds, plot_text=""):
        """
        :param predicted_pom:
        :param dst_peds: Ground truth agents
        :param src_peds: Projected (predicted) agents
        :param frame_id: frame Number
        :return:
        """
        xx, yy = meshgrid
        gt_pom, predicted_pom = np.zeros_like(xx).T, np.zeros_like(xx).T

        # POM = sum of Gaussians, centered on agents
        normalizer = np.sqrt(2 * np.pi) * self.sigma
        for agent_pos in dst_peds:
            gt_pom += np.exp(-((xx - agent_pos[0]) ** 2 +
                               (yy - agent_pos[1]) ** 2) / (2 * self.sigma ** 2)).T / normalizer

        for agent_pos in src_peds:
            predicted_pom += np.exp(-((xx - agent_pos[0]) ** 2 +
                                      (yy - agent_pos[1]) ** 2) / (2 * self.sigma ** 2)).T / normalizer

        if self.occ_map_fig:
            self.occ_map[0].axis("off")
            self.occ_map[0].set_title("Projections")
            self.occ_map[0].imshow(np.flipud(predicted_pom.T))
            self.occ_map[1].axis("off")
            self.occ_map[1].set_title("Ground Truth")
            self.occ_map[1].imshow(np.flipud(gt_pom.T))
            plt.pause(0.001)
            if len(plot_text):
                plt.savefig(os.path.join("/home/cyrus/Music/followbot-projections", "%s.jpg" % plot_text))

        # Binary Cross Entropy (BCE)
        MSE = np.mean((predicted_pom - gt_pom) ** 2)
        dim = predicted_pom.size
        BCE = -np.nansum(gt_pom.flatten() * np.log(predicted_pom.flatten())) / dim
        return MSE, BCE, gt_pom, predicted_pom

    def local_density(self, locs):
        # for all pedestrians find its distance to NN
        distNN = []
        density = []
        LAMBDA = 1
        if len(locs) > 1:
            # find pairwise min distance
            dist = squareform(pdist(locs))
            pair_dist = []
            for pi in dist:
                pair_dist.append(np.array(pi))
                min_pi = [j for j in pi if j > 0.01]
                if len(min_pi) == 0:
                    min_dist = 0.01
                else:
                    min_dist = np.min(min_pi)
                distNN.append(min_dist)

            # calculate local density for agent pj
            for pj in range(len(dist)):
                dens_t_i = 1 / (2 * np.pi) \
                           * np.sum(1 / ((LAMBDA * np.array(distNN)) ** 2)
                                    * np.exp(
                    -np.divide(
                        (pair_dist[pj] ** 2),
                        (2 * (LAMBDA * np.array(distNN)) ** 2)
                    )
                )
                                    )

                density.append(dens_t_i)

        return density
