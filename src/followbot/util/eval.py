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
    def __init__(self):
        self.sigma = 0.5

        self.occ_map_fig = plt.figure(figsize=(7, 5))
        self.grid_spec = gridspec.GridSpec(2, 1)
        self.occ_map = [plt.subplot(self.grid_spec[0, 0]),
                        plt.subplot(self.grid_spec[1, 0])]

    def calc_error(self, gt_peds, proj_peds, predicted_pom: MappedArray, deubg=False, frame_id=-1):
        """
        :param predicted_pom:
        :param proj_peds: Projected (predicted) agents
        :param gt_peds: Ground truth agents
        :param frame_id: frame Number
        :return:
        """

        predicted_pom = predicted_pom.copy_constructor()  # to not change the original data
        # predicted_pom.data /= 2
        predicted_pom.fill(0)

        gt_pom = predicted_pom.copy_constructor()
        gt_pom.fill(0)
        xx, yy = gt_pom.meshgrid()

        # POM = sum of Gaussians, centered on gt pedestrians
        fill_radius = 0.2
        normalizer = np.sqrt(2 * np.pi) * self.sigma
        for agent in gt_peds:
            gt_pom.data += np.exp(-((xx - agent.pos[0]) ** 2 +
                                    (yy - agent.pos[1]) ** 2) / (2 * self.sigma ** 2)).T / normalizer

        for agent in proj_peds:
            predicted_pom.data += np.exp(-((xx - agent.pos[0]) ** 2 +
                                           (yy - agent.pos[1]) ** 2) / (2 * self.sigma ** 2)).T / normalizer

        if deubg:
            self.occ_map[0].axis("off")
            self.occ_map[0].set_title("Projections")
            self.occ_map[0].imshow(np.flipud(predicted_pom.data.T))
            self.occ_map[1].axis("off")
            self.occ_map[1].set_title("Ground Truth")
            self.occ_map[1].imshow(np.flipud(gt_pom.data.T))
            plt.pause(0.001)
            plt.savefig(os.path.join("/home/cyrus/Videos/followbot/projections", "%5d.jpg" % frame_id))

        # Binary Cross Entropy (BCE)
        MSE = np.mean((predicted_pom.data - gt_pom.data) ** 2)
        dim = predicted_pom.data.size
        BCE = -np.nansum(gt_pom.data.flatten() * np.log(predicted_pom.data.flatten())) / dim
        # print("BCE =", BCE, "MSE =", MSE)
        return MSE, BCE

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
