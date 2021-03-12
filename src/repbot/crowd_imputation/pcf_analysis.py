import os
from math import sqrt
import numpy as np
import scipy.optimize, scipy.stats
from scipy.stats import gamma
from crowd_prediction.crowd_synthesis.pcf import PcfPattern
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_gcs import load_gcs
from toolkit.loaders.loader_hermes import load_bottleneck


def calc_pcf(dataset):
    """ given a crowd dataset, this function performs different analysis and displays the plots """

    pcf_for_all_frames = []
    sigma = 0.2
    pcf_range = np.arange(0.2, 4, 0.1)
    pcf_mean_obj = PcfPattern(sigma_=sigma)

    for t in dataset.t_p_dict:
        pcf_mean_obj.update(dataset.t_p_dict[t], pcf_range)
        pcf_t = pcf_mean_obj.pcf_values
        pcf_for_all_frames.append(pcf_t)

    # TPCF formula
    pcf_vanilla_mean = np.array(pcf_for_all_frames).mean(axis=0)
    pcf_vanilla_var = np.array(pcf_for_all_frames).var(axis=0)
    # std_tpcf = std_tpcf / sum(std_tpcf)  # normalize

    # ============================================================
    # filter irrelevant distances
    all_ids = list(dataset.id_t_dict.keys())
    n_peds = len(all_ids)
    dist_matrix = np.ones((n_peds, n_peds)) * 1000.
    max_id = max(all_ids)
    index_of = np.zeros(max_id + 1, dtype=int)
    for id in all_ids:
        index_of[id] = all_ids.index(id)

    for t in sorted(dataset.t_id_dict.keys()):
        ids_t = dataset.t_id_dict[t]
        ps_t = np.array(dataset.t_p_dict[t])
        dists = euclidean_distances(ps_t)

        N_t = len(ids_t)
        for ii, id_i in enumerate(ids_t):
            for jj, id_j in enumerate(ids_t):
                if id_i == id_j: continue
                dij = dists[ii, jj]

                id_i_index = index_of[id_i]
                id_j_index = index_of[id_j]
                dist_matrix[id_i_index, id_j_index] = min(dist_matrix[id_i_index, id_j_index], dij)

        # if t > 2000: break

    min_dist_inds = np.where(dist_matrix < 1000)
    all_min_dists = dist_matrix[min_dist_inds]

    pcf_filtered_mean = np.zeros(len(pcf_range))
    pcf_filtered_var = np.zeros(len(pcf_range))
    for ii, rr in enumerate(pcf_range):
        dists_sqr = np.power(all_min_dists - rr, 2)
        dists_exp = np.exp(-dists_sqr / (sigma ** 2)) / (sqrt(np.pi) * sigma)
        area_r = pcf_mean_obj.area_r(rr)
        pcf_r = np.sum(dists_exp) / area_r
        pcf_r /= 2 * len(dist_matrix)  # normalize
        pcf_filtered_mean[ii] = pcf_r
    # filtered_pcf = filtered_pcf / sum(filtered_pcf)  # normalize
    pcf_filtered_mean = pcf_filtered_mean * max(pcf_vanilla_mean) / max(pcf_filtered_mean)  # normalize

    plt.plot(pcf_range, pcf_filtered_mean, color='green', lw=3, label="TPCF / min distances")
    # plt.hist(all_min_dists, bins=pcf_range, density=True, zorder=-1, color='cyan', alpha=0.8, label='Hist[min distances]')
    plt.plot(pcf_range, pcf_vanilla_mean, color='r', lw=2, zorder=10, label="TPCF / Normal")
    plt.errorbar(pcf_range, pcf_vanilla_mean, pcf_vanilla_var, ecolor='b', lw=1, label="TPCF / Normal (Error)")
    plt.fill_between(pcf_range, pcf_vanilla_mean, pcf_vanilla_mean + pcf_vanilla_var, color='b', alpha=0.4)
    plt.fill_between(pcf_range, pcf_vanilla_mean, pcf_vanilla_mean - pcf_vanilla_var, color='b', alpha=0.4)
    plt.title('PCF on [%s] #Agents = %s' % (name, len(dataset.id_t_dict)))
    # plt.savefig('/home/cyrus/Dropbox/FollowBot/figs/pcf-Hermes-' + name + '.png')

    # =========== Fit to gamma distribution ============
    fitfunc = lambda x, p0, p1: p1 * x ** 2 + p0
    p0 = [0, 1.]
    popt, pcov = scipy.optimize.curve_fit(fitfunc, pcf_range, pcf_filtered_mean, p0)
    fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(pcf_filtered_mean)

    # plt.plot(pcf_range, fitfunc(pcf_range, *popt), 'g--')
    # plt.plot(pcf_range, scipy.stats.gamma.pdf(pcf_range, fit_alpha, loc=fit_loc))
    # plt.plot(pcf_range, scipy.stats.gamma.pdf(pcf_filtered_mean), 'K--', lw=1, label='chi-2 pdf')

    # alpha = 1.8  # test
    # plt.plot(pcf_range, gamma(alpha).pdf(pcf_range)/1.8, 'K-', lw=2, label='fit chi-2')

    plt.legend()
    plt.grid()


if __name__ == '__main__':
    open_traj_path = '/home/cyrus/workspace2/OpenTraj'

    datasets = {
        'GC': ParserGC(os.path.join(open_traj_path, 'GC/Annotation')),
        # 'ETH-Univ': ParserETH(os.path.join(open_traj_path, 'ETH/seq_eth/obsmat.txt')),
        # 'ETH-Hotel': ParserETH(os.path.join(open_traj_path, 'ETH/seq_hotel/obsmat.txt')),
        #
        # '1D-050-180-180': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-050-180-180_combined_MB.txt')),
        # '1D-100-180-180': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-100-180-180_combined_MB.txt')),
        # '1D-065-240-240': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-240-240-240_combined_MB.txt')),
        # '1D-095-240-240': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-095-240-240_combined_MB.txt')),
        #
        # # 2D - Symmetric
        # '2D-symmetric-300-050-050': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/bot-300-050-050_combined_MB.txt')),
        # '2D-symmetric-300-100-100': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/bot-300-100-100_combined_MB.txt')),
        # # 2D - Asymmetric
        # '2D-asymmetric-300-050-085': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/boa-300-050-085_combined_MB.txt')),
        # '2D-asymmetric-300-065-105': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/boa-300-065-105_combined_MB.txt')),
    }

    for name, dataset in datasets.items():
        print('num peds = %d' % len(dataset.id_t_dict))
        calc_pcf(dataset)

        plt.savefig('/home/cyrus/Dropbox/FollowBot/figs/pcf-' + name + '.png')
        # plt.show()

        plt.clf()
