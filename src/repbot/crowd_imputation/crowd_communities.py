# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csgraph, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from repbot.crowd_imputation.bivariate_gaussian import BivariateGaussianMixtureModel, BivariateGaussian


def connected_components(vertices, edges):
    N = len(vertices)  # number of nodes
    map_vertices = {}  # map vertex original id => counter id
    for ii, v in enumerate(vertices):
        map_vertices[v] = ii

    mapped_edges = []  # edges, rewritten with above mapping
    if len(edges):
        for e in edges:
            mapped_edges.append([map_vertices[e[0]], map_vertices[e[1]]])
        mapped_edges = np.stack(mapped_edges)
        # Sparse matrix of edges of the social graph
        G = csr_matrix(([1] * len(edges), (mapped_edges[:, 0], mapped_edges[:, 1])), shape=(N, N))
        nb_comps, comps = csgraph.connected_components(G)  # calc communities
    else:
        nb_comps, comps = csgraph.connected_components(np.eye(N))  # calc communities

    comps_set = [[] for _ in range(nb_comps)]
    for key, val in map_vertices.items():
        comps_set[comps[val]].append(key)
    return comps_set


class FlowClass:
    def __init__(self, id, name, color, velocity):
        self.id = id
        self.name = name
        self.color = color
        self.velocity = velocity


class CommunityHandler:
    def __init__(self):
        self.ped_ids = []
        self.communities = []
        self.community_velocities = []  # the same order of communities
        self.color_set = ['blue', 'red', 'green', 'gold', 'cyan', 'purple', 'hotpink', 'brown',
                          'darkblue', 'darkred', 'darkgreen', 'khaki', 'darkturquoise', 'darkviolet', 'chocolate']

    def calc_velocities(self, vels_t):  # vels_t has the same order as self.ped_ids
        self.community_velocities = []
        for community in self.communities:
            vel_sum = np.zeros(2)
            for ii, pid in enumerate(community):
                vel_sum += vels_t[self.ped_ids.index(pid)]
            self.community_velocities.append(vel_sum / len(community))

    def find_territories(self, poss_t, vels_t, xx, yy):
        bgm = BivariateGaussianMixtureModel()
        for i in range(len(self.ped_ids)):
            polar_vel = [np.linalg.norm(vels_t[i]), np.arctan2(vels_t[i][1], vels_t[i][0])]

            if polar_vel[0] < 0.1:  # filter still agents
                continue

            community_id = -1
            for jj, community in enumerate(self.communities):
                if self.ped_ids[i] in community:
                    community_id = jj
                    break
            bgm.add_component(BivariateGaussian(poss_t[i][0], poss_t[i][1],
                                                sigma_x=polar_vel[0] / 5 + 0.1, sigma_y=0.1,
                                                theta=polar_vel[1]), weight=1,
                              target=community_id)  # Todo: => the component labels
        # draw_bgmm(bgm, xx, yy)  # => for paper
        territory_id_map = bgm.classify_kNN(xx, yy).T
        onehot_encoder = OneHotEncoder(sparse=False, categories=[np.array(range(len(self.communities)))])
        onehot_terr_id = onehot_encoder.fit_transform(territory_id_map.astype(np.int8).reshape(-1, 1))
        velocity_map = np.matmul(onehot_terr_id, np.array(self.community_velocities))

        return territory_id_map, velocity_map

    def classify(self, agents_locs, agents_vel):
        return []

    def label2color(self, labels):  # for debug/visualization
        colors = np.zeros(labels.size, dtype='U11')
        for i, l in enumerate(labels.reshape(-1).astype(int)):
            colors[i] = self.color_set[l]
        return colors.reshape(labels.shape)


class CommunityDetector:
    def __init__(self, scenario_fps):
        self.all_ties = {}
        # algorithm thresholds
        self.max_distance = 5
        self.thre_length = 0.5
        self.thre_angle = np.pi / 8
        self.prev_t = max(1, int(scenario_fps * 0.25))  # 0.25s ago

    def cluster_communities(self, pids_t, poss_t, vels_t):
        strong_ties = []
        absent_ties = []
        strong_ties_idx = []
        absent_ties_idx = []

        N = len(pids_t)
        # rot_matrices_t = np.zeros((N, 2, 2))
        oriens_t = np.arctan2(vels_t[:, 1], vels_t[:, 0])

        tiled_poss = np.tile(poss_t, (N, 1, 1))
        D = tiled_poss - tiled_poss.transpose((1, 0, 2))

        link_angles = np.arctan2(D[:, :, 1], D[:, :, 0])
        link_lengths = np.linalg.norm(D, axis=2)

        for ii, pid_i in enumerate(pids_t):
            for jj, pid_j in enumerate(pids_t):
                if ii == jj:
                    continue

                IJ_idx = (pid_i, pid_j)
                if IJ_idx not in self.all_ties:
                    self.all_ties[IJ_idx] = []

                rotated_link_angle = (link_angles[ii, jj] - oriens_t[ii] + np.pi) % (2 * np.pi) - np.pi
                self.all_ties[IJ_idx].append([link_lengths[ii, jj], rotated_link_angle])

                if len(self.all_ties[IJ_idx]) >= self.prev_t:  # this link has existed for at least 1 sec
                    d_tie_len = self.all_ties[IJ_idx][-1][0] - self.all_ties[IJ_idx][-self.prev_t][0]
                    d_tie_angle = self.all_ties[IJ_idx][-1][1] - self.all_ties[IJ_idx][-self.prev_t][1]
                    d_tie_angle = (d_tie_angle + np.pi) % (2 * np.pi) - np.pi
                    d_orien = (oriens_t[ii] - oriens_t[jj] + np.pi) % (2 * np.pi) - np.pi
                    if self.all_ties[IJ_idx][-1][0] > self.max_distance:
                        continue
                    if abs(d_tie_len) < self.thre_length and abs(d_tie_angle) < self.thre_angle and abs(
                            d_orien) < np.pi / 4:
                        strong_ties.append(self.all_ties[IJ_idx][-1])
                        strong_ties_idx.append(IJ_idx)
                    else:
                        absent_ties.append(self.all_ties[IJ_idx][-1])
                        absent_ties_idx.append(IJ_idx)

        # find communities
        strong_ties_idx_np = np.array(strong_ties_idx)
        communities = connected_components(pids_t, strong_ties_idx_np)  # calc communities
        # print('***************************')
        # print("ids = ", pids_t)
        # print("strong_ties_idx = ", strong_ties_idx_np.T)
        # print("communities = ", communities)

        return strong_ties, absent_ties, strong_ties_idx, absent_ties_idx, communities


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    matplotlib.use('TkAgg')

    from toolkit.loaders.loader_metafile import load_crowds

    datasets = []
    OPENTRAJ_ROOT = "/home/cyrus/workspace2/OpenTraj"
    output_dir = "/home/cyrus/Dropbox/FollowBot/exp/pdf"
    save_links_dir = "/home/cyrus/Dropbox/FollowBot/exp/links"
    VISUALIZE = True

    # ======== load dataset =========
    # annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-075-075.txt')
    # datasets.append(load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4]))

    # annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-090-090.txt')
    # datasets.append(load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4]))

    annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/UCY/zara01/annotation.vsp')
    zara1 = load_crowds(annot_file, homog_file=os.path.join(OPENTRAJ_ROOT, "datasets/UCY/zara01/H.txt"), title="Zara01")
    datasets.append(zara1)
    # -------------------------------

    for dataset in datasets:
        min_x, max_x = min(dataset.data["pos_x"]), max(dataset.data["pos_x"])
        min_y, max_y = min(dataset.data["pos_y"]), max(dataset.data["pos_y"])
        # -------------------------------

        frames = dataset.get_frames()
        n_frames = len(frames)
        dt_ = np.diff(dataset.data["timestamp"].unique()[:2])[0]
        fps = (1 / dt_)
        n_agents = dataset.data["agent_id"].nunique()

        community_handler = CommunityHandler()
        community_detector = CommunityDetector(scenario_fps=fps)
        all_ties = {}
        strong_ties = []
        absent_ties = []
        tie_strength = []

        # visualize stuff
        if VISUALIZE:
            fig = plt.figure(figsize=(max_x - min_x, max_y - min_y))
            # fig, ax = plt.subplots()
            # ax.set_aspect(aspect='equal')

            _break = False
            _pause = False


            def p(event):
                global _break, _pause
                if event.key == 'escape':
                    _break = True
                if event.key == ' ':
                    _pause = not _pause


            fig.canvas.mpl_connect('key_press_event', p)

        for t in range(int(len(frames) * 0.7)):
            frame_data = frames[t]
            plt.axis('equal')
            pids_t = frame_data["agent_id"].to_numpy()
            poss_t = frame_data[["pos_x", "pos_y"]].to_numpy()
            vels_t = frame_data[["vel_x", "vel_y"]].to_numpy()
            N = len(pids_t)

            strong_ties, absent_ties, strong_ties_idx, absent_ties_idx, communities = \
                community_detector.cluster_communities(pids_t, poss_t, vels_t)

            community_handler.communities = communities
            community_handler.ped_ids = list(pids_t)
            community_handler.calc_velocities(vels_t)
            community_velocities = community_handler.community_velocities

            # assign territories (a map)
            mapResolution = 4
            xx, yy = np.meshgrid(np.arange(min_x, max_x, 1 / mapResolution),
                                 np.arange(min_y, max_y, 1 / mapResolution))
            territory_map, velocity_map = community_handler.find_territories(poss_t, vels_t, xx, yy)
            # =========================================

            # draw agents and ties
            if VISUALIZE:
                colors = community_handler.label2color(territory_map).T
                plt.clf()
                plt.xlim([min_x, max_x])
                plt.ylim([min_y, max_y])
                plt.scatter(xx.reshape(-1), yy.reshape(-1), alpha=0.1, c=colors.reshape(-1))
                plt.title("%d" % t)

                for ii in range(N):
                    for jj in range(N):
                        id_i, id_j = pids_t[ii], pids_t[jj]
                        if [id_i, id_j] in strong_ties_idx:
                            plt.plot([poss_t[ii, 0], poss_t[jj, 0]], [poss_t[ii, 1], poss_t[jj, 1]], 'g--', alpha=0.3)
                        # elif [id_i, id_j] in absent_ties_idx:
                        #     plt.plot([poss_t[ii, 0], poss_t[jj, 0]], [poss_t[ii, 1], poss_t[jj, 1]], 'r--', alpha=0.3)

                for k, community in enumerate(communities):
                    color = community_handler.color_set[k % len(community_handler.color_set)]
                    for ii, pid in enumerate(community):
                        ped_ind = pids_t.tolist().index(pid)
                        plt.scatter(poss_t[ped_ind, 0], poss_t[ped_ind, 1], c=color, s=84)
                        plt.arrow(poss_t[ped_ind, 0], poss_t[ped_ind, 1],
                                  vels_t[ped_ind, 0] * 0.7, vels_t[ped_ind, 1] * 0.7,
                                  head_width=0.15, color='black', alpha=0.5)

                plt.pause(0.01)

                if _break:
                    break
                while _pause:
                    plt.pause(0.25)
