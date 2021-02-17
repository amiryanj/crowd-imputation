# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import time
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csgraph, csr_matrix

from followbot.robot_functions.bivariate_gaussian import BivariateGaussianMixtureModel, BivariateGaussian


def connected_components(vertices, edges):
    N = len(vertices)
    map_vertices = {}  # map vertex original id => counter id

    mapped_edges = []  # edges, rewritten with above mapping
    if len(edges):
        for ii, v in enumerate(vertices):
            map_vertices[v] = ii
        for e in edges:
            mapped_edges.append([map_vertices[e[0]], map_vertices[e[1]]])
        mapped_edges = np.stack(mapped_edges)
        # Sparse matrix of edges of the social graph
        G = csr_matrix(([1] * len(edges), (mapped_edges[:, 0], mapped_edges[:, 1])), shape=(N, N))
        nb_comps, comps = csgraph.connected_components(G)  # calc communities
    else:
        nb_comps, comps = csgraph.connected_components(np.zeros((N, N)))  # calc communities

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


class FlowClassifier:
    def __init__(self):
        self.classes = []
        self.preset_flow_classes = {0: FlowClass(0, 'still', 'g', velocity=np.zeros(2)),
                                    1: FlowClass(1, 'to_right', 'r', velocity=np.array([1.2, 0])),
                                    2: FlowClass(2, 'to_left', 'b', velocity=np.array([-1.2, 0])),
                                    }

    def classify(self, agents_locs, agents_vel):
        output = []
        if len(agents_vel):
            agents_orien = np.arctan2(agents_vel[:, 1], agents_vel[:, 0])

            for i in range(len(agents_orien)):
                # still agent
                if np.linalg.norm(agents_vel[i]) < 0.2:
                    output.append(self.preset_flow_classes[0])
                # towards_R agent
                elif -np.pi / 2 <= agents_orien[i] < np.pi / 2:
                    output.append(self.preset_flow_classes[1])
                # towards_L agent
                else:
                    output.append(self.preset_flow_classes[2])
        return output

    def id2color(self, ids):
        ids_np = np.array(ids).astype(int)
        if not ids_np.shape:
            return self.preset_flow_classes[ids_np].color
        ids_np_reshaped = ids_np.reshape(-1)
        colors = np.zeros_like(ids_np_reshaped, dtype='<U5')
        for i in range(ids_np.size):
            colors[i] = self.preset_flow_classes[ids_np_reshaped[i]].color
        return colors.reshape(ids_np.shape)


class CommunityDetector:
    def __init__(self, scenario_fps):
        self.all_ties = {}
        # algorithm thresholds
        self.max_distance = 5
        self.thre_length = 0.5
        self.thre_angle = np.pi / 8
        self.prev_t = int(scenario_fps * 0.5)  # 0.5s ago

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
                    if abs(d_tie_len) < self.thre_length and abs(d_tie_angle) < self.thre_angle and abs(d_orien) < np.pi / 4:
                        strong_ties.append(self.all_ties[IJ_idx][-1])
                        strong_ties_idx.append(IJ_idx)
                    else:
                        absent_ties.append(self.all_ties[IJ_idx][-1])
                        absent_ties_idx.append(IJ_idx)

        strong_ties_idx = np.array(strong_ties_idx)

        print('***************************')
        print(pids_t)
        print(strong_ties_idx.T)
        communities = connected_components(pids_t, strong_ties_idx)  # calc communities
        print(communities)

        return strong_ties, absent_ties, strong_ties_idx.tolist(), absent_ties_idx, communities


if __name__ == "__main__":
    import os
    from tqdm import trange
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    matplotlib.use('TkAgg')

    from toolkit.loaders.loader_hermes import load_bottleneck
    from toolkit.loaders.loader_metafile import load_eth, load_crowds
    from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir

    datasets = []
    OPENTRAJ_ROOT = "/home/cyrus/workspace2/OpenTraj"
    output_dir = "/home/cyrus/Dropbox/FollowBot/exp/pdf"
    save_links_dir = "/home/cyrus/Dropbox/FollowBot/exp/links"
    VISUALIZE = True

    # ======== load dataset =========
    # annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-075-075.txt')
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

        community_detector = CommunityDetector(scenario_fps=fps)
        all_ties = {}
        strong_ties = []
        absent_ties = []
        tie_strength = []



        # visualize stuff
        if VISUALIZE:
            fig, ax = plt.subplots()
            ax.set_aspect(aspect='equal')

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

            pids_t = frame_data["agent_id"].to_numpy()
            poss_t = frame_data[["pos_x", "pos_y"]].to_numpy()
            vels_t = frame_data[["vel_x", "vel_y"]].to_numpy()
            N = len(pids_t)

            strong_ties, absent_ties, strong_ties_idx, absent_ties_idx, communities =\
                community_detector.cluster_communities(pids_t, poss_t, vels_t)

            # Compute Average Velocity of each Community
            community_velocities = []
            for community in communities:
                vel_sum = np.zeros(2)
                for ii, pid in enumerate(community):
                    vel_sum += vels_t[pids_t.tolist().index(pid)]
                community_velocities.append(vel_sum/len(community))

            # assign territories
            agents_vel_polar = np.array([[np.linalg.norm(v), np.arctan2(v[1], v[0])] for v in vels_t])
            bgm = BivariateGaussianMixtureModel()
            for i in range(N):
                if norm(vels_t[i]) < 0.1:  # filter still agents
                    continue

                agent_flow_class = -1
                for jj, community in enumerate(communities):
                    if pids_t[i] in community:
                        agent_flow_class = jj
                        break

                bgm.add_component(BivariateGaussian(poss_t[i][0], poss_t[i][1],
                                                    sigma_x=agents_vel_polar[i][0] / 5 + 0.1, sigma_y=0.1,
                                                    theta=agents_vel_polar[i][1]), weight=1,
                                  # target=agents_flow_class[i].id)
                                  target=agent_flow_class)  # Todo: => the component labels

            mapResolution = 5
            xx, yy = np.meshgrid(np.arange(min_x, max_x, 1 / mapResolution),
                                 np.arange(min_y, max_y, 1 / mapResolution))
            crowd_flow_map_data = bgm.classify_kNN(xx, yy).T
            # =========================================


            # draw agents and ties
            if VISUALIZE:
                color_set = ['blue', 'red', 'green', 'gold', 'cyan', 'purple', 'pink', 'brown']

                def label2color(labels):
                    colors = np.zeros(labels.size, dtype='U11')
                    for i, l in enumerate(labels.reshape(-1).astype(int)):
                        colors[i] = color_set[l]
                    return colors.reshape(labels.shape)

                colors = label2color(crowd_flow_map_data).T

                plt.clf()
                plt.xlim([min_x, max_x])
                plt.ylim([min_y, max_y])
                plt.scatter(xx.reshape(-1), yy.reshape(-1), alpha=0.1, c=colors.reshape(-1))
                plt.title("%d" %t)

                for ii in range(N):
                    for jj in range(N):
                        id_i, id_j = pids_t[ii],  pids_t[jj]
                        if [id_i, id_j] in strong_ties_idx:
                            plt.plot([poss_t[ii, 0], poss_t[jj, 0]], [poss_t[ii, 1], poss_t[jj, 1]], 'k--', alpha=0.4)
                        else:
                            pass
                            # plt.plot([poss_t[ii, 0], poss_t[jj, 0]], [poss_t[ii, 1], poss_t[jj, 1]], 'g', alpha=0.2)

                for k, community in enumerate(communities):
                    color = color_set[k%len(color_set)]
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

