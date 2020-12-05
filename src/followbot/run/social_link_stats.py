# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import entropy
import matplotlib

from toolkit.core.trajdataset import merge_datasets

matplotlib.use('TkAgg')

from followbot.robot_functions.social_ties import SocialTiePDF
from toolkit.loaders.loader_hermes import load_bottleneck
from toolkit.loaders.loader_metafile import load_eth, load_crowds
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir

datasets = []
opentraj_root = '/home/cyrus/workspace2/OpenTraj'
output_dir = "/home/cyrus/Dropbox/FollowBot/exp/links"
# ======== load dataset =========
# annot_file = os.path.join(opentraj_root, 'datasets/ETH/seq_eth/obsmat.txt')
# datasets.append(load_eth(annot_file, title="ETH-Univ"))
#
# annot_file = os.path.join(opentraj_root, 'datasets/ETH/seq_hotel/obsmat.txt')
# datasets.append(load_eth(annot_file, title="ETH-Hotel"))
#
# annot_file = os.path.join(opentraj_root, 'datasets/UCY/zara01/annotation.vsp')
# zara01 = load_crowds(annot_file, homog_file=os.path.join(opentraj_root, "datasets/UCY/zara01/H.txt"), title="Zara01")
# datasets.append(zara01)

# annot_file = os.path.join(opentraj_root, 'datasets/UCY/zara02/annotation.vsp')
# zara02 = load_crowds(annot_file, homog_file=os.path.join(opentraj_root, "datasets/UCY/zara02/H.txt"), title="Zara02")
# datasets.append(zara02)
# datasets.append(merge_datasets([zara01, zara02], new_title="Zara"))

# SDD datasets
scenes = [['bookstore', 'video0'], ['bookstore', 'video1'], ['coupa', 'video0']]
sdd_scales_yaml_file = os.path.join(opentraj_root, 'datasets/SDD', 'estimated_scales.yaml')
with open(sdd_scales_yaml_file, 'r') as f:
    scales_yaml_content = yaml.load(f, Loader=yaml.FullLoader)
for scene_i in scenes:
    scale = scales_yaml_content[scene_i[0]][scene_i[1]]['scale']
    sdd_dataset_i = load_sdd(os.path.join(opentraj_root, "datasets/SDD", scene_i[0], scene_i[1], "annotations.txt"),
                             scene_id="SDD-"+scene_i[0]+scene_i[1], title="SDD-"+scene_i[0]+"-"+scene_i[1][-1], # use_kalman=True,
                             scale=scale, drop_lost_frames=False, sampling_rate=6)  # original fps=30
    datasets.append(sdd_dataset_i)


# 1d HERMES
# annot_file = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-1D/uo-180-180-180.txt')
# annot_file = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-1D/uo-300-300-300.txt')
# datasets.append(load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4]))

# 2d HERMES
# annot_file = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-2D/bo-360-050-050.txt')
# annot_file = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-2D/bo-360-160-160.txt')
# annot_file = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-2D/bo-360-075-075.txt')
# annot_file = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-2D/bot-360-250-250.txt')
# datasets.append(load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4]))
# -------------------------------


for dataset in datasets:
    min_x, max_x = min(dataset.data["pos_x"]), max(dataset.data["pos_x"])
    min_y, max_y = min(dataset.data["pos_y"]), max(dataset.data["pos_y"])
    # -------------------------------

    frames = dataset.get_frames()
    n_frames = len(frames)
    dt_ = np.diff(dataset.data["timestamp"].unique()[:2])[0]
    fps = int(round(1 / dt_))
    n_agents = dataset.data["agent_id"].nunique()

    all_ties = {}
    strong_ties = []
    absent_ties = []
    tie_strength = []

    # algorithm thresholds
    max_distance = 5
    thre_length = 0.3
    thre_angle = np.pi / 8

    # visualize stuff
    play_animation = False
    if play_animation:
        plt.figure(figsize=(5, 10))

    for t in tqdm.trange(len(frames)):
        frame_data = frames[t]

        pids_t = frame_data["agent_id"]
        N = len(pids_t)
        if N < 2:
            continue

        poss_t = frame_data[["pos_x", "pos_y"]].to_numpy()
        vels_t = frame_data[["vel_x", "vel_y"]].to_numpy()
        rot_matrices_t = np.zeros((N, 2, 2))
        oriens_t = np.arctan2(vels_t[:, 1], vels_t[:, 0])

        if play_animation:
            plt.cla()
            plt.xlim([min_x, max_x])
            plt.ylim([min_y, max_y])
            plt.scatter(poss_t[:, 0], poss_t[:, 1], color='blue', alpha=0.3)

        tiled_poss = np.tile(poss_t, (N, 1, 1))
        D = tiled_poss - tiled_poss.transpose((1, 0, 2))

        link_angles = np.arctan2(D[:, :, 1], D[:, :, 0])
        link_lengths = np.linalg.norm(D, axis=2)
        for ii, pid_i in enumerate(pids_t):
            for jj, pid_j in enumerate(pids_t):
                if ii == jj:
                    continue

                IJ_idx = (pid_i, pid_j)
                if IJ_idx not in all_ties:
                    all_ties[IJ_idx] = []
                oriented_link_angle = (link_angles[ii, jj] - oriens_t[ii] + np.pi) % (2 * np.pi) - np.pi
                all_ties[IJ_idx].append([link_lengths[ii, jj], oriented_link_angle])
                if len(all_ties[IJ_idx]) >= fps:
                    d_len = all_ties[IJ_idx][-1][0] - all_ties[IJ_idx][-fps][0]
                    d_angle = all_ties[IJ_idx][-1][1] - all_ties[IJ_idx][-fps][1]
                    if abs(d_len) < thre_length and abs(d_angle) < thre_angle \
                            and abs(oriens_t[ii] - oriens_t[jj]) < np.pi / 4 \
                            and all_ties[IJ_idx][-1][0] < max_distance:
                        strong_ties.append(all_ties[IJ_idx][-1])
                        if play_animation:
                            plt.plot([poss_t[ii, 0], poss_t[jj, 0]], [poss_t[ii, 1], poss_t[jj, 1]], 'g')
                    else:
                        absent_ties.append(all_ties[IJ_idx][-1])
                        if play_animation:
                            plt.plot([poss_t[ii, 0], poss_t[jj, 0]], [poss_t[ii, 1], poss_t[jj, 1]], 'r--', alpha=0.2)

        if play_animation:
            plt.pause(0.01)
        frame_id = frame_data["frame_id"].unique()
        # print("frame_id:", frame_id)

    # count the percent
    n_absents = len(absent_ties)
    n_strongs = len(strong_ties)
    print("*******************************")
    print("Dataset:", dataset.title)
    print("# total ties= [%d]" % (n_absents + n_strongs))
    print("absent ties= [%d] , %.3f" % (n_absents, n_absents / (n_absents + n_strongs + 1E-6)))
    print("strong ties= [%d] , %.3f" % (n_strongs, n_strongs / (n_absents + n_strongs + 1E-6)))
    print("*******************************")

    p = SocialTiePDF(max_distance, radial_resolution=4, angular_resolution=36)
    p.add_links(strong_ties)
    p.update_histogram(smooth=False)

    # Compute Histogram
    pk = p.polar_link_pdf.copy()
    # H_p = entropy(pk=pk.reshape(-1, 1)) / np.log(pk.size)
    # print("Entropy of Link Distribution(%) = ", H_p)

    # considering area of each bin
    area_k = np.tile(np.pi * np.diff(p.rho_edges ** 2) / (len(p.theta_edges) - 1), (p.polar_link_pdf.shape[1], 1)).T
    H_p_w = -np.nansum(np.log(pk / area_k) * pk)
    H_p_max = np.log(np.sum(area_k))
    print("Entropy of Link Distribution(%) = ", H_p_w / H_p_max)
    print("*******************************")
    # print(p.polar_link_pdf)
    p.plot(dataset.title)
    plt.savefig(os.path.join(output_dir, '%s-link-pdf.pdf' % dataset.title), dpi=500, bbox_inches='tight')
plt.show()
