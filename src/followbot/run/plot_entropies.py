# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import os

from toolkit.loaders.loader_hermes import load_bottleneck
from toolkit.loaders.loader_metafile import load_eth, load_crowds
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk

from followbot.run.social_ties_stats import analyze

OPENTRAJ_ROOT = "/home/cyrus/workspace2/OpenTraj"
datasets = []

# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/ETH/seq_eth/obsmat.txt')
# datasets.append(load_eth(annot_file, title="ETH-Univ"))

annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/ETH/seq_hotel/obsmat.txt')
datasets.append(load_eth(annot_file, title="ETH-Hotel"))

# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/UCY/zara01/annotation.vsp')
# zara01 = load_crowds(annot_file, homog_file=os.path.join(OPENTRAJ_ROOT, "datasets/UCY/zara01/H.txt"), title="Zara01")
# datasets.append(zara01)

annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/UCY/zara02/annotation.vsp')
zara02 = load_crowds(annot_file, homog_file=os.path.join(OPENTRAJ_ROOT, "datasets/UCY/zara02/H.txt"), title="Zara02")
datasets.append(zara02)
# datasets.append(merge_datasets([zara01, zara02], new_title="Zara"))

# SDD datasets
# scenes = [['bookstore', 'video0'], ['bookstore', 'video1'], ['coupa', 'video0']]
# sdd_scales_yaml_file = os.path.join(OPENTRAJ_ROOT, 'datasets/SDD', 'estimated_scales.yaml')
# with open(sdd_scales_yaml_file, 'r') as f:
#     scales_yaml_content = yaml.load(f, Loader=yaml.FullLoader)
# for scene_i in scenes:
#     scale = scales_yaml_content[scene_i[0]][scene_i[1]]['scale']
#     sdd_dataset_i = load_sdd(os.path.join(OPENTRAJ_ROOT, "datasets/SDD", scene_i[0], scene_i[1], "annotations.txt"),
#                              scene_id="SDD-"+scene_i[0]+scene_i[1], title="SDD-"+scene_i[0]+"-"+scene_i[1][-1], # use_kalman=True,
#                              scale=scale, drop_lost_frames=False, sampling_rate=6)  # original fps=30
#     datasets.append(sdd_dataset_i)


# 1d HERMES
annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-1D/uo-180-180-180.txt')
# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-1D/uo-300-300-300.txt')
datasets.append(load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4]))

# 2d HERMES
# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-050-050.txt')
# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-160-160.txt')
# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-075-075.txt')
annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bo-360-090-090.txt')
# annot_file = os.path.join(OPENTRAJ_ROOT, 'datasets/HERMES/Corridor-2D/bot-360-250-250.txt')
datasets.append(load_bottleneck(annot_file, title="HERMES-" + os.path.basename(annot_file)[:-4]))


H_strong_dict = {}
H_absent_dict = {}
for dataset in datasets:
    H_strong, H_absent = analyze(dataset, verbose=False)
    H_strong[dataset.title] = H_strong
    H_absent[dataset.title] = H_absent
    
plt.bar(H_strong_dict.keys(), H_strong_dict.values())


