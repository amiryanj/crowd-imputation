# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

from followbot.util.eval import Evaluation
from followbot.util.mapped_array import MappedArray

matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk

evaluator = Evaluation()
results_dir = '/home/cyrus/Music/followbot-outputs-5/ETH-Univ'

results_files = sorted(glob.glob(results_dir + '*.npz'))
print(results_files)

for result_file in results_files:
    print('Reading [%s]' % result_file)
    try:
        result_np = np.load(result_file)
        robot_loc = result_np['robot_loc']
        detections = result_np['detections']
        gt_locs = result_np['gt_locs']
        preds_ours = result_np['preds_ours']
        preds_pcf = result_np['preds_pcf']
        x_range = result_np['x_range']
        y_range = result_np['y_range']

        dx = (x_range[1] - x_range[0])/8
        dy = (y_range[1] - y_range[0])/8
        x_range = np.arange(x_range[0] - 1, x_range[-1] + dx + 1, dx)
        y_range = np.arange(y_range[0] - 1.75, y_range[-1] + dy + 1, dy)
    except:
        continue

    if not len(preds_ours):
        continue

    xx, yy = np.meshgrid(x_range, y_range)

    _, _, gt_pom, predicted_pom = evaluator.calc_error((xx, yy), gt_locs, list(preds_ours) + list(detections))

    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    grid_spec = gridspec.GridSpec(2, 1)
    ours_map = plt.subplot(grid_spec[0, 0])
    gt_map = plt.subplot(grid_spec[1, 0])

    ours_map.axis("off")
    # ours_map.set_title("Imputation Result")

    gt_map.axis("off")
    # gt_map.set_title("Ground Truth")

    # cmap = 'viridis'
    cmap = 'Blues'
    ours_map.imshow(predicted_pom.T, cmap=cmap)
    gt_map.imshow(gt_pom.T, cmap=cmap)
    # plt.savefig(os.path.join("/home/cyrus/Videos/followbot/projections", "%5d.jpg" % frame_id))

    map = MappedArray(x_range[0], x_range[-1], y_range[0], y_range[-1], 1/dx)

    gt_locs_uv = np.array([map.map(p[0], p[1]) for p in gt_locs])
    gt_map.scatter(gt_locs_uv[:, 0], gt_locs_uv[:, 1],
                   s=60, facecolors='none', edgecolors='greenyellow', linewidths=2, label='Ground-Truth')

    detection_locs_uv = np.array([map.map(p[0], p[1]) for p in detections])
    ours_map.scatter(detection_locs_uv[:, 0], detection_locs_uv[:, 1],
                     s=80, facecolors='none', edgecolors='b', linewidths=2, label='Detections')

    our_preds_locs_uv = np.array([map.map(p[0], p[1]) for p in preds_ours])
    ours_map.scatter(our_preds_locs_uv[:, 0], our_preds_locs_uv[:, 1],
                     s=80, facecolors='none', edgecolors='r', linewidths=2, label='Imputations')

    robot_loc_uv = map.map(robot_loc[0], robot_loc[1])
    gt_map.scatter(robot_loc_uv[0], robot_loc_uv[1],
                   s=80, facecolors='pink', edgecolors='m', linewidths=2, label='Robot')
    ours_map.scatter(robot_loc_uv[0], robot_loc_uv[1],
                     s=80, facecolors='pink', edgecolors='m', linewidths=2, label='Robot')

    gt_map.legend(loc="upper right", ncol=1)
    ours_map.legend(loc="upper right", ncol=1)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    print()
    plt.savefig(os.path.join("/home/cyrus/Music/iros-eth-results", result_file[-13:-6] + ".pdf"))
    plt.savefig(os.path.join("/home/cyrus/Music/iros-eth-results", result_file[-13:-6] + ".png"), dpi=100)
    # plt.show()
