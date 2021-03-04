# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

from followbot.util.eval import Evaluation

matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk

evaluator = Evaluation()
results_dir = '/home/cyrus/Music/followbot-outputs'

results_files = sorted(glob.glob(results_dir + '/*.npz'))
print(results_files)


for result_file in results_files:
    result_np = np.load(result_file)
    robot_loc = result_np['robot_loc']
    detections = result_np['detections']
    gt_locs = result_np['gt_locs']
    robot_hypo = result_np['robot_hypo']
    x_range = result_np['x_range']
    y_range = result_np['y_range']

    if not len(robot_hypo):
        continue

    xx, yy = np.meshgrid(x_range, y_range)

    _, _, gt_pom, predicted_pom = evaluator.calc_error((xx, yy), gt_locs, robot_hypo[0])

    fig = plt.figure(figsize=(7, 5))
    grid_spec = gridspec.GridSpec(2, 1)
    gt_map = plt.subplot(grid_spec[0, 0])
    eval_map = plt.subplot(grid_spec[1, 0])

    gt_map.axis("off")
    gt_map.set_title("Projections")
    gt_map.imshow(np.flipud(predicted_pom.T))
    eval_map.axis("off")
    eval_map.set_title("Ground Truth")
    eval_map.imshow(np.flipud(gt_pom.T))
    # plt.savefig(os.path.join("/home/cyrus/Videos/followbot/projections", "%5d.jpg" % frame_id))

    plt.show()
