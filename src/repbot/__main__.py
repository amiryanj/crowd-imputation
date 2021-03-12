# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from repbot.config import *
from repbot.run.repbot import exec_scenario
from repbot.crowd_imputation.crowd_synthesis import CrowdSynthesizer
from repbot.scenarios.human_traj_scenario import HumanTrajectoryScenario
from repbot.util.read_lidar_data import read_lidar_data
from repbot.util.video_player import DatasetVideoPlayer

if __name__ == "__main__":
    temp_scenario = HumanTrajectoryScenario()
    temp_scenario.setup_with_config_file(SCENARIO_CONFIG_FILE)
    all_agent_ids = temp_scenario.dataset.get_agent_ids()
    test_agent_ids = range(int(0.7 * len(all_agent_ids)), len(all_agent_ids))
    # test_agent_ids = sorted(set(test_agent_ids).intersection(range(335, 10000)))

    pcf_crowd_rec = CrowdSynthesizer()
    pcf_crowd_rec.extract_features(temp_scenario.dataset)

    for agent_idx in test_agent_ids:
        robot_id = all_agent_ids[agent_idx]
        _scenario = HumanTrajectoryScenario()
        _scenario.setup(dataset=temp_scenario.dataset, fps=temp_scenario.fps, title=temp_scenario.title,
                        # obstacles=[], world_boundary=[],
                        robot_id=robot_id, biped_mode=BIPED_MODE)

        _scenario.title = _scenario.dataset.title

        if READ_LIDAR_FROM_CARLA:  # FixMe: make sure the robotId has been the same
            lidar_data = read_lidar_data()
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        # =======================================
        if VIDEO_ENABLED:
            video_player = DatasetVideoPlayer(_scenario.video_files)
            video_player.set_frame_id(_scenario.world.original_frame_id)
        print(type(_scenario).__name__)
        for exec_frame_id, robot in exec_scenario(_scenario):
            print("agent [%d/%d] frame= [%d(%d)/%d]" % (agent_idx, len(all_agent_ids),
                                                        _scenario.world.frame_id, exec_frame_id,
                                                        len(_scenario.ped_poss)))

            if VIDEO_ENABLED:
                # if exec_frame != 0:
                #     video_player.set_frame_id(exec_frame)
                im = video_player.get_frame().__next__()
                if im is not None:
                    cv2.imshow("im", im)
                else:
                    print("problem with video")
                cv2.waitKey(2)

            if READ_LIDAR_FROM_CARLA and exec_frame_id in lidar_data:
                lidar_data[exec_frame_id][:, 1] *= -1
                lidar_data[exec_frame_id][:, 2] = 1
                robot_rot = Rotation.from_euler('z', robot.orien, degrees=False).as_matrix()
                robot_tf = np.hstack([robot_rot[:2, :2], robot.pos.reshape(2, 1)])  # 2 x 3
                lidar_data[exec_frame_id][:, :2] = np.matmul(lidar_data[exec_frame_id][:, :3], robot_tf.T)

                plt.cla()
                ax.set_xlim([-8, 8])
                ax.set_ylim([-0, 4])
                # ax.set_zlim([0, 5])
                ax.scatter(lidar_data[exec_frame_id][:, 0],  # x
                           lidar_data[exec_frame_id][:, 1],  # y
                           # scan[data_range, 2],  # z
                           c=lidar_data[exec_frame_id][:, 3],  # reflectance
                           s=2, cmap='viridis')
                ax.plot(robot.pos[0], robot.pos[1], 'ro')
                ax.set_title('Lidar scan %s' % exec_frame_id)
                plt.grid()
                plt.pause(0.01)
