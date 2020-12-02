# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import re
import yaml
import numpy as np

from followbot.util.basic_geometry import Line
from followbot.scenarios.real_scenario import RealScenario
from toolkit.core.trajdataset import TrajDataset
from toolkit.loaders.loader_hermes import load_bottleneck


class HermesScenario(RealScenario):
    """
    Class for replaying Hermes (Bottleneck) Crowd Experiments
    """
    def __init__(self):
        super(HermesScenario, self).__init__()

    def setup_with_config_file(self, config_file):
        with open(config_file) as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            biped_mode = config['General']['biped']
            # opentraj_root = config['Dataset']['OpenTrajRoot']
            robot_replacement_id = config['Dataset']['RobotId']
            obstacles = config['Dataset']['Map']
            fps = config['Dataset']['fps']
            annotation_file = config['Dataset']['Annotation']
            dataset = load_bottleneck(annotation_file)

        self.setup(dataset=dataset, fps=fps, robot_id=robot_replacement_id, obstacles=obstacles, biped=biped_mode)

    def setup(self, **kwargs):
        self.dataset = kwargs.get("dataset", None)
        self.fps = kwargs.get("fps", 16)
        self.robot_replacement_id = kwargs.get("robot_id", -1)
        biped = kwargs.get("biped", False)

        # rotate 90 degree
        transform = np.array([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, 1]])
        self.dataset.apply_transformation(transform, inplace=True)
        self.create_sim_frames(biped=biped)

        # exp_dimensions = re.split('-|\.', annotation_file)[-4:-1]
        obstacles = kwargs.get("obstacles", [])
        for obs in obstacles:
            self.world.add_obstacle(Line(obs[0:2], obs[2:4]))

    # if '2D' in annotation_file:
        #     line_objs = corridor_map(int(exp_dimensions[0]) / 100., int(exp_dimensions[0]) / 100.)
        # else:
        #     line_objs = corridor_map(int(exp_dimensions[1]) / 100., int(exp_dimensions[2]) / 100.)
        # for line_obj in line_objs:
        #     self.world.add_obstacle(line_obj)


def corridor_map(width, bottleneck):
    wall_b = Line((-4, 0), (4, 0))
    wall_t = Line((-4, width), (4, width))
    bottleneck_b = Line((4, -1), (4, (width - bottleneck)/2.))
    bottleneck_t = Line((4, width+1), (4, (width + bottleneck) / 2.))
    stand_b = Line((-4, 0), (-4, -1))
    stand_t = Line((-4, width), (-4, width+1))
    lines = [wall_b, wall_t, bottleneck_b, bottleneck_t]
    return lines


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    scenario = HermesScenario()
    conf_file = "/home/cyrus/workspace2/ros-catkin/src/followbot/config/followbot_sim/real_scenario_config.yaml"
    scenario.setup_with_config_file(conf_file)
    [xdim, ydim] = scenario.world.world_dim

    pause = np.zeros(1, dtype=int)
    t = -1
    while t < len(scenario.frames):
        plt.cla()
        plt.xlim(xdim)
        plt.ylim(ydim)
        ped_poss_t = scenario.ped_poss[t]
        ped_poss_t = ped_poss_t[ped_poss_t[:, 0] > -100]
        robot_pos_t = scenario.robot_poss[t]

        plt.scatter(ped_poss_t[:, 0], ped_poss_t[:, 1], color='g')
        plt.scatter(robot_pos_t[0], robot_pos_t[1], color='r')

        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [pause.fill(1-pause[0]) if event.key == ' ' else None])

        if not pause[0]:
            t += 1
        plt.pause(0.01)


    plt.show()
