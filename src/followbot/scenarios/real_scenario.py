import xml.dom.minidom as xmldom
import numpy as np
import yaml
import os

from followbot.scenarios.scenario import Scenario
from followbot.util.basic_geometry import Line, Circle
from followbot.scenarios.world import World
from followbot.gui.visualizer import BLUE_COLOR
from followbot.gui.visualizer import Visualizer

# make sure opentraj is there
from toolkit.core.trajdataset import TrajDataset
from toolkit.loaders.loader_metafile import load_metafile


class RealScenario(Scenario):
    def step_crowd(self, dt):
        raise Exception("TODO")

    def __init__(self):
        super(RealScenario, self).__init__()
        # Replay Agents data
        self.dataset = None  # -> TrajDataset
        self.frames = []
        self.fps = 1

        self.ped_poss = []
        self.ped_vels = []
        self.ped_valid = []

        self.robot_poss = []
        self.robot_vels = []

        # The pedestrian with this id will be replaced by robot
        self.robot_replacement_id = -1


    def setup(self, **kwargs):
        self.dataset = kwargs.get("dataset", None)
        self.fps = kwargs.get("fps", 16)
        self.robot_replacement_id = kwargs.get("robot_id", -1)
        biped = kwargs.get("biped", False)

    # def setup(self, config_file, biped):
    #     # ===========================================
    #     # ============= Load config file ============
    #     with open(config_file) as stream:
    #         config = yaml.load(stream, Loader=yaml.FullLoader)
    #         dataset_metafile = config['Dataset']['Metafile']
    #         opentraj_root = config['Dataset']['OpenTrajRoot']
    #         # annotation_file = config['Dataset']['Annotation']
    #         # parser_type = config['Dataset']['Parser']
    #         map_file = config['Dataset']['Map']
    #         self.robot_replacement_id = config['Dataset']['HumanId']
    #         # biped = config['General']['biped']
    #         display_resolution = config['General']['resolution_dpm']
    #         # ========= Load dataset ===========
    #         # self.dataset = load_metafile(opentraj_root, dataset_metafile)

        # if not os.path.exists(dataset_metafile):
        #     raise ValueError('Error! Annotation file does not exist')

        # self.dataset = load_metafile(opentraj_root, metafile=dataset_metafile)

        # if kwargs.get("map_file"):
        self.create_sim_frames(biped=biped, map_file="")

    def create_sim_frames(self, **kwargs):
        map_file = kwargs.get("map_file", "")
        biped = kwargs.get("biped", False)

        # get frames in which the leader is present.
        self.frames = self.dataset.data['frame_id'].loc[
            self.dataset.data['agent_id'] == self.robot_replacement_id].tolist()
        self.frames = list(range(self.frames[0], self.frames[-1]+1))

        # get other agents that are present in selected frames
        all_ids = self.dataset.data['agent_id'].loc[self.dataset.data['frame_id'].isin(self.frames)].unique().tolist()
        all_ids.remove(self.robot_replacement_id)
        self.n_peds = len(all_ids)

        self.ped_poss = np.ones((len(self.frames), self.n_peds, 2)) * -100
        self.ped_vels = np.zeros((len(self.frames), self.n_peds, 2))
        self.ped_valid = np.zeros((len(self.frames), self.n_peds), dtype=bool)

        for ii, pid in enumerate(all_ids):
            data_pid = self.dataset.data[['pos_x', 'pos_y', 'vel_x', 'vel_y', 'frame_id']] \
                .loc[(self.dataset.data['agent_id'] == pid)
                   & (self.dataset.data['frame_id'] >= self.frames[0])
                   & (self.dataset.data['frame_id'] <= self.frames[-1])
                     ].to_numpy()
            frames_pid = data_pid[:, 4].astype(int)
            self.ped_poss[frames_pid[0] - self.frames[0]:frames_pid[-1] - self.frames[0] + 1, ii] = data_pid[:, :2]
            self.ped_vels[frames_pid[0] - self.frames[0]:frames_pid[-1] - self.frames[0] + 1, ii] = data_pid[:, 2:4]
            self.ped_valid[frames_pid[0] - self.frames[0]:frames_pid[-1] - self.frames[0] + 1, ii] = True

        self.robot_poss = self.dataset.data[['pos_x', 'pos_y']].loc[
            (self.dataset.data['agent_id'] == self.robot_replacement_id)].to_numpy()
        self.robot_vels = self.dataset.data[['vel_x', 'vel_y']].loc[
            (self.dataset.data['agent_id'] == self.robot_replacement_id)].to_numpy()


        # ==================================
        # ========== Setup world ===========
        x_min = min(self.dataset.data["pos_x"])
        x_max = max(self.dataset.data["pos_x"])
        y_min = min(self.dataset.data["pos_y"])
        y_max = max(self.dataset.data["pos_y"])
        x_dim = x_max - x_min
        y_dim = y_max - y_min
        world_dim = [[x_min, x_max], [y_min, y_max]]

        self.world = World(self.n_peds, self.n_robots, world_dim, "", biped)

        robot_state_t0 = self.dataset.data[['pos_x', 'pos_y', 'vel_x', 'vel_y']] \
            .loc[self.dataset.data['agent_id'] == self.robot_replacement_id].to_numpy()
        robot_init_pos = robot_state_t0[0, :2]
        robot_init_vel = robot_state_t0[0, 2:]
        # follow_distance = 0.2

        # robot_init_pos = robot_init_pos - follow_distance * robot_init_vel / (np.linalg.norm(robot_init_vel) + 1E-6)
        # self.world.set_robot_position(0, robot_init_pos)
        # self.world.set_robot_velocity(0, [0, 0])
        # self.world.set_robot_leader(0, 0)

        for ped_ind in range(self.n_peds):
            self.world.set_ped_position(ped_ind, self.ped_poss[0, ped_ind])
            self.world.set_ped_goal(ped_ind, self.ped_poss[0, ped_ind])
            self.world.set_ped_velocity(ped_ind, [0, 0])
            self.world.crowds[ped_ind].color = BLUE_COLOR

        if os.path.exists(map_file):
            map_doc = xmldom.parse(map_file)
            line_elems = map_doc.getElementsByTagName('Line')
            for line_elem in line_elems:
                x1 = line_elem.getAttribute('x1')
                y1 = line_elem.getAttribute('y1')
                x2 = line_elem.getAttribute('x2')
                y2 = line_elem.getAttribute('y2')
                line_obj = Line([x1, y1], [x2, y2])
                # self.line_objects.append(line_obj)
                self.world.add_obstacle(line_obj)

            # TODO: modify the XML file
            circle_elems = map_doc.getElementsByTagName('Circle')
            for circle_elem in circle_elems:
                x = float(circle_elem.getAttribute('x'))
                y = float(circle_elem.getAttribute('y'))
                rad = float(circle_elem.getAttribute('radius'))
                # self.circle_objects.append(Circle([x, y], rad))
                self.world.add_obstacle(Circle([x, y], rad))

        self.world.set_time(0)

    # def interpolate(self, dataset, id, frame):
    #     ts_id = dataset.id_t_dict[id]
    #     left_ind = bisect_left(ts_id, frame) -1
    #     right_ind = left_ind + 1
    #     alpha = (frame - ts_id[left_ind]) / (ts_id[right_ind] - ts_id[left_ind])
    #     pos = dataset.id_p_dict[id][left_ind] * (1-alpha) + dataset.id_p_dict[id][right_ind] * alpha
    #     vel = dataset.id_v_dict[id][left_ind] * (1-alpha) + dataset.id_v_dict[id][right_ind] * alpha
    #     return pos, vel

    def step(self, dt, lidar_enabled, save=False):
        not_finished = self.cur_t < len(self.frames) - 1
        if not self.world.pause and not_finished:
            new_t = int(self.cur_t + 1)
            self.cur_t = new_t
            self.world.set_time(new_t)

            for ii in range(self.n_peds):
                self.world.set_ped_position(ii, self.ped_poss[new_t, ii])
                self.world.set_ped_velocity(ii, self.ped_vels[new_t, ii])

                self.world.crowds[ii].step(dt)
            self.step_robots(dt, lidar_enabled)

        super(RealScenario, self).step(save, lidar_enabled)
        return not_finished


if __name__ == '__main__':
    scenario = RealScenario()
    try:
        scenario.setup()
        print("Real scenario was set up successfully!")
    except Exception:
        raise Exception("Failed to create the Real Scenario object.")
