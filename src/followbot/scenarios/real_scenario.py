import xml.dom.minidom as xmldom
import numpy as np
import yaml
import os
from followbot.scenarios.scenario import Scenario

from toolkit.loaders.loader_metafile import load_metafile
from followbot.util.basic_geometry import Line, Circle
from followbot.gui.visualizer import Visualizer
from followbot.simulator.world import World
from followbot.gui.visualizer import BLUE_COLOR


class RealScenario(Scenario):
    def __init__(self):
        super(RealScenario, self).__init__()
        self.dataset = []
        self.ped_poss = []
        self.ped_vels = []
        self.ped_valid = []

        self.frames = []

    def setup(self,
              config_file='/home/cyrus/workspace2/ros-catkin/src/followbot/config/followbot_sim/real_scenario_config.yaml',
              biped=False):
        # ===========================================
        # ============= Load config file ============
        with open(config_file) as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            dataset_metafile = config['Dataset']['Metafile']
            opentraj_root = config['Dataset']['OpenTrajRoot']
            # annotation_file = config['Dataset']['Annotation']
            # parser_type = config['Dataset']['Parser']
            map_file = config['Dataset']['Map']
            self.leader_id = config['Dataset']['LeaderId']
            # biped = config['General']['biped']
            display_resolution = config['General']['resolution_dpm']

        # ==================================
        # ========= Load dataset ===========
        if not os.path.exists(dataset_metafile):
            print('Error! annotaion file does not exist')
            return

        self.dataset = load_metafile(opentraj_root, dataset_metafile)
        self.dataset.interpolate_frames()

        # get frame(pid)s where leader is present.
        self.frames = self.dataset.data['frame_id'].loc[self.dataset.data['agent_id'] == self.leader_id].tolist()
        # self.frames = self.dataset.id_t_dict[self.leader_id]  # old

        # get other agents that are present in selected frames
        all_ids = self.dataset.data['agent_id'].loc[self.dataset.data['frame_id'].isin(self.frames)].unique().tolist()

        all_ids.remove(self.leader_id)
        all_ids.insert(0, self.leader_id)  # the leader pid should be at first index
        self.n_peds = len(all_ids)

        self.frames = [i for i in range(self.frames[0], self.frames[-1])]
        self.ped_poss = np.ones((len(self.frames), self.n_peds, 2)) * -100
        self.ped_vels = np.ones((len(self.frames), self.n_peds, 2)) * -100
        self.ped_valid = np.zeros((len(self.frames), self.n_peds), dtype=bool)

        for tt, frame in enumerate(self.frames):
            for ii, pid in enumerate(all_ids):
                data_pid_t = self.dataset.data[['pos_x', 'pos_y', 'vel_x', 'vel_y']]\
                    .loc[(self.dataset.data['frame_id'] == frame) & (self.dataset.data['agent_id'] == pid)].to_numpy()
                if len(data_pid_t):
                    self.ped_poss[tt, ii] = data_pid_t[0, :2]
                    self.ped_vels[tt, ii] = data_pid_t[0, 2:]
                    self.ped_valid[tt, ii] = True

        # ==================================
        # ========== Setup world ===========
        x_dim = self.dataset.bbox['x']['max'] - self.dataset.bbox['x']['min']
        y_dim = self.dataset.bbox['y']['max'] - self.dataset.bbox['y']['min']
        world_dim = [[self.dataset.bbox['x']['min'], self.dataset.bbox['x']['max']],
                     [self.dataset.bbox['y']['min'], self.dataset.bbox['y']['max']]]

        self.world = World(self.n_peds, self.n_robots, 'helbing', biped)
        self.visualizer = Visualizer(self.world, world_dim,
                                     (int(x_dim * display_resolution), int(y_dim * display_resolution)),
                                     self.dataset.title)

        pom_resolution = 10  # per meter
        self.world.walkable = np.ones((int(x_dim * pom_resolution),
                                       int(y_dim * pom_resolution)), dtype=bool)
        self.world.POM = self.world.walkable.copy() * 0.5

        self.world.mapping_to_grid = lambda x, y: (int((x - self.dataset.bbox['x']['min']) * pom_resolution),
                                                   int((y - self.dataset.bbox['y']['min']) * pom_resolution))

        data_leader_t0 = self.dataset.data[['pos_x', 'pos_y', 'vel_x', 'vel_y']] \
            .loc[self.dataset.data['agent_id'] == pid].to_numpy()
        leader_init_pos = data_leader_t0[0, :2]
        leader_init_vel = data_leader_t0[0, 2:]
        follow_distance = 0.2
        robot_init_pos = leader_init_pos - follow_distance * leader_init_vel / (np.linalg.norm(leader_init_vel) + 1E-6)

        self.world.set_robot_position(0, robot_init_pos)
        self.world.set_robot_velocity(0, [0, 0])
        self.world.set_robot_leader(0, 0)

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

    # def interpolate(self, dataset, id, frame):
    #     ts_id = dataset.id_t_dict[id]
    #     left_ind = bisect_left(ts_id, frame) -1
    #     right_ind = left_ind + 1
    #     alpha = (frame - ts_id[left_ind]) / (ts_id[right_ind] - ts_id[left_ind])
    #     pos = dataset.id_p_dict[id][left_ind] * (1-alpha) + dataset.id_p_dict[id][right_ind] * alpha
    #     vel = dataset.id_v_dict[id][left_ind] * (1-alpha) + dataset.id_v_dict[id][right_ind] * alpha
    #     return pos, vel


    def step(self, save=False):
        # def step_crowd(self):
        if not self.world.pause and self.cur_t < len(self.frames) - 1:
            self.cur_t += 1
            for ii in range(self.n_peds):
                self.world.set_ped_position(ii, self.ped_poss[self.cur_t, ii])
                self.world.set_ped_velocity(ii, self.ped_vels[self.cur_t, ii])

        # def step_robot(self):
        if not self.world.pause and self.cur_t < len(self.frames) - 1:
            self.world.step_robot(0.04)

        super(RealScenario, self).step(save)


if __name__ == '__main__':
    scenario = RealScenario()
    scenario.setup()
    print("Real scenario was set up successfully!")
