import time

from bisect import bisect_left
import xml.dom.minidom as xmldom
import numpy as np
import yaml
import os

from OpenTraj.tools.parser.parser_eth import ParserETH
from followbot.basic_geometry import Line, Circle
from followbot.display import Display
from followbot.world import World
from followbot.display import RED_COLOR, BLUE_COLOR


class RealScenario:
    def __init__(self):
        self.world = []
        self.display = []
        self.line_objects = []
        self.circle_objects = []

        self.n_robots = 1
        self.leader_id = -1

        self.n_peds = 0  # will be read from dataset
        self.dataset = []
        self.ped_poss = []
        self.ped_vels = []

        self.frames = []
        self.cur_t = -1

    def setup(self,
              config_file='/home/cyrus/workspace2/ros-catkin/src/followbot/config/followbot_sim/real_scenario_eth.yaml'):
        # ===========================================
        # ============= Load config file ============
        with open(config_file) as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            annotation_file = config['Dataset']['Annotation']
            map_file = config['Dataset']['Map']
            self.leader_id = config['Dataset']['LeaderId']

        # ==================================
        # ========= Load dataset ===========
        if os.path.exists(annotation_file):
            self.dataset = ParserETH(annotation_file)
        self.frames = self.dataset.id_t_dict[self.leader_id]

        all_ids = set()
        for frame in self.frames:
            for id in self.dataset.t_id_dict[frame]:
                all_ids.add(id)

        all_ids = list(all_ids)
        all_ids.remove(self.leader_id)
        all_ids.insert(0, self.leader_id)  # the leader id should be at first index
        self.n_peds = len(all_ids)

        self.frames = [i for i in range(self.frames[0], self.frames[-1])]
        self.ped_poss = np.ones((len(self.frames), self.n_peds, 2)) * -100
        self.ped_vels = np.ones((len(self.frames), self.n_peds, 2)) * -100
        for tt, frame in enumerate(self.frames):
            for ii, id in enumerate(all_ids):
                if frame in self.dataset.id_t_dict[id]:
                    tt_ind = self.dataset.id_t_dict[id].tolist().index(frame)
                    self.ped_poss[tt, ii] = self.dataset.id_p_dict[id][tt_ind]
                    self.ped_vels[tt, ii] = self.dataset.id_v_dict[id][tt_ind]

                elif self.dataset.id_t_dict[id][0] < frame < self.dataset.id_t_dict[id][-1]:
                    p, v = self.interpolate(self.dataset, id, frame)  # TODO
                    self.ped_poss[tt, ii] = p
                    self.ped_vels[tt, ii] = v

        # ==================================
        # ========== Setup world ===========
        x_dim = self.dataset.max_x - self.dataset.min_x
        y_dim = self.dataset.max_y - self.dataset.min_y
        world_dim = [[self.dataset.min_x, self.dataset.max_x], [self.dataset.min_y, self.dataset.max_y]]

        self.world = World(self.n_peds, self.n_robots, 'helbing')
        self.display = Display(self.world, world_dim, (int(x_dim * 40), int(y_dim * 40)), 'Followbot - ETH')

        pom_resolution = 10  # per meter
        self.world.walkable = np.ones((int(x_dim * pom_resolution),
                                       int(y_dim * pom_resolution)), dtype=bool)
        # self.world.walkable[:, :] = 0

        self.world.mapping_to_grid = lambda x, y: (int((x - self.dataset.min_x) * pom_resolution),
                                                   int((y - self.dataset.min_y) * pom_resolution))

        leader_init_pos = self.dataset.id_p_dict[self.leader_id][0]
        leader_init_vel = self.dataset.id_v_dict[self.leader_id][0]
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
                self.line_objects.append(line_obj)
                self.world.add_object(self.line_objects[-1])

            # TODO: modify the XML file
            circle_elems = map_doc.getElementsByTagName('Circle')
            for circle_elem in circle_elems:
                x = circle_elem.getAttribute('x')
                y = circle_elem.getAttribute('y')
                rad = circle_elem.getAttribute('radius')
                self.circle_objects.append(Circle([x, y], rad))
                self.world.add_object(self.circle_objects[-1])

    # TODO: to interpolate dataset points (not enough for tracking)
    def interpolate(self, dataset, id, frame):
        ts_id = dataset.id_t_dict[id]
        left_ind = bisect_left(ts_id, frame) -1
        right_ind = left_ind + 1
        alpha = (frame - ts_id[left_ind]) / (ts_id[right_ind] - ts_id[left_ind])
        pos = dataset.id_p_dict[id][left_ind] * (1-alpha) + dataset.id_p_dict[id][right_ind] * alpha
        vel = dataset.id_v_dict[id][left_ind] * (1-alpha) + dataset.id_v_dict[id][right_ind] * alpha
        return pos, vel

    def step(self, save=False):
        if not self.world.pause and self.cur_t < len(self.frames):
            self.cur_t += 1
            for ii in range(self.n_peds):
                self.world.set_ped_position(ii, self.ped_poss[self.cur_t, ii])
                self.world.set_ped_velocity(ii, self.ped_vels[self.cur_t, ii])
            self.world.step_robot(0.05)

        toggle_pause = self.display.update()
        if toggle_pause: self.world.pause = not self.world.pause

        time.sleep(0.2)
        if not self.world.pause and save:
            self.display.save('/home/cyrus/Videos/crowdsim/eth/')


if __name__ == '__main__':
    scenario = RealScenario()
    scenario.setup()
