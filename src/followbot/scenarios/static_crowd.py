# Author: Javad Amirian
# Email: amiryan.j@gmail.com
from followbot.gui.display import *
from followbot.scenarios.scenario import Scenario
import numpy as np

from followbot.simulator.world import World
from followbot.util.basic_geometry import Line


class StaticCrowd(Scenario):
    def __init__(self, n_peds_=32, n_robots_=1, corridor_wid=5, corridor_len=18):
        super(StaticCrowd, self).__init__()
        self.world = []
        self.display = []
        self.n_peds = n_peds_
        self.n_robots = n_robots_
        self.corridor_wid = corridor_wid
        self.corridor_len = corridor_len
        self.ped_radius = 0.25

    def setup(self):
        world_dim = [[-0, 0], [self.corridor_len, self.corridor_wid]]
        self.world = World(self.n_peds, self.n_robots, biped=True)
        self.display = Display(self.world, world_dim, (960, 960), 'Static Crowd')

        # Two walls of the corridor
        line_objects = [Line([0, 0], [self.corridor_len, 0]),
                        Line([0, self.corridor_wid], [self.corridor_len, self.corridor_wid])]

        for l_obj in line_objects:
            self.world.add_object(l_obj)

        pom_resolution = 4  # per meter
        self.world.walkable = np.ones((self.corridor_len * pom_resolution,
                                       self.corridor_wid * pom_resolution), dtype=bool)
        self.world.POM = np.ones_like(self.world.walkable)

        self.world.mapping_to_grid = lambda x, y: (int(x * pom_resolution), int(y * pom_resolution))

        ped_poss = []

        for ped_ind in range(0, self.n_peds):
            px_i = np.random.uniform(-self.inner_dim + self.ped_radius, self.inner_dim - self.ped_radius)
            py_i = np.random.uniform(-self.outer_dim + self.ped_radius, -self.inner_dim - self.ped_radius)
            ped_poss.append([px_i, py_i])


        for ped_ind in range(len(ped_poss)):
            self.world.set_ped_position(ped_ind, ped_poss[ped_ind])
            self.world.set_ped_goal(ped_ind, ped_poss[ped_ind])
            self.world.set_ped_velocity(ped_ind, [0, 0])
            if ped_ind == 0: continue
            self.world.crowds[ped_ind].color = RED_COLOR

        # set the Robot position just behind first ped
        ped0_pos = self.world.crowds[0].pos
        self.world.set_robot_position(0, [ped0_pos[0] - 1.5, ped0_pos[1]])
        self.world.set_robot_leader(0, 0)
        self.world.sim.setTime(0)
        self.display.update()

