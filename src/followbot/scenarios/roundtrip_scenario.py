import time

from followbot.scenarios.scenario import Scenario
from followbot.scenarios.simulation_scenario import SimulationScenario
from followbot.scenarios.world import World
from followbot.gui.visualizer import *


class RoundTrip(SimulationScenario):
    def __init__(self, n_peds_=32, n_robots_=1, inner_dim_=12, outer_dim_=18):
        super(RoundTrip, self).__init__()
        self.world = []
        self.visualizer = []
        self.n_peds = n_peds_
        self.n_robots = n_robots_
        self.inner_dim = inner_dim_
        self.outer_dim = outer_dim_
        self.ped_radius = 0.25

    def setup(self, sim_model, flow_2d=True):  # 1D flow / 2D flow
        k_in_each_corridor = self.n_peds // 4
        world_dim = [[-self.outer_dim, self.outer_dim], [-self.outer_dim, self.outer_dim]]
        self.world = World(self.n_peds, self.n_robots, sim_model, biped=True)
        # world.pref_speed = 1.5  # FIXME : set it for sim as well

        # NOTE Symmetric with center at (0, 0)
        line_objects = [Line([-self.inner_dim, -self.inner_dim], [self.inner_dim, -self.inner_dim]),
                        Line([self.inner_dim, -self.inner_dim], [self.inner_dim, self.inner_dim]),
                        Line([self.inner_dim, self.inner_dim], [-self.inner_dim, self.inner_dim]),
                        Line([-self.inner_dim, self.inner_dim], [-self.inner_dim, -self.inner_dim]),

                        Line([-self.outer_dim, -self.outer_dim], [self.outer_dim, -self.outer_dim]),
                        Line([self.outer_dim, -self.outer_dim], [self.outer_dim, self.outer_dim]),
                        Line([self.outer_dim, self.outer_dim], [-self.outer_dim, self.outer_dim]),
                        Line([-self.outer_dim, self.outer_dim], [-self.outer_dim, -self.outer_dim]) ]
        for l_obj in line_objects:
            self.world.add_obstacle(l_obj)

        pom_resolution = 4  # per meter
        self.world.walkable = np.ones((self.outer_dim * 2 * pom_resolution,
                                       self.outer_dim * 2 * pom_resolution), dtype=bool)
        self.world.occupancy_map = np.zeros_like(self.world.walkable)
        self.world.walkable[(-self.inner_dim + self.outer_dim) * pom_resolution:
                            (self.inner_dim + self.outer_dim) * pom_resolution,
                            (-self.inner_dim + self.outer_dim) * pom_resolution:
                            (self.inner_dim + self.outer_dim) * pom_resolution] = 0

        self.world.mapping_to_grid = lambda x, y: (int(((x + self.outer_dim) * pom_resolution)),
                                                   int(((y + self.outer_dim) * pom_resolution)))

        ped_poss = []
        # Region A
        for ped_ind in range(0, k_in_each_corridor):
            px_i = np.random.uniform(-self.inner_dim + self.ped_radius, self.inner_dim - self.ped_radius)
            py_i = np.random.uniform(-self.outer_dim + self.ped_radius, -self.inner_dim - self.ped_radius)
            ped_poss.append([px_i, py_i])

        # Region B
        for ped_ind in range(k_in_each_corridor, k_in_each_corridor * 2):
            px_i = np.random.uniform(self.inner_dim + self.ped_radius, self.outer_dim - self.ped_radius)
            py_i = np.random.uniform(-self.inner_dim + self.ped_radius, self.inner_dim - self.ped_radius)
            ped_poss.append([px_i, py_i])

        # Region C
        for ped_ind in range(k_in_each_corridor * 2, k_in_each_corridor * 3):
            px_i = np.random.uniform(-self.inner_dim + self.ped_radius, self.inner_dim - self.ped_radius)
            py_i = np.random.uniform(self.inner_dim + self.ped_radius, self.outer_dim - self.ped_radius)
            ped_poss.append([px_i, py_i])

        # Region D
        for ped_ind in range(k_in_each_corridor * 3, k_in_each_corridor * 4):
            px_i = np.random.uniform(-self.outer_dim + self.ped_radius, -self.inner_dim - self.ped_radius)
            py_i = np.random.uniform(-self.inner_dim + self.ped_radius, self.inner_dim - self.ped_radius)
            ped_poss.append([px_i, py_i])

        for ped_ind in range(len(ped_poss)):
            self.world.set_ped_position(ped_ind, ped_poss[ped_ind])
            self.world.set_ped_goal(ped_ind, ped_poss[ped_ind])
            self.world.set_ped_velocity(ped_ind, [0, 0])
            self.world.crowds[ped_ind].ccw = False

            if ped_ind == 0 or not flow_2d: continue
            if np.random.rand() > 0.5:
                self.world.crowds[ped_ind].ccw = True
                self.world.crowds[ped_ind].color = BLUE_COLOR
            else:
                self.world.crowds[ped_ind].ccw = False
                self.world.crowds[ped_ind].color = RED_COLOR

        # set the Robot position just behind first ped
        ped0_pos = self.world.crowds[0].pos
        self.world.set_robot_position(0, [ped0_pos[0] - 1.5, ped0_pos[1]])
        self.world.set_robot_leader(0, 0)
        self.world.set_time(0)

    def set_goals_roundtrip(self, ped):
        goal = []

        x, y = ped.pos[0], ped.pos[1]
        RADIUS = ped.radius
        THRESH = self.inner_dim + ped.radius
        MIDDLE = (self.inner_dim + self.outer_dim) / 2

        # Region A
        if ped.ccw and x < THRESH and y < -THRESH:
            goal = [MIDDLE, -THRESH-RADIUS]
        elif not ped.ccw and x > -THRESH and y < -THRESH:
            goal = [-MIDDLE, -THRESH-RADIUS]

        # Region B
        if ped.ccw and x > THRESH and y < THRESH:
            goal = [THRESH+RADIUS, MIDDLE]
        elif not ped.ccw and x > THRESH and y > -THRESH:
            goal = [THRESH+RADIUS, -MIDDLE]

        # Region C
        if ped.ccw and x > -THRESH and y > THRESH:
            goal = [-MIDDLE, THRESH+RADIUS]
        elif not ped.ccw and x < THRESH and y > THRESH:
            goal = [MIDDLE, THRESH+RADIUS]

        # Region D
        if ped.ccw and x < -THRESH and y > -THRESH:
            goal = [-THRESH-RADIUS, -MIDDLE]
        elif not ped.ccw and x < -THRESH and y < THRESH:
            goal = [-THRESH-RADIUS, MIDDLE]

        return goal

    def step(self, dt, save=False):
        if not self.world.pause:
            self.step_crowd(dt)
            self.step_robots(dt)

            for ii, ped in enumerate(self.world.crowds):
                ped.step()
                if np.linalg.norm(ped.pos - ped.goal) > 2.5: continue
                goal = self.set_goals_roundtrip(ped)
                if len(goal) > 0:
                    self.world.set_ped_goal(ii, goal)

        super(RoundTrip, self).step(dt, save)
        



