# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from abc import ABC, abstractmethod

from followbot.crowdsim.grouping_behavior import GroupingBehavior
from followbot.scenarios.scenario import Scenario
import numpy as np


class SimulationScenario(Scenario, ABC):
    """Any non-real scenario should inherit this class"""

    def __init__(self, **kwargs):
        self.ped_radius = kwargs.get("pedRadius", 0.25)
        self.n_peds = kwargs.get("numPeds", 0)
        self.group_ids = []
        self.grouping_behavior_handler = None
        super(SimulationScenario, self).__init__()

    @abstractmethod
    def setup(self, **kwargs):
        super(SimulationScenario, self).setup()

    def step_crowd(self, dt):
        self.world.sim.doStep(dt)
        for ii in range(self.n_peds):
            try:
                p = self.sim.getCenterNext(ii)
                v = self.sim.getCenterVelocityNext(ii)
                # apply inertia
                v_new = np.array(v) * (1 - self.inertia_coeff) + self.crowds[ii].vel * self.inertia_coeff
                p_new = self.crowds[ii].pos + v_new * dt
                self.set_ped_position(ii, p_new)
                self.set_ped_velocity(ii, v_new)
            except Exception:
                raise ValueError('exception occurred in running crowd sim')

    def step_robots(self, dt, lidar_enabled):
        super(SimulationScenario, self).step_robots(dt, lidar_enabled)

    def step(self, dt, save=False):
        if not self.world.pause:
            self.world.sim.doStep(dt)

            group_vels = self.grouping_behavior_handler.step(crowds=self.world.crowds)

            for ii in range(self.n_peds):
                p_new = self.world.sim.getCenterNext(ii)
                v = self.world.sim.getCenterVelocityNext(ii)

                # apply inertia
                v_new = np.array(v) * (1 - self.world.inertia_coeff) \
                        + self.world.crowds[ii].vel * self.world.inertia_coeff
                # p_new = self.world.crowds[ii].pos + v_new * dt

                v_new += group_vels[ii]

                self.world.set_ped_position(ii, p_new)
                self.world.set_ped_velocity(ii, v_new)

                self.world.crowds[ii].step(dt)

            for jj in range(self.n_peds, self.n_peds + self.n_robots):
                self.world.robots[jj - self.n_peds].vel = np.array(self.world.sim.getCenterVelocityNext(jj))

                if self.world.robots[jj - self.n_peds].pos[0] > self.world.world_dim[0][1]:
                    self.world.robots[jj - self.n_peds].pos[0] = self.world.world_dim[0][0]
            self.step_robots(dt)

        super(SimulationScenario, self).step(dt, save)
