# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.scenarios.scenario import Scenario
import numpy as np


class SimulationScenario(Scenario):
    """Any non-real scenario should inherit this class"""

    def __init__(self, **kwargs):
        self.ped_radius = kwargs.get("pedRadius", 0.25)
        self.n_peds = kwargs.get("numPeds", 0)
        super(SimulationScenario, self).__init__()

    def setup(self, **kwargs):
        super(SimulationScenario, self).setup()

    def step_crowd(self, dt):
        self.sim.doStep(dt)
        for ii in range(self.n_peds):
            try:
                p = self.sim.getCenterNext(ii)
                v = self.sim.getCenterVelocityNext(ii)
                # apply inertia
                v_new = np.array(v) * (1 - self.inertia_coeff) + self.crowds[ii].vel * self.inertia_coeff
                p_new = self.crowds[ii].pos + v_new * dt
                self.set_ped_position(ii, p_new)
                self.set_ped_velocity(ii, v_new)
            except:
                print('exception occurred in running crowd sim')

    def step(self, dt, save=False):
        if not self.world.pause:
            self.world.sim.doStep(dt)
            for ii in range(self.n_peds):
                p_new = self.world.sim.getCenterNext(ii)
                v = self.world.sim.getCenterVelocityNext(ii)
                # apply inertia
                v_new = np.array(v) * (1 - self.world.inertia_coeff) \
                        + self.world.crowds[ii].vel * self.world.inertia_coeff
                # p_new = self.world.crowds[ii].pos + v_new * dt

                self.world.set_ped_position(ii, p_new)
                self.world.set_ped_velocity(ii, v_new)

            self.world.step_robot(dt)
        super(SimulationScenario, self).step(dt, save)
