# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.scenarios.scenario import Scenario


class SimulationScenario(Scenario):
    def __init__(self):
        super(SimulationScenario, self).__init__()

    def setup(self):
        super(SimulationScenario, self).setup()

    def step_crowd(self):
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

    def step(self, save=False):
        super(SimulationScenario, self).step()
