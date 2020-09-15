# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.scenarios.scenario import Scenario


class SimulationScenario(Scenario):
    def __init__(self):
        super(SimulationScenario, self).__init__()

    def setup(self):
        super(SimulationScenario, self).setup()

    def step(self, save=False):
        super(SimulationScenario, self).step()
