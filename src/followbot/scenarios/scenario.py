# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.simulator.world import World
from followbot.gui.display import Display
import time


class Scenario:
    def __init__(self):
        self.world = World
        self.display = Display

        self.cur_t = -1
        self.n_robots = 1
        self.leader_id = -1

        self.n_peds = 0  # will be read from dataset

    def setup(self):
        pass

    def step(self):
        pass

    def update_disply(self):
        self.display.update()
        time.sleep(0.05)