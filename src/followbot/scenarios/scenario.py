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

    def step(self, save=False):
        if not self.world.pause and save:
            self.display.save('/home/cyrus/Videos/crowdsim/followbot/')

    def update_disply(self):
        toggle_pause = self.display.update()
        if toggle_pause: self.world.pause = not self.world.pause
        time.sleep(0.01)