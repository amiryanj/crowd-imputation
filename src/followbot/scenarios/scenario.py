# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.simulator.world import World
from followbot.gui.display import Display
import os.path
import time


class Scenario:
    def __init__(self):
        self.world = World
        self.display = Display

        self.cur_t = -1
        self.n_robots = 1
        self.leader_id = -1

        self.n_peds = 0  # will be read from dataset


    # FixMe: Don't forget to override this function in an inherited class
    def setup(self):
        raise NotImplementedError

    # FixMe: Don't forget to override this function in an inherited class
    def step_crowd(self):
        raise NotImplementedError

    def step(self, save=False):
        if not self.world.pause and save:
            home = os.path.expanduser("~")
            self.display.save(os.path.join(home, 'Videos/followbot/'))

    def update_disply(self):
        toggle_pause = self.display.update()
        if toggle_pause: self.world.pause = not self.world.pause
        time.sleep(0.01)
