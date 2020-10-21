# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.simulator.world import World
from followbot.gui.visualizer import Visualizer
import os.path
import time


class Scenario:
    def __init__(self, **kwargs):
        self.world = World
        self.visualizer = Visualizer

        self.cur_t = -1
        self.n_robots = kwargs.get("numRobots", 1)
        self.leader_id = kwargs.get("LeaderPedId", -1)

        self.n_peds = 0  # will be read from dataset

    # @Warning: Don't forget to override this function in an inherited class
    def setup(self, **kwargs):
        raise NotImplementedError

    # @Warning: Don't forget to override this function in an inherited class
    def step_crowd(self, dt):
        raise NotImplementedError

    def step(self, dt, save=False):
        if not self.world.pause and save:
            home = os.path.expanduser("~")
            self.visualizer.save_screenshot(os.path.join(home, 'Videos/followbot/'))
        self.update_display()

    def update_display(self, delay_sec=0.01):
        toggle_pause = self.visualizer.update()
        if toggle_pause: self.world.pause = not self.world.pause
        time.sleep(delay_sec)
