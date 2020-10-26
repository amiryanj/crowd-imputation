# Author: Javad Amirian
# Email: amiryan.j@gmail.com
from abc import ABC, abstractmethod

import pygame

from followbot.simulator.world import World
from followbot.gui.visualizer import Visualizer
import os.path
import time


class Scenario(ABC):
    def __init__(self, **kwargs):
        self.world = World
        self.visualizer = Visualizer

        self.cur_t = -1

        self.n_robots = kwargs.get("numRobots", 1)       # to be used to initialize the world
        self.leader_id = kwargs.get("LeaderPedId", -1)

        self.n_peds = 0  # will be read from dataset

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def step_crowd(self, dt):
        pass

    def step_robots(self, dt):
        for jj, robot in enumerate(self.world.robots):
            robot.step(dt)
            self.world.set_robot_position(jj, robot.pos)
            self.world.set_robot_velocity(jj, robot.vel)

    @abstractmethod
    def step(self, dt, save=False):
        if not self.world.pause and save:
            home = os.path.expanduser("~")
            self.visualizer.save_screenshot(os.path.join(home, 'Videos/followbot/'))
        self.update_display()

    def update_display(self, delay_sec=0.01):
        toggle_pause = self.visualizer.update()
        if toggle_pause == pygame.K_SPACE:
            self.world.pause = not self.world.pause
        time.sleep(delay_sec)
