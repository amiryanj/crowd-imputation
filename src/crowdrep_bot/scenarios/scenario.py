# Author: Javad Amirian
# Email: amiryan.j@gmail.com
from abc import ABC, abstractmethod

import pygame

from crowdrep_bot.scenarios.world import World
from crowdrep_bot.gui.visualizer import Visualizer
import os.path
import time


class Scenario(ABC):
    def __init__(self, **kwargs):
        self.world = World
        self.title = kwargs.get("title", "")
        self.visualizer = None

        self.n_robots = kwargs.get("numRobots", 1)       # to be used to initialize the world

        self.n_peds = 0  # will be read from dataset

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def step_crowd(self, dt):
        pass

    def step_robots(self, dt, lidar_enabled):
        for jj, robot in enumerate(self.world.robots):
            robot.step(dt, lidar_enabled)
            self.world.set_robot_position(jj, robot.pos)
            self.world.set_robot_velocity(jj, robot.vel)

    @abstractmethod
    def step(self, dt, lidar_enabled, save=False):
        if not self.world.pause:
            self.world.time += dt
            self.world.frame_id += 1
            if save:
                home = os.path.expanduser("~")
                self.visualizer.save_screenshot(os.path.join(home, 'Videos/crowdrep_bot/scene'), self.world.original_frame_id)
        self.update_display()
        return True


    def update_display(self, delay_sec=0.01):
        if self.visualizer:
            toggle_pause = self.visualizer.update()
            if toggle_pause == pygame.K_SPACE:
                self.world.pause = not self.world.pause
            time.sleep(delay_sec)
