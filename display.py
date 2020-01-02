import math

from world import World
import pygame
import numpy as np

RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (0, 0, 255)
MAGENTA_COLOR = (255, 0, 255)
YELLOW_COLOR = (255, 255, 0)
CYAN_COLOR = (0, 255, 255)

BLUE_LIGHT = (120, 120, 255)


class Display:
    def __init__(self, world, world_size, win_size=(640, 640), caption='followbot'):
        pygame.init()
        self.world = world
        self.win = pygame.display.set_mode(win_size)
        pygame.display.set_caption(caption)

        margin = 0.1
        sx = float(win_size[0]) / (world_size[0] * (1 + 2 * margin))
        sy = -float(win_size[1]) / (world_size[1] * (1 + 2 * margin))
        self.scale = np.array([[sx, 0], [0, sy]])
        self.trans = np.array([margin * win_size[0], win_size[1] - margin * win_size[1]], dtype=np.float)
        x = 1

    def transform(self, x):
        return self.trans + np.matmul(x, self.scale)

    def circle(self, center, radius, color, width=0):
        center_uv = self.transform(center)
        pygame.draw.circle(self.win, color, (int(center_uv[0]), int(center_uv[1])), radius, width)

    def line(self, p1, p2, color, width=1):
        p1_uv = self.transform(p1)
        p2_uv = self.transform(p2)
        pygame.draw.line(self.win, color, p1_uv, p2_uv, width)

    # returns pause state
    def update(self):
        self.win.fill((0, 0, 0))  # ms
        self.circle([0, 0], 4, GREEN_COLOR)  # DEBUG

        for ii in range(len(self.world.crowds)):
            self.circle(self.world.crowds[ii].pos, 10, YELLOW_COLOR)

        for obj in self.world.objects:
            self.line(obj.line[0], obj.line[1], RED_COLOR, 3)

        # TODO: draw robot
        for robot in self.world.robots:
            self.circle(robot.pos, 12, GREEN_COLOR, 3)
            self.circle(robot.leader_ped.pos, 14, GREEN_COLOR, 2)
            # draw a vector showing orientation
            u, v = math.cos(robot.orien) * 0.5, math.sin(robot.orien) * 0.5
            self.line(robot.pos, robot.pos + [u, v], MAGENTA_COLOR)
            # draw Lidar output
            for pnt in robot.range_data:
                if math.isnan(pnt[0]) or math.isnan(pnt[1]):
                    print('Nan Value in Lidar')
                    exit(1)
                else:
                    self.circle(pnt, 2, MAGENTA_COLOR)

            for seg in robot.lidar_segments:
                self.line(seg[0], seg[-1], BLUE_LIGHT, 3)

            for pos in robot.detections:
                self.circle(pos, 5, GREEN_COLOR, 2)

            for track in robot.tracks:
                if track.coasted: continue
                self.circle(track.position(), 3, RED_COLOR)

        # pygame.display.flip()
        pygame.display.update()
        pygame.time.delay(10)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                print('Simulation exited by user')
                exit(1)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return True
        return False



