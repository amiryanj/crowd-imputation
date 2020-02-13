import os
import math
import pygame
import numpy as np

from followbot.cv_importer import *
from followbot.world import World

BLACK_COLOR = (0, 0, 0)
RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (0, 0, 255)
YELLOW_COLOR = (150, 150, 0)
CYAN_COLOR = (0, 255, 255)
MAGENTA_COLOR = (255, 0, 255)

DARK_GREEN_COLOR = (0, 128, 0)
NAVY_COLOR = (0, 0, 128)
PINK_COLOR = (255, 20, 147)
KHAKI_COLOR = (240, 230, 140)
ORANGE_COLOR = (255, 69, 0)
OLIVE_COLOR = (128, 128, 0)
BLUE_LIGHT = (120, 120, 255)
WHITE_COLOR = (255, 255, 255)


class Display:
    def __init__(self, world, world_dim, win_size=(960, 960), caption='followbot'):
        '''
        :param world:  pointer to world object
        :param world_dim:  [[x0, x1], [y0, y1]]
        :param win_size:
        :param caption:
        '''
        pygame.init()
        self.world = world
        self.win = pygame.display.set_mode(win_size)
        self.win.fill([255, 255, 255])
        pygame.display.set_caption(caption)

        margin = 0.1
        world_w = world_dim[0][1] - world_dim[0][0]
        world_h = world_dim[1][1] - world_dim[1][0]

        sx = float(win_size[0]) / (world_w * (1 + 2 * margin))
        sy = -float(win_size[1]) / (world_h * (1 + 2 * margin))
        self.scale = np.array([[sx, 0], [0, sy]])
        self.trans = np.array([margin * win_size[0] - world_dim[0][0] * sx,
                               margin * win_size[1] - world_dim[1][1] * sy], dtype=np.float)
        self.local_time = 0
        self.grid_map = []

    def transform(self, x):
        return self.trans + np.matmul(x, self.scale)

    def circle(self, center, radius, color, width=0):
        center_uv = self.transform(center)
        pygame.draw.circle(self.win, color, (int(center_uv[0]), int(center_uv[1])), radius, width)

    def line(self, p1, p2, color, width=1):
        p1_uv = self.transform(p1)
        p2_uv = self.transform(p2)
        pygame.draw.line(self.win, color, p1_uv, p2_uv, width)

    def lines(self, points, color, width=1):
        if len(points) < 2: return
        points_uv = self.transform(points)
        pygame.draw.lines(self.win, color, False, points_uv, width)

    # returns pause state
    def update(self):
        self.local_time += 1
        self.win.fill(WHITE_COLOR)  # ms
        self.circle([0, 0], 2, WHITE_COLOR)  # DEBUG

        # Pedestrians
        for ii in range(len(self.world.crowds)):
            self.circle(self.world.crowds[ii].pos, 10, self.world.crowds[ii].color)
            self.lines(self.world.crowds[ii].trajectory, DARK_GREEN_COLOR, 3)

        # Objects
        for obj in self.world.objects:
            self.line(obj.line[0], obj.line[1], RED_COLOR, 3)

        # TODO: draw robot
        for robot in self.world.robots:
            self.circle(robot.pos, 12, ORANGE_COLOR)
            self.circle(robot.leader_ped.pos, 11, PINK_COLOR)
            # draw a vector showing orientation
            u, v = math.cos(robot.orien) * 0.5, math.sin(robot.orien) * 0.5
            self.line(robot.pos, robot.pos + [u, v], GREEN_COLOR, 3)
            # draw Lidar output
            for pnt in robot.lidar.last_range_pnts:
                if math.isnan(pnt[0]) or math.isnan(pnt[1]):
                    print('Nan Value in Lidar')
                    exit(1)
                else:
                    self.circle(pnt, 2, WHITE_COLOR)

            for seg in robot.lidar_segments:
                self.line(seg[0], seg[-1], BLUE_LIGHT, 3)

            for pos in robot.lidar.last_range_pnts:
                self.circle(pos, 2, GREEN_COLOR, 2)

            for track in robot.tracks:
                if track.coasted: continue
                self.circle(track.position(), 4, YELLOW_COLOR)
                if len(track.recent_detections) >= 2:
                    self.lines(track.recent_detections, ORANGE_COLOR, 1)

            if len(robot.lidar.last_occupancy_gridmap) > 1:
                self.grid_map = np.rot90(robot.lidar.last_occupancy_gridmap.copy().astype(float))  # + self.world.walkable * 0.5)
                cv2.namedWindow('grid', cv2.WINDOW_NORMAL)
                cv2.imshow('grid', self.grid_map)
                cv2.waitKey(2)

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

    def save(self, dir):
        pygame.image.save(self.win, os.path.join(dir, 'win-%05d.jpg' % self.local_time))
        if len(self.grid_map) > 1:
            cv2.imwrite(os.path.join(dir, 'grid-%05d.png' % self.local_time), self.grid_map * 255)





