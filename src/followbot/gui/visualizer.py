import os
import math
import pygame
import pygame.gfxdraw
import numpy as np
from sklearn.externals._pilutil import imresize
import matplotlib.pyplot as plt
from followbot.robot_functions.follower_bot import FollowerBot
from followbot.util.basic_geometry import Line, Circle
from followbot.util.cv_importer import *

BLACK_COLOR = (0, 0, 0)
GREY_COLOR = (100, 100, 100)
LIGHT_GREY_COLOR = (200, 200, 200)
RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (0, 0, 255)
SKY_BLUE_COLOR = (0, 160, 255)  # Deep Sky Blue
YELLOW_COLOR = (180, 180, 0)
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


class Visualizer:
    def __init__(self, world, world_dim, caption='followbot', subViewRowCount=1, subViewColCount=1):
        """
        :param world:  pointer to world object
        :param world_dim:  [[x0, x1], [y0, y1]]
        :param win_size:
        :param caption: caption for display window
        """
        pygame.init()
        pygame.font.init()  # you have to call this at the start, # if you want to use this module.
        self.font = pygame.font.SysFont('Comic Sans MS', 20)

        win_size = (1200, 920)  # FixMe
        flags = pygame.DOUBLEBUF  # | pygame.FULLSCREEN
        self.win = pygame.display.set_mode(win_size, flags)

        # the display can be divided into an array of subviews
        self.subviews_array_size = np.array([subViewRowCount, subViewColCount]).astype(int)

        self.win_size = win_size
        self.subview_size = [win_size[0] / self.subviews_array_size[1],
                             win_size[1] / self.subviews_array_size[0]]

        margin = 0.1
        self.world_dim = world_dim
        self.world_w = world_dim[0][1] - world_dim[0][0]
        self.world_h = world_dim[1][1] - world_dim[1][0]

        sx = float(win_size[0]) / (self.world_w * (1 + 2 * margin))
        sy = -float(win_size[1]) / (self.world_h * (1 + 2 * margin))

        # make (-sy) == sx
        sx = min(sx, abs(sy))
        sy = -sx

        self.scale = np.stack([[np.array([[sx, 0], [0, sy]])
                                for __ in range(self.subviews_array_size[1])]
                               for _ in range(self.subviews_array_size[0])])
        # self.trans = np.array([margin * win_size[0] - world_dim[0][0] * sx,
        #                        margin * win_size[1] - world_dim[1][1] * sy], dtype=np.float)

        self.trans = np.stack([[np.array(self.subview_size, dtype=np.float) / 2.
                                for __ in range(self.subviews_array_size[1])]
                               for _ in range(self.subviews_array_size[0])])

        # for enabling drag screen by mouse
        self.trans_offset_by_user = np.zeros(2, int)
        self.drag_is_active = False
        self.last_click_pt = None
        # pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

        self.world = world
        self.win.fill([255, 255, 255])
        pygame.display.set_caption(caption)

    def transform(self, x, view_index=(0, 0)):
        return self.trans[view_index[0]][view_index[1]] + np.matmul(x, self.scale[view_index[0]][view_index[1]])

    # =========== draw methods ================
    # =========================================
    def draw_circle(self, center, radius, color, width=0, gfx=True, view_index=(0, 0)):
        """
        :param center: center of circle
        :param radius: radius of circle
        :param color: color
        :param width: width of circle stroke, filled if width == 0
        :param gfx: boolean, whether to use gfx functions of pygame which has antialias feature
        :param view_index: the index of view that is used to find the corresponding transformation matrix
        :return: None
        """
        center_uv = self.transform(center, view_index)
        if not -radius < center_uv[0] < self.win_size[0] + radius or\
           not -radius < center_uv[1] < self.win_size[1] + radius:
            return
        if gfx:
            if width <= 0:
                pygame.gfxdraw.filled_circle(self.win, int(center_uv[0]), int(center_uv[1]), radius, color)
            else:
                pygame.gfxdraw.aacircle(self.win, int(center_uv[0]), int(center_uv[1]), radius, color)
                pygame.gfxdraw.filled_circle(self.win, int(center_uv[0]), int(center_uv[1]), radius, color)
        else:
            pygame.draw.circle(self.win, color, (int(center_uv[0]), int(center_uv[1])), radius, width)

    def draw_trigon(self, center, orien, radius, color, width=0, view_index=(0, 0)):
        center_uv = self.transform(center, view_index)
        if not -radius < center_uv[0] < self.win_size[0] + radius or\
           not -radius < center_uv[1] < self.win_size[1] + radius:
            return
        trigon_vertice_angles = np.array([-orien, -orien + np.deg2rad(90), -orien - np.deg2rad(90)])
        pnts = center_uv + np.stack([np.cos(trigon_vertice_angles), np.sin(trigon_vertice_angles)], axis=1) * radius
        pnts = np.round(pnts).astype(int)

        if width <= 0:
            pygame.gfxdraw.filled_trigon(self.win, pnts[0, 0], pnts[0, 1], pnts[1, 0], pnts[1, 1],
                                         pnts[2, 0], pnts[2, 1], color)
        else:
            pygame.gfxdraw.trigon(self.win, pnts[0, 0], pnts[0, 1], pnts[1, 0], pnts[1, 1],
                                  pnts[2, 0], pnts[2, 1], color)

    def draw_line(self, p1, p2, color, width=1, view_index=(0, 0)):
        p1_uv = self.transform(p1, view_index)
        p2_uv = self.transform(p2, view_index)
        # pygame.draw.line(self.win, color, p1_uv, p2_uv, width)
        pygame.draw.aaline(self.win, color, p1_uv, p2_uv)  # antialias

    def draw_lines(self, points, color, width=1, view_index=(0, 0)):
        if len(points) < 2: return
        points_uv = self.transform(points, view_index)
        # pygame.draw.lines(self.win, color, False, points_uv, width)
        pygame.draw.aalines(self.win, color, False, points_uv)  # antialias

    def draw_image(self, im, alpha, view_index=(0, 0)):
        im_surf = pygame.surfarray.make_surface(im)
        im_surf.set_alpha(alpha)
        self.win.blit(im_surf, (self.trans[view_index[0], view_index[1], 0] + self.scale[view_index[0], view_index[1], 0, 0] * self.world_dim[0][0],
                      self.trans[view_index[0], view_index[1], 1] + self.scale[view_index[0], view_index[1], 1, 1] * self.world_dim[1][1]))

    # =========================================
    def update(self):
        # the scene will be centered on the (1st) robot position
        if len(self.world.robots):
            for col_jj in range(self.subviews_array_size[1]):
                for row_ii in range(self.subviews_array_size[0]):
                    center_of_screen = self.world.robots[0].pos.copy()
                    # center_of_screen[0] = np.mean(self.world_dim[0])
                    center_of_screen[1] = np.mean(self.world_dim[1])
                    self.trans[row_ii][col_jj] = np.array(self.subview_size, dtype=np.float) / 2. \
                                                 - center_of_screen * [self.scale[row_ii, col_jj, 0, 0],
                                                                       self.scale[row_ii][col_jj][1, 1]] \
                                                 + self.subview_size * np.array([col_jj, row_ii]) \
                                                 + self.trans_offset_by_user

        self.win.fill(WHITE_COLOR)

        # print time / and some other info in Main View
        title_surface = self.font.render('t=[%.2f] frame=[%d]' % (self.world.time, self.world.original_frame_id),
                                         True, (200, 0, 200))
        self.win.blit(title_surface, (20, 20))

        # Draw lines between subviews
        for row_ii in range(1, self.subviews_array_size[0]):
            pygame.draw.line(self.win, GREY_COLOR,
                             (0, row_ii * self.subview_size[1]),
                             (self.win_size[0], row_ii * self.subview_size[1]), width=2)
            #
            title_surface = self.font.render('H%d' % row_ii, False, (200, 0, 0))
            self.win.blit(title_surface, (20, row_ii * self.subview_size[1] + 20))

        for col_jj in range(1, self.subviews_array_size[1]):
            pygame.draw.line(self.win, GREY_COLOR,
                             (col_jj * self.subview_size[0], 0),
                             (col_jj * self.subview_size[0], self.win_size[1]), width=2)
        # -----------------------------

        # Draw Obstacles
        for obs in self.world.obstacles:
            if isinstance(obs, Line):
                self.draw_line(obs.line[0], obs.line[1], RED_COLOR, 3)
            elif isinstance(obs, Circle):
                self.draw_circle(obs.center, int(obs.radius * self.scale[0][0][0, 0]), RED_COLOR)
        # -----------------------------

        # Draw Pedestrians
        for ii in range(len(self.world.crowds)):
            self.draw_circle(self.world.crowds[ii].pos, 8, self.world.crowds[ii].color, width=1)
            self.draw_line(self.world.crowds[ii].pos, self.world.crowds[ii].pos + self.world.crowds[ii].vel * 0.5,
                           MAGENTA_COLOR, 2)
            self.draw_trigon(self.world.crowds[ii].pos, self.world.crowds[ii].orien(), 7,
                             LIGHT_GREY_COLOR)  # orien triangle
            self.draw_lines(self.world.crowds[ii].trajectory, DARK_GREEN_COLOR + (100,), 3)  # track of robot

            if self.world.crowds[ii].biped_mode:  # if we use the biped model for pedestrians
                ped_geo = self.world.crowds[ii].geometry()
                # self.draw_circle(ped_geo.center1, 4, SKY_BLUE_COLOR)
                # self.draw_circle(ped_geo.center2, 4, MAGENTA_COLOR)

        # -----------------------------

        # Draw robot(s)
        for robot in self.world.robots:
            self.draw_circle(robot.pos, 9, MAGENTA_COLOR)
            if isinstance(robot, FollowerBot):
                self.draw_circle(robot.leader_ped.pos, 10, PINK_COLOR, 5, gfx=False)
            # draw a vector showing orientation of the robot
            u, v = math.cos(robot.orien) * 0.5, math.sin(robot.orien) * 0.5
            # self.draw_line(robot.pos, robot.pos + [u, v], ORANGE_COLOR, 3)
            self.draw_trigon(robot.pos, np.arctan2(v, u), 7, CYAN_COLOR, width=0)

            # draw Lidar output as points
            for jj, pnt in enumerate(robot.lidar.data.last_points):
                if robot.lidar.data.last_range_data[jj] < robot.lidar.range_max - 0.01:
                    self.draw_circle(pnt, 2, YELLOW_COLOR, gfx=False)

            # for seg in robot.lidar_segments:
            #     self.line(seg[0], seg[-1], BLUE_LIGHT, 3)

            # Robot Hypotheses
            # ====================================
            for ii, hypothesis in enumerate(robot.hypothesis_worlds):
                if ii == 0 or ii == 1:  # Never
                    # show Blind-Spot-Map as a background image
                    blindspot_map = np.clip(np.fliplr(robot.blind_spot_map.data), a_min=0, a_max=255)
                    blindspot_map = imresize(blindspot_map, self.scale[0, 0, 0, 0] / robot.mapped_array_resolution)
                    blindspot_map = np.stack([255 - blindspot_map, 255 - blindspot_map, 255 - blindspot_map], axis=2)
                    self.draw_image(blindspot_map, 80, view_index=(ii + 1, 0))

                # show Crowd-Flow-Map as a background image
                if ii == 1:
                    crowdflow_map = np.clip(np.fliplr(robot.crowd_territory_map.data[:, :]), a_min=0, a_max=255)
                    crowdflow_map = imresize(crowdflow_map, self.scale[0, 0, 0, 0] / robot.mapped_array_resolution)
                    crowdflow_map = np.stack([255 - crowdflow_map, np.zeros_like(crowdflow_map), crowdflow_map], axis=2)
                    self.draw_image(crowdflow_map, 60, view_index=(ii + 1, 0))

                # show Link-Pdf-Map as a background image
                if ii == -1:
                    link_pdf_map = np.clip(np.fliplr(robot.blind_spot_projector.social_ties_cartesian_pdf_aggregated.data), a_min=0, a_max=255)
                    link_pdf_map = imresize(link_pdf_map, self.scale[0, 0, 0, 0] / robot.mapped_array_resolution)
                    link_pdf_map = np.stack([link_pdf_map, link_pdf_map, np.zeros_like(link_pdf_map)], axis=2)
                    self.draw_image(link_pdf_map, 100, view_index=(ii + 1, 0))

                # Draw robot
                self.draw_circle(robot.pos, 9, MAGENTA_COLOR, view_index=(ii + 1, 0))
                self.draw_trigon(robot.pos, np.arctan2(v, u), 7, CYAN_COLOR, view_index=(ii + 1, 0))
                if isinstance(robot, FollowerBot):
                    self.draw_circle(robot.leader_ped.pos, 11, PINK_COLOR, 5, view_index=(ii + 1, 0))
                # u, v = math.cos(robot.orien) * 0.5, math.sin(robot.orien) * 0.5
                # self.draw_line(robot.pos, robot.pos + [u, v], ORANGE_COLOR, 5, view_index=(ii + 1, 0))

                # draw tracks
                for track in robot.tracks:
                    if track.coasted: continue

                    if ii == 0 or ii == 1:
                        self.draw_circle(track.get_position(), 8, GREEN_COLOR, view_index=(ii + 1, 0))
                        self.draw_line(track.get_position(), track.get_position() + track.get_velocity() * 0.5,
                                       MAGENTA_COLOR, 2, view_index=(ii + 1, 0))
                        # if len(track.recent_detections) >= 2:
                        #     self.draw_lines(track.recent_detections, WHITE_COLOR, 1, view_index=(ii + 1, 0))


                # Draw detected agents
                for det in robot.detected_peds:
                    self.draw_circle(det, 4, BLACK_COLOR, view_index=(ii + 1, 0))

                for jj in range(len(hypothesis.crowds)):
                    if hasattr(hypothesis.crowds[jj], 'projected'):
                        self.draw_circle(hypothesis.crowds[jj].pos, 9, SKY_BLUE_COLOR, view_index=(ii + 1, 0))
                        self.draw_line(hypothesis.crowds[jj].pos, hypothesis.crowds[jj].pos + hypothesis.crowds[jj].vel * 0.5,
                                       MAGENTA_COLOR, view_index=(ii + 1, 0))

                for jj in range(len(self.world.crowds)):
                    self.draw_circle(self.world.crowds[jj].pos, 8, self.world.crowds[jj].color + (60,),
                                     width=1, view_index=(ii + 1, 0))
                    if ii == 0:
                        self.draw_line(self.world.crowds[jj].pos,
                                       self.world.crowds[jj].pos + self.world.crowds[jj].vel * 0.5,
                                       MAGENTA_COLOR, 2, view_index=(ii + 1, 0))

                # Draw lidar points
                for jj, pnt in enumerate(robot.lidar.data.last_points):
                    if robot.lidar.data.last_range_data[jj] < robot.lidar.range_max - 0.01:
                        self.draw_circle(pnt, 1, YELLOW_COLOR, view_index=(ii + 1, 0), gfx=False)

        # pygame.display.flip()
        pygame.display.update()
        pygame.time.delay(5)
        plt.pause(0.001)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                print('Simulation exited by user')
                exit(1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click press
                click_loc = np.matmul(np.linalg.inv(self.scale[0][0]), (pygame.mouse.get_pos() - self.trans[0][0]))
                self.drag_is_active = True
                self.last_click_pt = np.array(pygame.mouse.get_pos())
                print("click location: [%.3f %.3f]" % (click_loc[0], click_loc[1]))
                # print('- ped:\n\t\tpos_x: %.3f\n\t\tpos_y: %.3f\n\t\torien: 0' % (click_loc[0], click_loc[1]))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Left click release
                self.drag_is_active = False
                self.trans_offset_by_user += np.array(pygame.mouse.get_pos()) - self.last_click_pt

            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.scale *= 1.1
                else:
                    self.scale /= 1.1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return event.key
        return None

    def save_screenshot(self, save_path, frame_id):
        pygame.image.save(self.win, os.path.join(save_path, 'win-%05d.jpg' % frame_id))
