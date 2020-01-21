import time

from display import *
from world import World
from basics_2d import Line, Circle
from robot import FollowBot
from pedestrian import Pedestrian
import numpy as np


N_ped = 10
N_robots = 0
inner_dim, outer_dim = 12, 18
default_model = 'powerlaw'
agent_color = GREEN_COLOR


def init_world(world_dim, model=default_model, title='followBot'):
    global world, display
    world = World(N_ped, N_robots, model)
    display = Display(world, world_dim, (960, 960), title + '/' + model)

    # world.pref_speed = 1.5  # FIXME : set it for sim as well


def test1():
    # FIXME: init objects
    line_objects = [Line([5, 0], [5, 8]), Line([6, 0], [6, 9])]
    for l_obj in line_objects:
        world.add_object(l_obj)

    # FIXME: init crowd
    for ii in range(N_ped):
        world.set_ped_position(ii, [0, ii])
        world.set_ped_velocity(ii, [0, 0])
        world.set_ped_goal(ii, [10, ii])

    world.set_robot_position(0, [3, 3.2])
    world.set_robot_leader(0, 0)


def circle():
    init_world(world_dim=[[-5, 5], [-5, 5]], title='circle')
    for ii in range(N_ped):
        theta = ii / N_ped * 2 * np.pi
        world.set_ped_position(ii, np.array([np.cos(theta), np.sin(theta)]) * 5 + np.random.randn(1) * 0.01)
        world.set_ped_goal(ii, np.array([np.cos(theta+np.pi), np.sin(theta+np.pi)]) * 5)
        world.crowds[ii].color = agent_color
    # world.set_robot_position(0, [-1, -1])
    # world.set_robot_leader(0, 0)


def corridor():
    init_world(world_dim=[[-outer_dim, outer_dim], [-outer_dim, outer_dim]], title='corridor')

    line_objects = [Line([0, 0], [10, 0]), Line([2, 7], [10, 7])]
    for l_obj in line_objects:
        world.add_object(l_obj)

    global N_ped, N_robots
    N_ped, N_robots = 10, 1
    for ii in range(5):
        world.set_ped_position(ii, [2, ii + 2])
        world.set_ped_velocity(ii, [0, 0])
        world.set_ped_goal(ii, [10, 6 - ii])

    for ii in range(5, 10):
        world.set_ped_position(ii, [10, ii -5 + 2])
        world.set_ped_velocity(ii, [0, 0])
        world.set_ped_goal(ii, [0, ii])

    world.set_robot_position(0, [1, 2])
    world.set_robot_leader(0, 0)


def RoundTrip(flow_2d):  # 1D flow / 2D flow
    global N_ped, N_robots
    global inner_dim, outer_dim
    k_in_each_corridor = 10
    N_robots, N_ped = 1, 4 * k_in_each_corridor
    ped_radius = 0.25
    inner_dim = 12
    outer_dim = 18

    init_world(world_dim=[[-outer_dim, outer_dim], [-outer_dim, outer_dim]], title='Round Trip')

    # NOTE Symmetric with center at (0, 0)
    line_objects = [Line([-inner_dim, -inner_dim], [inner_dim, -inner_dim]),
                    Line([inner_dim, -inner_dim], [inner_dim, inner_dim]),
                    Line([inner_dim, inner_dim], [-inner_dim, inner_dim]),
                    Line([-inner_dim, inner_dim], [-inner_dim, -inner_dim]),

                    Line([-outer_dim, -outer_dim], [outer_dim, -outer_dim]),
                    Line([outer_dim, -outer_dim], [outer_dim, outer_dim]),
                    Line([outer_dim, outer_dim], [-outer_dim, outer_dim]),
                    Line([-outer_dim, outer_dim], [-outer_dim, -outer_dim]) ]
    for l_obj in line_objects:
        world.add_object(l_obj)

    pom_resolution = 10  # per meter
    world.walkable = np.ones((outer_dim * 2 * pom_resolution, outer_dim * 2 * pom_resolution), dtype=bool)
    world.walkable[(-inner_dim+outer_dim) * pom_resolution: (inner_dim+outer_dim) * pom_resolution,
                   (-inner_dim+outer_dim) * pom_resolution: (inner_dim+outer_dim) * pom_resolution] = 0

    world.mapping_to_grid = lambda x,y : (int(((x+outer_dim) * pom_resolution)),
                                          int(((y+outer_dim) * pom_resolution)))

    ped_poss = []
    # Region A
    for ped_ind in range(0, k_in_each_corridor):
        px_i = np.random.uniform(-inner_dim + ped_radius, inner_dim - ped_radius)
        py_i = np.random.uniform(-outer_dim + ped_radius, -inner_dim - ped_radius)
        ped_poss.append([px_i, py_i])

    # Region B
    for ped_ind in range(k_in_each_corridor, k_in_each_corridor * 2):
        px_i = np.random.uniform(inner_dim + ped_radius, outer_dim - ped_radius)
        py_i = np.random.uniform(-inner_dim + ped_radius, inner_dim - ped_radius)
        ped_poss.append([px_i, py_i])

    # Region C
    for ped_ind in range(k_in_each_corridor * 2, k_in_each_corridor * 3):
        px_i = np.random.uniform(-inner_dim + ped_radius, inner_dim - ped_radius)
        py_i = np.random.uniform(inner_dim + ped_radius, outer_dim - ped_radius)
        ped_poss.append([px_i, py_i])

    # Region D
    for ped_ind in range(k_in_each_corridor * 3, k_in_each_corridor * 4):
        px_i = np.random.uniform(-outer_dim + ped_radius, -inner_dim - ped_radius)
        py_i = np.random.uniform(-inner_dim + ped_radius, inner_dim - ped_radius)
        ped_poss.append([px_i, py_i])

    for ped_ind in range(len(ped_poss)):
        world.set_ped_position(ped_ind, ped_poss[ped_ind])
        world.set_ped_goal(ped_ind, ped_poss[ped_ind])
        world.set_ped_velocity(ped_ind, [0, 0])
        if ped_ind == 0 or not flow_2d: continue
        if np.random.rand() > 0.5:
            world.crowds[ped_ind].ccw = True
            world.crowds[ped_ind].color = BLUE_COLOR
        else:
            world.crowds[ped_ind].ccw = False
            world.crowds[ped_ind].color = RED_COLOR

    # set the Robot position just behind first ped
    ped0_pos = world.crowds[0].pos
    world.set_robot_position(0, [ped0_pos[0] - 1.5, ped0_pos[1]])
    world.set_robot_leader(0, 0)


def set_goals_roundtrip(ped):
    goal = []

    x, y = ped.pos[0], ped.pos[1]
    RADIUS = ped.radius
    THRESH = inner_dim + ped.radius
    MIDDLE = (inner_dim + outer_dim)/2

    # Region A
    if ped.ccw and x < THRESH and y < -THRESH:
        goal = [MIDDLE, -THRESH-RADIUS]
    elif not ped.ccw and x > -THRESH and y < -THRESH:
        goal = [-MIDDLE, -THRESH-RADIUS]

    # Region B
    if ped.ccw and x > THRESH and y < THRESH:
        goal = [THRESH+RADIUS, MIDDLE]
    elif not ped.ccw and x > THRESH and y > -THRESH:
        goal = [THRESH+RADIUS, -MIDDLE]

    # Region C
    if ped.ccw and x > -THRESH and y > THRESH:
        goal = [-MIDDLE, THRESH+RADIUS]
    elif not ped.ccw and x < THRESH and y > THRESH:
        goal = [MIDDLE, THRESH+RADIUS]

    # Region D
    if ped.ccw and x < -THRESH and y > -THRESH:
        goal = [-THRESH-RADIUS, -MIDDLE]
    elif not ped.ccw and x < -THRESH and y < THRESH:
        goal = [-THRESH-RADIUS, MIDDLE]

    return goal


# corridor()
# circle()
RoundTrip(True)
# TODO: setup pygame
world.sim.setTime(0)
display.update()
pause = True
for tt in range(0, 10000):
    if not pause:
        world.step(0.02)
        # update goals
        # if tt % 5 != 0: continue

        for ii, ped in enumerate(world.crowds):
            if np.linalg.norm(ped.pos - ped.goal) > 2.5: continue
            goal = set_goals_roundtrip(ped)
            if len(goal) > 0:
                world.set_ped_goal(ii, goal)

    toggle_pause = display.update()
    if toggle_pause: pause = not pause
    time.sleep(0.01)

    if not pause:
        display.save('/home/cyrus/Videos/crowdsim/followbot/')


