import time
import numpy as np

from followbot.display import *
from followbot.world import World
from followbot.basics_2d import Line, Circle
from followbot.robot import FollowBot
from followbot.pedestrian import Pedestrian
from followbot.roundtrip import RoundTrip


default_model = 'powerlaw'
agent_color = GREEN_COLOR


def init_world(world_dim, model=default_model, title='followBot'):
    world = World(N_ped, N_robots, model)
    display = Display(world, world_dim, (960, 960), title + '/' + model)
    # world.pref_speed = 1.5  # FIXME : set it for sim as well
    return world, display


def setup_linear_peds():
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
    world.sim.setTime(0)
    display.update()


def setup_circle():
    init_world(world_dim=[[-5, 5], [-5, 5]], title='circle')
    for ii in range(N_ped):
        theta = ii / N_ped * 2 * np.pi
        world.set_ped_position(ii, np.array([np.cos(theta), np.sin(theta)]) * 5 + np.random.randn(1) * 0.01)
        world.set_ped_goal(ii, np.array([np.cos(theta+np.pi), np.sin(theta+np.pi)]) * 5)
        world.crowds[ii].color = agent_color
    # world.set_robot_position(0, [-1, -1])
    # world.set_robot_leader(0, 0)
    world.sim.setTime(0)
    display.update()


def setup_corridor():
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
    world.sim.setTime(0)
    display.update()


def update_default(save=False):
    if not world.pause:
        world.step(0.02)

    toggle_pause = display.update()
    if toggle_pause: world.pause = not world.pause
    time.sleep(0.01)

    if not world.pause and save:
        display.save('/home/cyrus/Videos/crowdsim/followbot/')


if __name__ == '__main__':
    scenario = RoundTrip()
    # setup_corridor()
    # setup_circle()
    scenario.setup('powerlaw', flow_2d=True)

    while True:
        scenario.step(save=False)


