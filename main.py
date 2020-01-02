from display import Display
from world import World
from basics_2d import Line, Circle
from robot import FollowBot
from pedestrian import Pedestrian
import numpy as np
import crowdsim.crowdsim as crowdsim


N_ped = 10
N_robots = 1
world = World(N_ped, N_robots, 'rvo2')
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
    for ii in range(N_ped):
        theta = ii / N_ped * 2 * np.pi
        world.set_ped_position(ii, np.array([np.cos(theta), np.sin(theta)]) * 5 + 5 + np.random.randn(1) * 0.01)
        world.set_ped_goal(ii, np.array([np.cos(theta+np.pi), np.sin(theta+np.pi)]) * 5 + 5)
    world.set_robot_position(0, [-1, -1])
    world.set_robot_leader(0, 0)


def corridor():
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


# corridor()
circle()
# TODO: setup pygame
display = Display(world, (10, 10), (640, 640), 'followBot')
world.sim.setTime(0)
display.update()
pause = False
for tt in range(0, 10000):
    if not pause:
        world.step(0.01)
    toggle_pause = display.update()
    if toggle_pause: pause = not pause



