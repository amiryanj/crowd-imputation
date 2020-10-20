# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.scenarios.scenario import Scenario


class UmansScenario(Scenario):
    def __init__(self):
        super(UmansScenario, self).__init__()

    def setup(self):
        pass

    def step(self, save=False):
        pass


# def init_world(world_dim, title='followBot'):
#     world = World(N_ped, N_robots)
#     display = Visualizer(world, world_dim, (960, 960), title)
#     # world.pref_speed = 1.5  # FIXME : set it for sim as well
#     return world, display
#
#
# def setup_linear_peds():
#     # FIXME: init objects
#     world, display = init_world(world_dim=[[0, 8], [-5, 5]], title='linear')
#     line_obstacles = [Line([5, 0], [5, 8]), Line([6, 0], [6, 9])]
#     for l_obj in line_obstacles:
#         world.add_obstacle(l_obj)
#
#     # FIXME: init crowd
#     for ii in range(N_ped):
#         world.set_ped_position(ii, [0, ii])
#         world.set_ped_velocity(ii, [0, 0])
#         world.set_ped_goal(ii, [10, ii])
#
#     world.set_robot_position(0, [3, 3.2])
#     world.set_robot_leader(0, 0)
#     world.sim.setTime(0)
#     display.update()
#
#
# def setup_circle():
#     world, display = init_world(world_dim=[[-5, 5], [-5, 5]], title='circle')
#     for ii in range(N_ped):
#         theta = ii / N_ped * 2 * np.pi
#         world.set_ped_position(ii, np.array([np.cos(theta), np.sin(theta)]) * 5 + np.random.randn(1) * 0.01)
#         world.set_ped_goal(ii, np.array([np.cos(theta+np.pi), np.sin(theta+np.pi)]) * 5)
#         world.crowds[ii].color = agent_color
#     # world.set_robot_position(0, [-1, -1])
#     # world.set_robot_leader(0, 0)
#     world.sim.setTime(0)
#     display.update()
#
