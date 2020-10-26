import numpy as np

import followbot.crowdsim.crowdsim as crowdsim
import followbot.crowdsim.umans_api as umans_api
from followbot.crowdsim.pedestrian import Pedestrian


class World:
    def __init__(self, n_peds, n_robots, world_dim, sim_model="RVO", biped=False):
        self.pause = False
        self.n_peds = n_peds
        self.n_robots = n_robots
        self.robots = []
        self.obstacles = []
        self.world_dim = world_dim

        # # Deprecated! -> use UMANS
        # self.sim = crowdsim.CrowdSim(sim_model)
        self.sim = umans_api.CrowdSimUMANS(sim_model)

        self.sim.initSimulation(n_peds + n_robots)
        self.inertia_coeff = 0.25  # larger, more inertia, zero means no inertia

        self.crowds = []
        for ii in range(n_peds):
            ped_i = Pedestrian(biped=biped)
            ped_i.id = ii
            ped_i.world = self  # set world ptr
            self.crowds.append(ped_i)
            ped_i.radius = 0.25
            ped_i.pref_speed = np.random.uniform(1.2, 1.7)
            # self.sim.setAgentParameters(ii, radius=ped_i.radius * 2,
            #                             prefSpeed=ped_i.pref_speed, maxNeighborDist=3, maxTimeHorizon=2)
            # self.sim.setAgentRadius(ii, ped_i.radius * 2)
            # self.sim.setAgentSpeed(ii, ped_i.pref_speed)
            # self.sim.setAgentNeighborDist(ii, 3)
            # self.sim.setAgentTimeHorizon(ii, 2)

    def add_robot(self, robot):
        robot.real_world = self
        self.robots.append(robot)

    def add_obstacle(self, obj):
        self.obstacles.append(obj)
        if hasattr(obj, 'line'):
            try:
                self.sim.addObstacleCoords(obj.draw_line[0][0], obj.draw_line[0][1], obj.draw_line[1][0], obj.draw_line[1][1])
            except Exception:
                raise ValueError('obstacles should be defined in config file')
        else:
            print('Circle objects are not supported in crowd simulation!')

    def set_ped_position(self, index, pos):
        self.crowds[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(index, pos[0], pos[1])

        # append the position to pedestrian trajectory
        if len(self.crowds[index].trajectory) and \
            np.linalg.norm(self.crowds[index].trajectory[-1] - np.array(pos)) > 2.0:
            self.crowds[index].trajectory.clear()
        self.crowds[index].trajectory.append(np.array(pos))
        if len(self.crowds[index].trajectory) > 150:
            del self.crowds[index].trajectory[0]

    def set_ped_velocity(self, index, vel):
        self.crowds[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(index, vel[0], vel[1])

    def set_ped_goal(self, index, goal):
        self.crowds[index].goal = np.array(goal, dtype=np.float)
        self.sim.setGoal(index, goal[0], goal[1])

    def set_robot_position(self, index, pos):
        """Notice: Robot is assumed to be the last agent in the CrowdSim agents"""
        self.robots[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(self.n_peds + index, pos[0], pos[1])

    def set_robot_velocity(self, index, vel):
        """Notice: Robot is assumed to be the last agent in the CrowdSim agents"""
        self.robots[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(self.n_peds + index, vel[0], vel[1])

    def set_robot_goal(self, index, goal):
        self.robots[index].goal = np.array(goal, dtype=np.float)
        self.sim.setGoal(self.n_peds + index, goal[0], goal[1])



