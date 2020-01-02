from pedestrian import Pedestrian
from robot import FollowBot
import crowdsim.crowdsim as crowdsim
import numpy as np


class World:
    def __init__(self, n_peds, n_robots, sim_model="helbing"):
        self.n_peds = n_peds
        self.n_robots = n_robots
        self.objects = []
        self.robots = []
        for ii in range(n_robots):
            robot = FollowBot()
            robot.world = self
            self.robots.append(robot)

        self.sim = crowdsim.CrowdSim(sim_model)
        self.sim.initSimulation(n_peds + n_robots)

        self.crowds = []
        for ii in range(n_peds):
            ped_i = Pedestrian()
            ped_i.id = ii
            ped_i.world = self  # set world ptr
            self.crowds.append(ped_i)
            ped_i.radius = 0.25
            ped_i.pref_speed = 2.8
            self.sim.setAgentRadius(ii, ped_i.radius)
            self.sim.setAgentSpeed(ii, ped_i.pref_speed)
            self.sim.setAgentNeighborDist(ii, 3)
            self.sim.setAgentTimeHorizon(ii, 2)

    def add_object(self, obj):
        self.objects.append(obj)
        if hasattr(obj, 'line'):
            self.sim.addObstacleCoords(obj.line[0][0], obj.line[0][1], obj.line[1][0], obj.line[1][1])
        else:
            print('Circle objects are not supported!')
            exit(1)

    def set_ped_position(self, index, pos):
        self.crowds[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(index, pos[0], pos[1])

    def set_ped_velocity(self, index, vel):
        self.crowds[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(index, vel[0], vel[1])

    def set_ped_goal(self, index, goal):
        self.crowds[index].goal = np.array(goal, dtype=np.float)
        self.sim.setGoal(index, goal[0], goal[1])

    def set_robot_position(self, index, pos):
        self.robots[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(self.n_peds + index, pos[0], pos[1])

    def set_robot_velocity(self, index, vel):
        self.robots[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(self.n_peds + index, vel[0], vel[1])

    def set_robot_leader(self, index_robot, index_ped):
        self.robots[index_robot].leader_ped = self.crowds[index_ped]

    def step(self, dt):
        self.sim.doStep(dt)
        for ii in range(self.n_peds):
            try:
                p = self.sim.getCenterNext(ii)
                v = self.sim.getCenterVelocityNext(ii)
                self.set_ped_position(ii, p)
                self.set_ped_velocity(ii, v)
            except:
                pass

        for jj, robot in enumerate(self.robots):
            self.robots[jj].step(dt)
            self.sim.setPosition(self.n_peds + jj, robot.pos[0], robot.pos[1])
            self.sim.setVelocity(self.n_peds + jj, robot.vel[0], robot.vel[1])





