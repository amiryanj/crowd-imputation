import numpy as np

from followbot.crowdsim.pedestrian import Pedestrian
# from followbot.robot_functions.follower_bot import FollowerBot
from followbot.robot_functions.robot import MyRobot
import followbot.crowdsim.crowdsim as crowdsim
import followbot.crowdsim.umans_api as umans_api


class World:
    def __init__(self, n_peds, n_robots, world_dim, sim_model="SocialForces", biped=False):
        self.pause = False
        self.n_peds = n_peds
        self.n_robots = n_robots
        self.robots = []
        self.obstacles = []
        self.world_dim = world_dim

        # self.sim = crowdsim.CrowdSim(sim_model)
        self.sim = umans_api.CrowdSimUMANS(sim_model)

        self.sim.initSimulation(n_peds + n_robots)
        self.inertia_coeff = 0.4  # larger, more inertia, zero means no inertia

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
            except:
                print('obstacles should be defined in config file')
                exit(-1)
        else:
            print('Circle objects are not supported in crowd simulation!')

    def set_ped_position(self, index, pos):
        self.crowds[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(index, pos[0], pos[1])
        u, v = self.mapping_to_grid(pos[0], pos[1])
        if 0 <= u <= self.walkable.shape[0] and 0 <= v <= self.walkable.shape[1]:
            self.crowds[index].trajectory.append(pos)
        if len(self.crowds[index].trajectory) > 150:
            del self.crowds[index].trajectory[0]

    def set_ped_velocity(self, index, vel):
        self.crowds[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(index, vel[0], vel[1])

    def set_ped_goal(self, index, goal):
        self.crowds[index].goal = np.array(goal, dtype=np.float)
        self.sim.setGoal(index, goal[0], goal[1])

    def set_robot_position(self, index, pos):
        """Robot is assumed to be the last agent in the CrowdSim agents"""
        self.robots[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(self.n_peds + index, pos[0], pos[1])

    def set_robot_velocity(self, index, vel):
        """Robot is assumed to be the last agent in the CrowdSim agents"""
        self.robots[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(self.n_peds + index, vel[0], vel[1])

    def set_robot_goal(self, index, goal):
        self.robots[index].goal = np.array(goal, dtype=np.float)
        self.sim.setGoal(self.n_peds + index, goal[0], goal[1])

    def step_robot(self, dt):
        for jj, robot in enumerate(self.robots):

            update_pom = False   # Fixme
            self.robots[jj].lidar.scan(self, update_pom, walkable_area=self.walkable)
            if update_pom:
                pom_new = self.robots[jj].lidar.last_occupancy_gridmap.copy()
                seen_area_indices = np.where(pom_new != 0)
                self.POM[:] = self.POM[:] * 0.4 + 0.5
                self.POM[seen_area_indices] = 0
                for track in self.robots[jj].tracks:
                    if track.coasted: continue
                    px, py = track.position()
                    u, v = self.mapping_to_grid(px, py)
                    if 0 <= u < self.POM.shape[0] and 0 <= v < self.POM.shape[1]:
                        self.POM[u - 2:u + 2, v - 2:v + 2] = 1

            self.robots[jj].step(dt)
            self.sim.setPosition(self.n_peds + jj, robot.pos[0], robot.pos[1])
            self.sim.setVelocity(self.n_peds + jj, robot.vel[0], robot.vel[1])





