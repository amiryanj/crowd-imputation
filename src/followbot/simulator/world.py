import numpy as np

from followbot.crowdsim.pedestrian import Pedestrian
from followbot.robot_functions.robot import MyRobot
import followbot.crowdsim.crowdsim as crowdsim


class World:
    def __init__(self, n_peds, n_robots, sim_model="helbing", biped=False):
        self.pause = False
        self.n_peds = n_peds
        self.n_robots = n_robots
        self.robots = []
        self.objects = []
        self.walkable = []  # would be a constant-matrix that is determined by the scenario maker
        self.POM = []  # probabilistic occupancy map
        self.mapping_to_grid = []

        for ii in range(n_robots):
            robot = MyRobot()
            robot.world = self
            self.robots.append(robot)

        self.sim = crowdsim.CrowdSim(sim_model)
        self.sim.initSimulation(n_peds + n_robots)
        self.inertia_coeff = 0.8  # larger, more inertia, zero means no inertia

        self.crowds = []
        for ii in range(n_peds):
            ped_i = Pedestrian(biped=biped)
            ped_i.id = ii
            ped_i.world = self  # set world ptr
            self.crowds.append(ped_i)
            ped_i.radius = 0.25
            ped_i.pref_speed = np.random.uniform(1.2, 1.7)
            self.sim.setAgentRadius(ii, ped_i.radius * 2)
            self.sim.setAgentSpeed(ii, ped_i.pref_speed)
            self.sim.setAgentNeighborDist(ii, 3)
            self.sim.setAgentTimeHorizon(ii, 2)

    def add_object(self, obj):
        self.objects.append(obj)
        if hasattr(obj, 'line'):
            self.sim.addObstacleCoords(obj.line[0][0], obj.line[0][1], obj.line[1][0], obj.line[1][1])
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
        self.robots[index].pos = np.array(pos, dtype=np.float)
        self.sim.setPosition(self.n_peds + index, pos[0], pos[1])

    def set_robot_velocity(self, index, vel):
        self.robots[index].vel = np.array(vel, dtype=np.float)
        self.sim.setVelocity(self.n_peds + index, vel[0], vel[1])

    def set_robot_leader(self, index_robot, index_ped):
        self.robots[index_robot].leader_ped = self.crowds[index_ped]

    def step_crowd(self, dt):
        self.sim.doStep(dt)
        for ii in range(self.n_peds):
            try:
                p = self.sim.getCenterNext(ii)
                v = self.sim.getCenterVelocityNext(ii)
                # apply inertia
                v_new = np.array(v) * (1-self.inertia_coeff) + self.crowds[ii].vel * self.inertia_coeff
                p_new = self.crowds[ii].pos + v_new * dt
                self.set_ped_position(ii, p_new)
                self.set_ped_velocity(ii, v_new)
            except:
                print('exception occurred in running crowd sim')

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





