# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import followbot.crowdsim.umans as umans
import xml.etree.ElementTree as ET
import random
import os

from followbot.util.basic_geometry import Line


class CrowdSimUMANS:

    def __init__(self, config):
        """
            :param confige: UMANS-specific config file which contains:
                                    1. World Type (Infinite / Toric)
                                    2. Policies (Sim Algorithm): address of the policy file
                                               * Dutra (no obstacle support)
                                               * FOEAvoidance (no obstacle support)
                                               * Karamouzas
                                               * Moussaid
                                               * ORCA (no obstacle support)
                                               * Paris (no obstacle support)
                                               * PLEdestrians
                                               * PowerLaw (no obstacle support)
                                               * RVO
                                               * SocialForces
                                               * VanToll
                                    3. Agents: address of the agents file
                                               - Position
                                               - Goal
                                               - Radius
                                               - Pref/max speed/acceleration
        """

        self.sim = umans.UMANS()
        self.agent_data_cache = {}
        self.obstacles = []

        if os.path.exists(config):
            config_file = config
        else:
            config_file = os.path.join(os.path.dirname(__file__), 'umans-config', config + '-Empty.xml')
        if not os.path.exists(config_file):
            print("invalid simulation name to initiate UMANS")
            exit(-1)
        obstacle_file = ""
        try:
            config_root_elem = ET.parse(config_file).getroot()
            for child in config_root_elem.iter('Obstacles'):
                obstacle_file = os.path.join(os.path.dirname(config_file), child.attrib['file'])
                self.obstacles = self.loadObstaclesXML(obstacle_file)
        except:
            print('Failed to load obstacles')

        self.startSimulation(config_file)
        self.getSimulationTimeStep()

    def startSimulation(self, configFile, numThreads=4):
        """
        :param numThreads: for parallel computing
        :return: bool (True if the call has been success)
        """

        # self.cleanUp()
        self.agent_data_cache.clear()
        return self.sim.startSimulation(configFile, numThreads)

    def initSimulation(self, numAgents):
        """ This method initiate a UMANS object with the given num_Agents. """
        for ii in range(numAgents):
            id = self.addAgent(x=random.randrange(20), y=random.randrange(20),
                               # default values from UMANS
                               radius=0.24,
                               prefSpeed=1.3,
                               maxSpeed=1.8,
                               maxAcceleration=5.0,
                               policyID=0, customID=-1)

        agent_data = self.sim.getAgentPositions()
        for agent in agent_data:
            self.agent_data_cache[agent.id] = agent

    def getSimulationTimeStep(self):
        return self.sim.getSimulationTimeStep()

    def doSimulationSteps(self, nrSteps=1):
        self.sim.doSimulationSteps(nrSteps)
        agent_data = self.sim.getAgentPositions()

        self.agent_data_cache.clear()
        for agent in agent_data:
            self.agent_data_cache[agent.id] = agent

    def getAgentsData(self):
        return self.sim.getAgentPositions()

    def setAgentsData(self, agentData: list):
        self.agent_data_cache.clear()
        for agent in agentData:
            self.agent_data_cache[agent.id] = agent
        return self.sim.setAgentPositions(agentData)

    def setAgentGoal(self, id: int, x: float, y: float):
        return self.sim.setAgentGoal(id, x, y)

    # load obstacles config file
    # ===============================================
    def loadObstaclesXML(self, xml_file):
        obstacles = []
        obs_root = ET.parse(xml_file).getroot()
        for obs_elem in obs_root.iter('Obstacle'):
            obs_i = []
            for point_elem in obs_elem.iter('Point'):
                px = point_elem.attrib['x']
                py = point_elem.attrib['y']
                obs_i.append([px, py])
            if len(obs_i):
                obs_i.append(obs_i[0])
                for jj in range(len(obs_i)-1):
                    obstacles.append(Line(obs_i[jj], obs_i[jj+1]))
        return obstacles

    # functions that do nothing!
    # ===============================================
    def setAgentParameters(self, id, radius, prefSpeed, maxNeighborDist, maxTimeHorizon):
        print("*** Umans is not able to modify parameter through code. You can update the config file.")
        return False

    def setTime(self, t):
        print("*** Umans has no time variable!")
        return False

    # functions to be compatible with CrowdBag
    # ===============================================
    def setPosition(self, id, pos_x, pos_y):
        self.agent_data_cache[id].position_x = pos_x
        self.agent_data_cache[id].position_y = pos_y
        self.setAgentsData(list(self.agent_data_cache.values()))

    def setVelocity(self, id, vel_x, vel_y):
        self.agent_data_cache[id].velocity_x = vel_x
        self.agent_data_cache[id].velocity_y = vel_y
        self.setAgentsData(list(self.agent_data_cache.values()))

    def setGoal(self, id, goal_x, goal_y):
        self.setAgentGoal(id, goal_x, goal_y)

    def getPosition_x(self, id):
        return self.agent_data_cache[id].position_x

    def getPosition_y(self, id):
        return self.agent_data_cache[id].position_y

    def getCenterNext(self, id):
        return [self.agent_data_cache[id].position_x, self.agent_data_cache[id].position_y]

    def getVelocity_x(self, id):
        return self.agent_data_cache[id].velocity_x

    def getVelocity_y(self, id):
        return self.agent_data_cache[id].velocity_y

    def getCenterVelocityNext(self, id):
        return [self.agent_data_cache[id].velocity_x, self.agent_data_cache[id].velocity_y]

    def doStep(self, dt):
        self.doSimulationSteps(1)
    # ===============================================

    def addAgent(self, x: float, y: float, radius: float,
                 prefSpeed: float, maxSpeed: float, maxAcceleration: float,
                 policyID: int = 0, customID: int = -1):
        return self.sim.addAgent(x, y, radius, prefSpeed, maxSpeed, maxAcceleration, policyID, customID)

    def removeAgent(self, id: int):
        return self.sim.removeAgent(id)

    def getNumberOfAgents(self):
        return self.sim.getNumberOfAgents()

    def getNumberOfObstacles(self):
        return self.sim.getNumberOfObstacles()

    def cleanUp(self):
        self.agent_data_cache.clear()
        return self.sim.cleanUp()


if __name__ == "__main__":
    sim = CrowdSimUMANS('PowerLaw')
    sim.initSimulation(10)
    print(sim.getNumberOfObstacles())