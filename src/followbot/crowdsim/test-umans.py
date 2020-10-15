import umans

u = umans.UMANS()
u.startSimulation("/home/cyrus/workspace2/UMANS/examples/1to1-90Degrees-ORCA.xml", 4)

# u.cleanUp()
print(u.getNumberOfAgents())
dt = u.getSimulationTimeStep()
agentData_1 = u.getAgentPositions()
agentData_2 = agentData_1.copy()

u.doSimulationSteps(3)

agentData_2[0].position_x = 30.4
u.setAgentPositions(agentData_2)

u.doSimulationSteps(3)
agentData_3 = u.getAgentPositions()

print(agentData_1[0].position_x)
print(agentData_2[0].position_x)
print(agentData_3[0].position_x)

dummy = 0


