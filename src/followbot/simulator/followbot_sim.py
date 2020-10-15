import time
from datetime import datetime
from scipy.spatial.transform import Rotation
from followbot.gui.display import *
from followbot.robot_functions.follower_bot import FollowerBot
from followbot.robot_functions.robot import MyRobot
from followbot.scenarios.corridor_crowd import CorridorCrowd
from followbot.scenarios.group_crowd import GroupCrowd
from followbot.scenarios.roundtrip_scenario import RoundTrip
from followbot.scenarios.static_crowd import StaticCrowd
from followbot.simulator.world import World
from followbot.util.basic_geometry import Line
from followbot.scenarios.real_scenario import RealScenario
from followbot.robot_functions.tracking import PedestrianDetection
from followbot.util.transform import Transform
# from crowd_synthesis.crowd_synthesis import CrowdSynthesizer


def init_world(world_dim, title='followBot'):
    world = World(N_ped, N_robots)
    display = Display(world, world_dim, (960, 960), title)
    # world.pref_speed = 1.5  # FIXME : set it for sim as well
    return world, display


def setup_linear_peds():
    # FIXME: init objects
    world, display = init_world(world_dim=[[0, 8], [-5, 5]], title='linear')
    line_obstacles = [Line([5, 0], [5, 8]), Line([6, 0], [6, 9])]
    for l_obj in line_obstacles:
        world.add_obstacle(l_obj)

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
    world, display = init_world(world_dim=[[-5, 5], [-5, 5]], title='circle')
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
    outer_dim = 18
    world, display = init_world(world_dim=[[-outer_dim, outer_dim], [-outer_dim, outer_dim]], title='corridor')

    line_objects = [Line([0, 0], [10, 0]), Line([2, 7], [10, 7])]
    for l_obj in line_objects:
        world.add_obstacle(l_obj)

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


def run():
    ## scenario = setup_corridor()   # FixMe
    ## scenario = setup_circle()     # FixMe

    # scenario = RealScenario()
    # scenario.setup()

    # scenario = RoundTrip()
    # scenario.setup('powerlaw', flow_2d=True)

    # FixME: 4 main types of scenarios:
    #  1. Static groups (french people standing to talk all the weekend!)
    #  2. 1-D flow of singles (parades of bachelors!)
    #  3. 2-D flow of singles
    #  4. 2-D flow of couples (love street!)
    # scenario = StaticCrowd()
    # scenario.setup()

    # scenario = CorridorCrowd()
    # scenario.setup(biD_flow=False)

    scenario = CorridorCrowd()
    scenario.setup(biD_flow=True)

    # scenario = GroupCrowd()
    # scenario.setup()

    # FixMe: uncomment to use FollowerBot
    # robot = FollowerBot()
    # scenario.world.add_robot(robot)
    # robot.set_leader_ped(scenario.world.crowds[0])

    # FixMe: uncomment to use Std Robot
    robot = MyRobot()
    scenario.world.add_robot(robot)
    robot.init([0, 2])
    scenario.world.set_robot_goal(0, [100, 2])

    # Todo:
    #  capture the agents around the robot and extract the pattern
    #  1. Distance to (k)-nearest agent
    #  2. Velocity vector
    #  3. You can feed these data to a network with a pooling layer to classify different cases

    # crowd_syn = CrowdSynthesizer()
    # crowd_syn.extract_features(scenario.dataset)

    # lidar = robot.lidar
    # ped_detector = PedestrianDetection(robot.lidar.range_max, np.deg2rad(1 / robot.lidar.resolution))

    dt = 0.1

    # Write robot observations to file  # FixME: I start with recording ground truth values
    output_dir = "/home/cyrus/workspace2/ros-catkin/src/followbot/src/followbot/temp/robot_obsvs"
    output_filename = os.path.join(output_dir, type(scenario).__name__ + ".txt")
    output_file = open(output_filename, 'a+')

    frame_id = -1
    while True:
        # frame_id += 1
        frame_id = datetime.now().timestamp() * 1000  # Unix Time - in millisecond

        scenario.step(dt)
        robot = scenario.world.robots[0]

        # FixMe: This is for StdRobot
        if not isinstance(robot, FollowerBot):
            v_new_robot = scenario.world.sim.getCenterVelocityNext(scenario.n_peds)
            robot.vel = np.array(v_new_robot)

        peds_t = []

        for pid in range(scenario.n_peds):
            ped_i = scenario.world.crowds[pid]
            if robot.lidar.range_min < np.linalg.norm(robot.pos - ped_i.pos) < robot.lidar.range_max:
                output_file.write("%d %d %.3f %.3f %.3f %.3f\n" % (frame_id, pid,
                                                                   ped_i.pos[0], ped_i.pos[1],
                                                                   ped_i.vel[0], ped_i.vel[1]))
            peds_t.append(ped_i.pos)
        # pcf(peds_t)

        #  scenario.step_crowd()
        #  scenario.step_robot()
        # cur_t = scenario.cur_t

        # angles = np.arange(lidar.angle_min_radian(), lidar.angle_max_radian() - 1E-10, lidar.angle_increment_radian())
        # segments = ped_detector.segment_range(lidar.last_range_data, angles)
        # detections, walls = ped_detector.detect(segments, [0, 0])
        # robot.lidar_segments = segments
        # detections_tf = []
        # robot_rot = Rotation.from_euler('z', robot.orien, degrees=False)
        # robot_tf = Transform(np.array([robot.pos[0], robot.pos[1], 0]), robot_rot)
        # for det in detections:
        #     tf_trans, tf_orien = robot_tf.apply(np.array([det[0], det[1], 0]), Rotation.from_quat([0, 0, 0, 1]))
        #     detections_tf.append(np.array([tf_trans[0], tf_trans[1]]))
        # robot.detected_peds = detections_tf
        #
        # # robot_functions  #Todo
        # robot.tracks = robot.tracker.track(robot.detected_peds)

        # scenario.update_disply()

        # FixMe:
        # n_configs = 2
        # if scenario.display.event == pygame.K_p:
        #     inds = np.where(scenario.ped_valid[scenario.cur_t])
        #     gt_pnts = scenario.ped_poss[scenario.cur_t, np.where(scenario.ped_valid[scenario.cur_t])]
        #     crowd_syn.analyze_and_plot(np.array(robot.detected_peds),
        #                                gt_pnts.reshape((-1, 2)), n_configs)


if __name__ == '__main__':
    run()



