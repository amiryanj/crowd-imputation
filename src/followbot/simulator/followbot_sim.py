import time
from datetime import datetime
from scipy.spatial.transform import Rotation
from followbot.gui.visualizer import *
from followbot.robot_functions.follower_bot import FollowerBot
from followbot.robot_functions.robot import MyRobot
from followbot.scenarios.corridor_scenario import CorridorScenario
from followbot.scenarios.grouping_scenario import GroupScenario
from followbot.scenarios.roundtrip_scenario import RoundTrip
from followbot.scenarios.static_crowd import StaticCorridorScenario
from followbot.simulator.world import World
from followbot.util.basic_geometry import Line
from followbot.scenarios.real_scenario import RealScenario
from followbot.robot_functions.tracking import PedestrianDetection
from followbot.util.transform import Transform
# from followbot.crowd_synthesis.crowd_synthesis import CrowdSynthesizer


def run():
    num_robot_belief_worlds = 2
    # =======================================
    # Main scenarios
    # =======================================
    # # Static groups (french people standing to talk all the weekend!)
    # scenario = StaticCrowd()
    # scenario.setup()

    # # Uni-D flow of singles (parades of bachelors!)
    # scenario = CorridorCrowd()
    # scenario.setup(biD_flow=False)

    # # Bi-D flow of singles
    scenario = CorridorScenario()
    scenario.setup(biD_flow=True)

    # # Bi-D flow of couples (love street!)
    # scenario = GroupCrowd()
    # scenario.setup()
    # =======================================
    

    # =======================================
    # Older scenarios
    # =======================================
    # scenario = setup_corridor()   # FixMe
    # scenario = setup_circle()     # FixMe

    # scenario = RealScenario()
    # scenario.setup()

    # scenario = RoundTrip()
    # scenario.setup('powerlaw', flow_2d=True)
    # =======================================
    print(type(scenario).__name__)
    scenario.visualizer = Visualizer(scenario.world, scenario.world.world_dim,
                                     subViewRowCount=1 + num_robot_belief_worlds,
                                     subViewColCount=1,
                                     caption=type(scenario).__name__)


    # =======================================
    # Choose the robot type
    # =======================================
    # FixMe: uncomment to use FollowerBot
    # robot = FollowerBot()
    # scenario.world.add_robot(robot)
    # robot.set_leader_ped(scenario.world.crowds[0])

    # FixMe: uncomment to use Std Robot
    robot = MyRobot(numBeliefWorlds=num_robot_belief_worlds)
    scenario.world.add_robot(robot)
    robot.init([-15, 0])
    scenario.world.set_robot_goal(0, [100, 0])
    # =======================================
    
    # Todo:
    #  capture the agents around the robot and extract the pattern
    #  1. Distance to (k)-nearest agent
    #  2. Velocity vector
    #  3. You can feed these data to a network with a pooling layer to classify different cases

    # crowd_syn = CrowdSynthesizer()
    # crowd_syn.extract_features(scenario.dataset)

    lidar = robot.lidar
    ped_detector = PedestrianDetection(robot.lidar.range_max, np.deg2rad(1 / robot.lidar.resolution))

    dt = 0.1

    # FixMe: Write robot observations to file  
    # Todo: At the moment it only records the ground truth locations
    #  It can be the output of tracking module or even after using RWTH's DROW tracker
    # =======================================
    output_dir = "/home/cyrus/workspace2/ros-catkin/src/followbot/src/followbot/temp/robot_obsvs"
    output_filename = os.path.join(output_dir, type(scenario).__name__ + ".txt")
    output_file = open(output_filename, 'a+')
    # =======================================

    frame_id = -1
    robot = scenario.world.robots[0]

    # ****************** Program Loop *******************
    while True:
        # frame_id += 1
        frame_id = datetime.now().timestamp() * 1000  # Unix Time - in millisecond

        scenario.step(dt)

        # FixMe: This is for StdRobot
        # if not isinstance(robot, FollowerBot):
        #     v_new_robot = scenario.world.sim.getCenterVelocityNext(scenario.n_peds)
        #     robot.vel = np.array(v_new_robot)

        # Write detected pedestrians to the output file
        # =======================================
        for pid in range(scenario.n_peds):
            ped_i = scenario.world.crowds[pid]
            if robot.lidar.range_min < np.linalg.norm(robot.pos - ped_i.pos) < robot.lidar.range_max:
                output_file.write("%d %d %.3f %.3f %.3f %.3f\n" % (frame_id, pid,
                                                                   ped_i.pos[0], ped_i.pos[1],
                                                                   ped_i.vel[0], ped_i.vel[1]))
        # =======================================

        # peds_t = []
        # for pid in range(scenario.n_peds):
        #     peds_t.append(ped_i.pos)
        # pcf(peds_t)

        # cur_t = scenario.cur_t

        # Casting LiDAR rays to get detections
        # ====================================
        angles = np.arange(lidar.angle_min_radian(), lidar.angle_max_radian() - 1E-10, lidar.angle_increment_radian())
        segments = ped_detector.segment_range(lidar.last_range_data, angles)
        detections, walls = ped_detector.detect(segments, [0, 0])
        robot.lidar_segments = segments
        # ====================================

        # Transform (rotate + translate) the detections, given the robot pose
        # ====================================
        detections_tf = []
        robot_rot = Rotation.from_euler('z', robot.orien, degrees=False)
        robot_tf = Transform(np.array([robot.pos[0], robot.pos[1], 0]), robot_rot)
        for det in detections:
            tf_trans, tf_orien = robot_tf.apply(np.array([det[0], det[1], 0]), Rotation.from_quat([0, 0, 0, 1]))
            detections_tf.append(np.array([tf_trans[0], tf_trans[1]]))
        robot.detected_peds = detections_tf
        # ====================================

        # Todo: robot_functions
        # ====================================
        robot.tracks = robot.tracker.track(robot.detected_peds)
        # ====================================

        # ToDo:
        #  Robot Beliefs
        # ====================================
        n_beliefs = 2
        # if scenario.visualizer.event == pygame.K_p:
        # inds = np.where(scenario.ped_valid[scenario.cur_t])
        # gt_pnts = scenario.ped_poss[scenario.cur_t, np.where(scenario.ped_valid[scenario.cur_t])]
        # crowd_syn.analyze_and_plot(np.array(robot.detected_peds),
        #                            gt_pnts.reshape((-1, 2)), n_beliefs)
        # ====================================



if __name__ == '__main__':
    run()



