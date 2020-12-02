import time
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn import neighbors
from numpy.linalg import norm as norm

from followbot.crowdsim.pedestrian import Pedestrian
from followbot.gui.visualizer import *
from followbot.robot_functions.bivariate_gaussian import BivariateGaussianMixtureModel, BivariateGaussian, draw_bgmm
from followbot.robot_functions.dr_spaam_detector import DrSpaamDetector
from followbot.robot_functions.flow_classifier import FlowClassifier
from followbot.robot_functions.pairwise_distribution import PairwiseDistribution
from followbot.robot_functions.human_detection import PedestrianDetection
from followbot.robot_functions.robot import MyRobot
from followbot.robot_functions.follower_bot import FollowerBot
from followbot.robot_functions.robot_replace_human import RobotReplaceHuman
from followbot.scenarios.corridor_scenario import CorridorScenario
from followbot.scenarios.hermes_scenario import HermesScenario
from followbot.scenarios.roundtrip_scenario import RoundTrip
from followbot.scenarios.static_crowd import StaticCorridorScenario
from followbot.scenarios.real_scenario import RealScenario
from followbot.scenarios.world import World
from followbot.util.basic_geometry import Line
from followbot.util.transform import Transform
# from followbot.crowd_synthesis.crowd_synthesis import CrowdSynthesizer

import matplotlib

matplotlib.use('TkAgg')

np.random.seed(4)
random.seed(4)

# Config
biped_mode = True
lidar_enabled = True
write_trajectories = False
visualizer_enabled = False
tracking_enabled = False
synthesis_enabled = False
num_robot_hypothesis_worlds = 2


def exec_scenario(scenario):
    if visualizer_enabled:
        scenario.visualizer = Visualizer(scenario.world, scenario.world.world_dim,
                                         subViewRowCount=1 + num_robot_hypothesis_worlds,
                                         subViewColCount=1,
                                         caption=type(scenario).__name__)

    # Choose the robot type
    if isinstance(scenario, RealScenario):
        robot = RobotReplaceHuman(scenario.robot_poss, scenario.robot_vels, scenario.world,
                                  num_robot_hypothesis_worlds)
        scenario.world.add_robot(robot)
        robot.init()
        dt = 1 / scenario.fps
        # robot = FollowerBot()  # deprecated ?
        # robot.set_leader_ped(scenario.world.crowds[0])

    else:  # use Std Robot
        robot = MyRobot(scenario.world, prefSpeed=0.8, numHypothesisWorlds=num_robot_hypothesis_worlds)
        scenario.world.add_robot(robot)
        dt = 0.1
        # Robot: starting in up side of the corridor, facing toward right end of the corridor
        robot.init([-25, 1.5])
        scenario.world.set_robot_goal(0, [1000, 0])
    global robot
    # =======================================

    # Todo:
    #  capture the agents around the robot and extract the pattern
    #  1. Distance to (K)-nearest agent
    #  2. Velocity vector
    #  3. You can feed these data to a network with a pooling layer to classify different cases

    # crowd_syn = CrowdSynthesizer()
    # crowd_syn.extract_features(scenario.dataset)

    if biped_mode:  # use dr-spaam
        detector = DrSpaamDetector(robot.lidar.num_pts(),
                                   np.degrees(robot.lidar.angle_increment_radian()),
                                   gpu=True)
    else:
        detector = PedestrianDetection(robot.lidar.range_max, np.deg2rad(1 / robot.lidar.resolution))

    # FixMe: Write robot observations to file
    # Todo: At the moment it only records the ground truth locations
    #  It can be the output of tracking module or even after using RWTH's DROW tracker
    # =======================================
    # output_dir = "/home/cyrus/workspace2/ros-catkin/src/followbot/src/followbot/temp/robot_obsvs"
    # output_filename = os.path.join(output_dir, type(scenario).__name__ + ".txt")
    # output_file = open(output_filename, 'a+')
    # =======================================

    frame_id = -1
    robot.blind_spot_projector = PairwiseDistribution()

    # ****************** Program Loop *******************
    while True:
        frame_id += 1

        not_finished = scenario.step(dt, lidar_enabled)
        if not not_finished:
            break
        # print("Robot pos =", robot.pos)
        if scenario.world.pause:
            continue

        # Write detected pedestrians to the output file
        # =======================================
        if write_trajectories:
            time_stamp = int(datetime.now().timestamp() * 1000) / 1000.  # Unix Time - in millisecond
            for pid in range(scenario.n_peds):
                ped_i = scenario.world.crowds[pid]
                if robot.lidar.range_min < np.linalg.norm(robot.pos - ped_i.pos) < robot.lidar.range_max:
                    output_file.write("%d %d %.3f %.3f %.3f %.3f\n" % (time_stamp, pid,
                                                                       ped_i.pos[0], ped_i.pos[1],
                                                                       ped_i.vel[0], ped_i.vel[1]))
        # =======================================

        # ***************************************************
        # ***************************************************
        if tracking_enabled:
            # Detection (Base method / DR-SPAAM)
            if biped_mode:
                detections, dets_cls = detector.detect(robot.lidar.data.last_range_data)
            else:
                angles = np.arange(robot.lidar.angle_min_radian(), robot.lidar.angle_max_radian() - 1E-10,
                                   robot.lidar.angle_increment_radian())
                segments = detector.cluster_range_data(robot.lidar.data.last_range_data, angles)
                detections, _ = detector.detect(segments, [0, 0])
                robot.lidar_segments = segments
            # ====================================

            # Transform (rotate + translate) detections, given the robot pose
            # ====================================
            detections_tf = []
            robot_rot = Rotation.from_euler('z', robot.orien, degrees=False)
            robot_tf = Transform(np.array([robot.pos[0], robot.pos[1], 0]), robot_rot)
            for det in detections:
                tf_trans, tf_orien = robot_tf.apply(np.array([det[0], det[1], 0]), Rotation.from_quat([0, 0, 0, 1]))
                detections_tf.append(np.array([tf_trans[0], tf_trans[1]]))
            robot.detected_peds = detections_tf
            # ====================================

            # ====================================
            robot.tracks = robot.tracker.track(robot.detected_peds, scenario.world.time)
            # robot should be added to the list of tracks, bcuz it is impacting the flow
            tracks_loc = np.array([tr.kf.x[:2] for tr in robot.tracks if not tr.coasted] + [robot.pos])
            tracks_vel = np.array([tr.kf.x[2:4] for tr in robot.tracks if not tr.coasted] + [robot.vel])
            n_tracked_agents = len(tracks_loc)
            # ====================================

        # calc the occupancy map and crowd flow map
        # ====================================
        # robot.occupancy_map.fill(0)
        # robot.crowd_flow_map.fill(0)
        # for tr in robot._tracks:
        #     robot.occupancy_map.set(tr.kf.x[:2], 1)
        #     v_ped = tr.kf.x[2:4]
        #     angle_between_robot_motion_and_ped = np.dot(robot.vel, v_ped) / (norm(robot.vel) * norm(v_ped) + 1E-6)
        #     robot.crowd_flow_map.set(tr.kf.x[:2], [norm(v_ped), angle_between_robot_motion_and_ped])

        # classify the flow type of each agent
        # ================================

        # ***************************************************
        # ***************************************************
        if synthesis_enabled:
            # calc blind spots
            # ================================
            robot.blind_spot_map.fill(1)  # everywhere is in blind_spot_map if it's not!
            rays = robot.lidar.data.last_rotated_rays
            for ii in range(len(rays)):
                ray_i = rays[ii]
                scan_i = robot.lidar.data.last_points[ii]
                white_line = [robot.pos, scan_i]
                line_len = np.linalg.norm(white_line[1] - white_line[0])

                for z in np.arange(0, line_len / robot.lidar.range_max, 0.01):
                    px, py = z * ray_i[1] + (1 - z) * ray_i[0]
                    robot.blind_spot_map.set([px, py], 0)
            # =================================

            # flow estimation: by `multiple gaussian`
            # ================================
            agents_vel_polar = np.array([[np.linalg.norm(v), np.arctan2(v[1], v[0])] for v in tracks_vel])
            agents_flow_class = FlowClassifier().classify(tracks_loc, tracks_vel)
            bgm = BivariateGaussianMixtureModel()
            for i in range(n_tracked_agents):
                # FixMe: filter still agents
                if norm(tracks_vel[i]) < 0.1:
                    continue
                bgm.add_component(BivariateGaussian(tracks_loc[i][0], tracks_loc[i][1],
                                                    sigma_x=agents_vel_polar[i][0] / 5 + 0.1, sigma_y=0.1,
                                                    theta=agents_vel_polar[i][1]),
                                  weight=1, target=agents_flow_class[i].id)
            x_min, x_max = scenario.world.world_dim[0][0], scenario.world.world_dim[0][1]
            y_min, y_max = scenario.world.world_dim[1][0], scenario.world.world_dim[1][1]
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / robot.mapped_array_resolution),
                                 np.arange(y_min, y_max, 1 / robot.mapped_array_resolution))
            robot.crowd_flow_map.data = bgm.classify_kNN(xx, yy).T
            # draw_bgmm(bgm, xx, yy)  # => for paper
            # =================================

            # Crowd Synthesis: Using Pairwise Distances
            # =================================
            robot.blind_spot_projector.add_frame(tracks_loc, tracks_vel, agents_flow_class, dt)
            robot.blind_spot_projector.update_histogram(smooth=True)
            robot.blind_spot_projector.plot()
            for robot_hypo in robot.hypothesis_worlds:
                # initialize once
                if scenario.world.time > 0.5:  # and len(robot_hypo.crowds) == 0
                    synthetic_agents = robot.blind_spot_projector.synthesis(tracks_loc, tracks_vel,
                                                                            walkable_map=robot_hypo.walkable_map,
                                                                            blind_spot_map=robot.blind_spot_map,
                                                                            crowd_flow_map=robot.crowd_flow_map)
                    robot_hypo.crowds = [Pedestrian(tracks_loc[i], tracks_vel[i], False, synthetic=False,
                                                    color=GREEN_COLOR)
                                         for i in range(n_tracked_agents)] + synthetic_agents
                # evolve (ToDo)
                else:
                    for ped in robot_hypo.crowds:
                        if ped.synthetic:
                            ped.pos = np.array(ped.pos) + np.array(ped.vel) * dt
                            if robot.blind_spot_map.get(ped.pos) < 0.1:
                                del ped
                                # print("delete the synthetic agent")
            # # pcf(peds_t)
            # =================================

        yield frame_id, robot


if __name__ == '__main__':
    # =======================================
    # Main scenarios
    # =======================================
    # # Static groups (french people standing to talk all the weekend!)
    # _scenario = StaticCrowd()
    # _scenario.setup()

    # # Uni-D flow of singles (parades of bachelors!)
    # _scenario = CorridorCrowd()
    # _scenario.setup(biD_flow=False)

    # # Bi-D flow of singles
    # _scenario = CorridorScenario(biD_flow=False, group_size_choices=[2], corridor_len=60)
    # _scenario.setup()

    # # Bi-D flow of couples (love street!)
    # _scenario = GroupCrowd()
    # _scenario.setup()
    # =======================================

    # =======================================
    # Older scenarios
    # =======================================
    # _scenario = setup_corridor()   # FixMe
    # _scenario = setup_circle()     # FixMe

    _scenario = HermesScenario()
    _scenario.setup(os.path.abspath(os.path.join(__file__, "../../../..",
                                                 "config/followbot_sim/real_scenario_config.yaml")), biped=biped_mode)

    # _scenario = RoundTrip()
    # _scenario.setup('powerlaw', flow_2d=True)
    # =======================================
    print(type(scenario).__name__)
    for exec_frame in exec_scenario(_scenario):
        print(exec_frame)
        print(robot.lidar.data.last_range_data)
