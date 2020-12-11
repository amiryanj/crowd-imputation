import random
from datetime import datetime
from scipy.spatial.transform import Rotation
from numpy.linalg import norm as norm

from followbot.crowdsim.pedestrian import Pedestrian
from followbot.gui.visualizer import *
from followbot.robot_functions.bivariate_gaussian import BivariateGaussianMixtureModel, BivariateGaussian
from followbot.robot_functions.dr_spaam_detector import DrSpaamDetector
from followbot.robot_functions.flow_classifier import FlowClassifier
from followbot.robot_functions.social_ties import SocialTiePDF
from followbot.robot_functions.human_detection import PedestrianDetection
from followbot.robot_functions.robot import MyRobot
from followbot.robot_functions.robot_replace_human import RobotReplaceHuman
from followbot.util.video_player import DatasetVideoPlayer
from followbot.scenarios.hermes_scenario import HermesScenario
from followbot.scenarios.real_scenario import RealScenario
from followbot.util.transform import Transform
# from followbot.crowd_synthesis.crowd_synthesis import CrowdSynthesizer

import matplotlib
matplotlib.use('TkAgg')

# Config
BIPED_MODE = False
LIDAR_ENABLED = True
write_trajectories = False
VISUALIZER_ENABLED = True
VIDEO_ENABLED = False
TRACKING_ENABLED = True
SYNTHESIS_ENABLED = True
NUM_HYPOTHESES = 2

RAND_SEED = 4
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


def exec_scenario(scenario):
    if VISUALIZER_ENABLED:
        scenario.visualizer = Visualizer(scenario.world, scenario.world.world_dim,
                                         subViewRowCount=1 + NUM_HYPOTHESES,
                                         subViewColCount=1,
                                         caption=type(scenario).__name__)

    # Choose the robot type
    if isinstance(scenario, RealScenario):
        robot = RobotReplaceHuman(scenario.fps, scenario.robot_poss, scenario.robot_vels, scenario.world,
                                  NUM_HYPOTHESES)
        scenario.world.add_robot(robot)
        robot.init()
        dt = 1 / scenario.fps
        # robot = FollowerBot()  # deprecated ?
        # robot.set_leader_ped(scenario.world.crowds[0])

    else:  # use Std Robot
        robot = MyRobot(scenario.world, prefSpeed=0.8, numHypothesisWorlds=NUM_HYPOTHESES, sensor_fps=scenario.fps)
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

    if BIPED_MODE:  # use dr-spaam
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
    robot.blind_spot_projector = SocialTiePDF()

    # ****************** Program Loop *******************
    while True:
        not_finished = scenario.step(dt, LIDAR_ENABLED)
        if not not_finished:
            break
        if scenario.world.pause:
            continue
        frame_id = scenario.world.original_frame_id

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
        if TRACKING_ENABLED:
            # Detection (Base method / DR-SPAAM)
            if BIPED_MODE:
                detections, dets_cls = detector.detect(robot.lidar.data.last_range_data)
            else:
                angles = np.arange(robot.lidar.angle_min_radian(), robot.lidar.angle_max_radian() - 1E-10,
                                   robot.lidar.angle_increment_radian())
                clusters = detector.cluster_range_data(robot.lidar.data.last_range_data, angles)
                detections, _ = detector.detect(clusters, [0, 0])
                robot.lidar_clusters = clusters
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
            tracks_idx = [tr.id for tr in robot.tracks if not tr.coasted] + [0]
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
        if SYNTHESIS_ENABLED:
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
            robot.blind_spot_projector.update_pdf(smooth=True)
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

    # _scenario = RoundTrip()
    # _scenario.setup('powerlaw', flow_2d=True)

    _scenario = HermesScenario()
    _scenario.setup_with_config_file(os.path.abspath(os.path.join(__file__, "../../../..",
                                                     "config/followbot_sim/real_scenario_config.yaml")),
                                     biped_mode=BIPED_MODE)

    # =======================================
    if VIDEO_ENABLED:
        video_player = DatasetVideoPlayer(_scenario.video_files)
        video_player.set_frame_id(_scenario.world.original_frame_id)
    print(type(_scenario).__name__)
    for exec_frame, robot in exec_scenario(_scenario):
        print(exec_frame)

        if VIDEO_ENABLED:
            # if exec_frame != 0:
            #     video_player.set_frame_id(exec_frame)
            im = video_player.get_frame().__next__()
            if im is not None:
                cv2.imshow("im", im)
            else:
                print("problem with video")
            cv2.waitKey(2)
