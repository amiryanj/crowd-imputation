import sys
import random
from datetime import datetime
from scipy.spatial.transform import Rotation
from numpy.linalg import norm as norm
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import visdom
import logging


from sklearn.metrics import euclidean_distances
import scipy.optimize as op

from crowd_synthesis import CrowdSynthesizer
from followbot.scenarios.corridor_scenario import CorridorScenario
from followbot.crowdsim.pedestrian import Pedestrian
from followbot.gui.visualizer import *
from followbot.robot_functions.bivariate_gaussian import BivariateGaussianMixtureModel, BivariateGaussian, draw_bgmm
from followbot.robot_functions.dr_spaam_detector import DrSpaamDetector
from followbot.robot_functions.crowd_communities import CommunityHandler, CommunityDetector
from followbot.robot_functions.social_ties import SocialTiePDF
from followbot.robot_functions.human_detection import PedestrianDetection
from followbot.robot_functions.robot import MyRobot
from followbot.robot_functions.robot_replace_human import RobotReplaceHuman
from followbot.util.csv_logger import CsvLogger
from followbot.util.eval import Evaluation
from followbot.util.mapped_array import MappedArray
from followbot.util.video_player import DatasetVideoPlayer
from followbot.scenarios.hermes_scenario import HermesScenario
from followbot.scenarios.real_scenario import RealScenario
from followbot.util.transform import Transform
from followbot.util.read_lidar_data import read_lidar_data
# from followbot.crowd_synthesis.crowd_synthesis import CrowdSynthesizer

# from rich import print
from rich.console import Console

console = Console()
console.print("FollowBot Simulator", style="bold red")
matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk

# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.WARNING)

# Config
BIPED_MODE = True
LIDAR_ENABLED = True
RECORD_MAIN_SCENE = False
TRACKING_ENABLED = True
OVERRIDE_TRACKS = True  # Use ground-truth loc-vel of the agents
PROJECTION_ENABLED = True
READ_LIDAR_FROM_CARLA = False
NUM_HYPOTHESES = 1
MAP_RESOLUTION = 8  # per meter
# FRAME_RANGE = range(443, sys.maxsize)
FRAME_RANGE = range(0, sys.maxsize)
# ****************************

# Debug/Visualization Settings
VISUALIZER_ENABLED = False
DEBUG_VISDOM = False
VIDEO_ENABLED = False

LOG_SIM_TRAJS = False  # Deprecated?
LOG_PRED_OUTPUTS = True
LOG_PRED_EVALS = True
SAVE_PRED_PLOTS = False
SAVE_SIM_DATA_DIR = os.path.abspath(__file__ + "../../../../../data/prior-social-ties")
EVAL_RESULT_PATH = '/home/cyrus/Music/num-results/num_results-3.csv'
PRED_RESULTS_OUTPUT_DIR = '/home/cyrus/Music/followbot-outputs-2'
# ****************************

# Setup Random Seed
RAND_SEED = 4
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


# ****************************


# -----------------------------
def exec_scenario(scenario):
    if VISUALIZER_ENABLED:
        scenario.visualizer = Visualizer(scenario.world, scenario.world.world_dim,
                                         subViewRowCount=1 + NUM_HYPOTHESES,
                                         subViewColCount=1,
                                         caption=type(scenario).__name__)

    # Choose the robot type
    if isinstance(scenario, RealScenario):
        robot = RobotReplaceHuman(scenario.robot_poss, scenario.robot_vels, scenario.world,
                                  scenario.fps, NUM_HYPOTHESES, MAP_RESOLUTION)
        scenario.world.add_robot(robot)
        robot.init()
        dt = 1 / scenario.fps
        # robot = FollowerBot()  # deprecated ?
        # robot.set_leader_ped(scenario.world.crowds[0])

    else:  # use Std Robot
        robot = MyRobot(scenario.world, prefSpeed=1.2, sensorFps=scenario.fps,
                        numHypothesisWorlds=NUM_HYPOTHESES, mapResolution=MAP_RESOLUTION)
        scenario.world.add_robot(robot)
        dt = 1 / scenario.fps
        # Robot: starting in up side of the corridor, facing toward right end of the corridor
        robot.init([-25, 1.5])
        scenario.world.set_robot_goal(0, [1000, 0])
    # global robot
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
                                   np.degrees(robot.lidar.angle_increment_radian()), gpu=True)
    else:
        detector = PedestrianDetection(robot.lidar.range_max, np.deg2rad(1 / robot.lidar.resolution))

    # FixMe: Write robot observations to file
    # Todo: At the moment it only records the ground truth locations
    #  It can be the output of tracking module or even after using RWTH's DR-SPAAM tracker
    # =======================================
    # output_dir = "/home/cyrus/workspace2/ros-catkin/src/followbot/src/followbot/temp/robot_obsvs"
    # output_filename = os.path.join(output_dir, type(scenario).__name__ + ".txt")
    # output_file = open(output_filename, 'a+')
    # =======================================

    frame_id = -1
    # Read social-ties from file
    robot.occlusion_predictor = SocialTiePDF(radial_resolution=MAP_RESOLUTION)
    try:
        robot.occlusion_predictor.load_pdf(os.path.join(SAVE_SIM_DATA_DIR, scenario.title + '.npz'))
        robot.occlusion_predictor.smooth_pdf()
        if VISUALIZER_ENABLED:
            robot.occlusion_predictor.plot(scenario.title)
    except ValueError:
        print("WARNING: Could not read social-tie files ...")
    community_detector = CommunityDetector(scenario_fps=scenario.fps)
    community_handler = CommunityHandler()

    # Evaluator
    evaluator = Evaluation(show_heatmap=SAVE_PRED_PLOTS)
    eval_meshgrid = robot.blind_spot_map.meshgrid()

    if LOG_PRED_EVALS:
        # logging.basicConfig(filename=os.path.join(RESULT_DIR, 'num_results.csv'), filemode='a',
        #                     format='%(asctime)s, %(message)s', level=logging.WARNING)
        # eval_logger = logging.getLogger('ResultLogger')
        eval_logger = CsvLogger(filename=EVAL_RESULT_PATH)
        eval_logger.log(["New session: " + scenario.title, scenario.robot_replacement_id])
        eval_logger.log(['time, robot_id, frame_id, hh, robot_density, mse, bce, mse_bl, bce_bl'],
                        add_timestamp=False)

    if DEBUG_VISDOM:
        viz_dbg = visdom.Visdom()
        viz_win = viz_dbg.scatter(X=np.zeros((1, 2)), opts=dict(markersize=10,
                                                                markercolor=np.array([0, 0, 200]).reshape(-1, 3)))

    # ****************** Program Loop *******************
    while True:
        # run the scenario (+ world) for 1 step
        scenario_ret = scenario.step(dt, LIDAR_ENABLED, save=RECORD_MAIN_SCENE)
        if not scenario_ret:
            break
        if scenario.world.pause:
            continue
        frame_id = scenario.world.original_frame_id
        if frame_id not in FRAME_RANGE:
            continue

        # Write detected pedestrians to the output file
        # =======================================
        if LOG_SIM_TRAJS:
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
            if BIPED_MODE:  # Dr-SPAAM (RWTH)
                detections, dets_cls = detector.detect(robot.lidar.data.last_range_data)
            else:
                ray_angles = np.arange(robot.lidar.angle_min_radian(), robot.lidar.angle_max_radian() - 1E-10,
                                       robot.lidar.angle_increment_radian())
                clusters = detector.cluster_range_data(robot.lidar.data.last_range_data, ray_angles)
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
            tracks_idx = [0] + [tr.id for tr in robot.tracks if not tr.coasted]
            tracks_loc = np.array([robot.pos] + [tr.kf.x[:2] for tr in robot.tracks if not tr.coasted])
            tracks_vel = np.array([robot.vel] + [tr.kf.x[2:4] for tr in robot.tracks if not tr.coasted])
            n_tracked_agents = len(tracks_loc)
            # ====================================

            if scenario.world.frame_id > len(scenario.ped_poss) - 2:
                print("****** End of current robot_id ******")
                break

            # Ground Truth location/velocity of ALL agents at (t), NOT include robot
            gt_peds = [(p.pos, p.vel) for p in scenario.world.crowds if p.pos[0] != -100]
            gt_locs, gt_vels = list(map(itemgetter(0), gt_peds)), list(map(itemgetter(1), gt_peds))
            # gt_locs = [pos[i] for i in range(len(scenario.ped_poss[scenario.world.frame_id]))]
            # gt_vels = scenario.ped_vels[scenario.world.frame_id]
            # scenario.ped_valid

            # fixme: override detections with ground-truth locations?
            # ===================================
            if OVERRIDE_TRACKS:
                # gt_ids =  scenario.ped_valid
                gt_locs_and_robot = np.array([robot.pos] + gt_locs)
                gt_vels_and_robot = np.array([robot.vel] + gt_vels)
                D = euclidean_distances(tracks_loc, gt_locs_and_robot)
                row_ids, col_ids = op.linear_sum_assignment(D)
                # overriding takes place at this point:
                tracks_loc = np.array([gt_locs_and_robot[i] for i in col_ids])
                tracks_vel = np.array([gt_vels_and_robot[i] for i in col_ids])
                for ii in range(len(tracks_idx)):
                    for tr in robot.tracks:  # a little stupid but it's Ok.
                        if tr.id == tracks_idx[ii] and not tr.coasted:  # and tr.id != 0:
                            tr.kf.x[:2] = tracks_loc[ii]
                            tr.kf.x[2:4] = tracks_vel[ii]
                            break
            # ===================================

        # ***************************************************
        if LIDAR_ENABLED:
            # calc Blind Spots
            # ================================
            robot.blind_spot_map.fill(1)  # every pixel is occluded if not otherwise inferred!
            rays = robot.lidar.data.last_rotated_rays
            for ii in range(len(rays)):
                ray_i = rays[ii]
                scan_i = robot.lidar.data.last_points[ii]
                white_line = [robot.pos, scan_i]
                line_len = np.linalg.norm(white_line[1] - white_line[0])
                Zs = np.arange(0, line_len / robot.lidar.range_max, 0.01)  # step: fixme
                Pxy = Zs.reshape(-1, 1) * ray_i[1].reshape(1, 2) + (1 - Zs.reshape(-1, 1)) * ray_i[0].reshape(1, 2)
                for pxy in Pxy:
                    robot.blind_spot_map.set(pxy, 0)
            # =================================

        # ***************************************************
        if PROJECTION_ENABLED:
            # Classify Ties
            # =================================
            # strong_ties_t, absent_ties_t, agents_flow_class = \
            #     robot.blind_spot_projector.classify_ties(tracks_loc, tracks_vel)
            #
            # fixme: in this version, the new ties will not be used to update the tie distributions
            # robot.blind_spot_projector.add_strong_ties(strong_ties_t)
            # robot.blind_spot_projector.add_absent_ties(absent_ties_t)
            # robot.blind_spot_projector.update_pdf()

            # agents_flow_class = FlowClassifier().classify(tracks_loc, tracks_vel)   # => fixme: @deprecated
            strong_ties, absent_ties, strong_ties_idx, absent_ties_idx, communities = \
                community_detector.cluster_communities(tracks_idx, tracks_loc, tracks_vel)

            community_handler.ped_ids = tracks_idx
            community_handler.communities = communities
            community_handler.calc_velocities(tracks_vel)

            # ================================
            xx, yy = robot.crowd_territory_map.meshgrid()
            # the map will hold id of corr community @ each pixel  / using `multiple Gaussian`
            robot.crowd_territory_map.data, velocity_map_data = \
                community_handler.find_territories(tracks_loc, tracks_vel, xx, yy)
            velocity_map = MappedArray(robot.crowd_territory_map.min_x, robot.crowd_territory_map.max_x,
                                       robot.crowd_territory_map.min_y, robot.crowd_territory_map.max_y,
                                       robot.crowd_territory_map.resolution, n_channels=2, dtype=np.float)
            velocity_map.data = velocity_map_data.reshape(xx.shape[::-1] + (2,))
            # =================================

            # Crowd Projection: Using Social Ties
            # =================================
            # robot.blind_spot_projector.add_frame(tracks_loc, tracks_vel, agents_flow_class, dt)
            # robot.blind_spot_projector.update_pdf()
            # robot.blind_spot_projector.plot(scenario.title)

            for hh, robot_hypo in enumerate(robot.hypothesis_worlds):
                if scenario.world.time >= 0.1 and (frame_id % 5 == 0):
                    # The returned list, only include the new projected poitns
                    robot_hypo.crowds = robot.occlusion_predictor.project(tracks_loc, tracks_vel,
                                                                          walkable_map=robot_hypo.walkable_map,
                                                                          blind_spot_map=robot.blind_spot_map,
                                                                          crowd_territory_map=robot.crowd_territory_map,
                                                                          community_velocity_map=velocity_map,
                                                                          verbose=VISUALIZER_ENABLED)

                    # all the projected (predicted) locs + tracked locs (including the robot)
                    robot_hypo_locs = [pi.pos for pi in robot_hypo.crowds] + list(tracks_loc)  # which include robot

                    mse, bce, _, _ = evaluator.calc_error(eval_meshgrid,
                                                          [robot.pos] + gt_locs,  # which does not include robot
                                                          robot_hypo_locs,
                                                          plot_text="%05d-ours" % frame_id)  # for visualization

                    # PCF Baseline
                    # =================================
                    pcf_locs = pcf_crowd_rec.pcf_synthesize(list(tracks_loc))
                    pcf_hypo_locs = list(pcf_locs) + list(tracks_loc)
                    # dart_thrower_locs = pcf_crowd_rec.dart_thrower.create_samples(tracks_loc)
                    mse_bl, bce_bl, _, _ = evaluator.calc_error(eval_meshgrid,
                                                                [robot.pos] + gt_locs,  # which does NOT include robot
                                                                pcf_hypo_locs,
                                                                plot_text="%05d-pcf" % frame_id)
                    # =================================


                    density = evaluator.local_density([robot.pos] +
                                                      [p.pos for p in scenario.world.crowds if p.pos[0] != -100]
                                                      )
                    robot_density = density[0]

                    # print(scenario.robot_replacement_id, hh, robot_density, mse, bce)

                    if LOG_PRED_EVALS:
                        print('writing to eval_logger...')
                        eval_logger.log([scenario.robot_replacement_id, frame_id, hh,
                                         robot_density,
                                         mse, bce,
                                         mse_bl, bce_bl])

                    if DEBUG_VISDOM:
                        viz_dbg.scatter(np.array([[robot_density, mse]]),
                                        opts=dict(markersize=10, markercolor=np.array([200, 0, 0]).reshape(-1, 3)),
                                        name='MSE', update='append', win=viz_win)

                    # if hh == 0:
                    #     break
                    # #evolve (ToDo: use TrajPredictor)
                    # else:
                    #     for ped in robot_hypo.crowds:
                    #         if ped.synthetic:
                    #             ped.pos = np.array(ped.pos) + np.array(ped.vel) * dt
                    #             if robot.blind_spot_map.get(ped.pos) < 0.1:
                    #                 del ped
                    # print("delete the synthetic agent")

                    if LOG_PRED_OUTPUTS and PROJECTION_ENABLED:
                        print('writing pred output to npz file...')
                        np.savez(os.path.join(PRED_RESULTS_OUTPUT_DIR,
                                              '%s--%d-%d-%d.npz' % (scenario.title,
                                                                    scenario.robot_replacement_id,
                                                                    scenario.world.original_frame_id,
                                                                    hh)),
                                 robot_loc=robot.pos,
                                 detections=tracks_loc,
                                 gt_locs=gt_locs,
                                 preds_ours=[agent.pos for agent in robot_hypo.crowds],
                                 preds_pcf=pcf_locs,
                                 x_range=eval_meshgrid[0][0],
                                 y_range=eval_meshgrid[1][:, 0])

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

    temp_scenario = HermesScenario()
    temp_scenario.setup_with_config_file(os.path.abspath(os.path.join(__file__, "../../../..",
                                                                      "config/followbot_sim/real_scenario_config.yaml"
                                                                      )))
    agent_ids = temp_scenario.dataset.get_agent_ids()
    test_ids = range(int(0.7 * len(agent_ids)), len(agent_ids))

    pcf_crowd_rec = CrowdSynthesizer()
    pcf_crowd_rec.extract_features(temp_scenario.dataset)

    for a_id in test_ids:
        robot_id = agent_ids[a_id]
        _scenario = HermesScenario()
        _scenario.setup_with_config_file(os.path.abspath(os.path.join(__file__, "../../../..",
                                                                      "config/followbot_sim/real_scenario_config.yaml")),
                                         title="HERMES-bo-360-075-075", biped_mode=BIPED_MODE,
                                         robot_id=robot_id)
        _scenario.title = _scenario.dataset.title

        if READ_LIDAR_FROM_CARLA:  # FixMe: make sure the robotId has been the same
            lidar_data = read_lidar_data()
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        # =======================================
        if VIDEO_ENABLED:
            video_player = DatasetVideoPlayer(_scenario.video_files)
            video_player.set_frame_id(_scenario.world.original_frame_id)
        print(type(_scenario).__name__)
        for exec_frame_id, robot in exec_scenario(_scenario):
            print("agent [%d/%d] frame= [%d(%d)/%d]" % (a_id, len(agent_ids),
                                                        _scenario.world.frame_id, exec_frame_id, len(_scenario.ped_poss)))

            if VIDEO_ENABLED:
                # if exec_frame != 0:
                #     video_player.set_frame_id(exec_frame)
                im = video_player.get_frame().__next__()
                if im is not None:
                    cv2.imshow("im", im)
                else:
                    print("problem with video")
                cv2.waitKey(2)

            if READ_LIDAR_FROM_CARLA and exec_frame_id in lidar_data:
                lidar_data[exec_frame_id][:, 1] *= -1
                lidar_data[exec_frame_id][:, 2] = 1
                robot_rot = Rotation.from_euler('z', robot.orien, degrees=False).as_matrix()
                robot_tf = np.hstack([robot_rot[:2, :2], robot.pos.reshape(2, 1)])  # 2 x 3
                lidar_data[exec_frame_id][:, :2] = np.matmul(lidar_data[exec_frame_id][:, :3], robot_tf.T)

                plt.cla()
                ax.set_xlim([-7, 7])
                ax.set_ylim([-0, 4])
                # ax.set_zlim([0, 5])
                ax.scatter(lidar_data[exec_frame_id][:, 0],  # x
                           lidar_data[exec_frame_id][:, 1],  # y
                           # scan[data_range, 2],  # z
                           c=lidar_data[exec_frame_id][:, 3],  # reflectance
                           s=2, cmap='viridis')
                ax.plot(robot.pos[0], robot.pos[1], 'ro')
                ax.set_title('Lidar scan %s' % exec_frame_id)
                plt.grid()
                plt.pause(0.01)
