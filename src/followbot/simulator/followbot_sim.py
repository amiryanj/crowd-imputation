import time
from scipy.spatial.transform import Rotation
from followbot.gui.display import *
from followbot.scenarios.corridor_crowd import CorridorCrowd
from followbot.scenarios.roundtrip_scenario import RoundTrip
from followbot.scenarios.static_crowd import StaticCrowd
from followbot.simulator.world import World
from followbot.util.basic_geometry import Line
from followbot.scenarios.real_scenario import RealScenario
from followbot.robot_functions.tracking import PedestrianDetection
from followbot.util.transform import Transform
# from crowd_synthesis.crowd_synthesis import CrowdSynthesizer

default_model = 'powerlaw'
agent_color = GREEN_COLOR


def init_world(world_dim, model=default_model, title='followBot'):
    world = World(N_ped, N_robots, model)
    display = Display(world, world_dim, (960, 960), title + '/' + model)
    # world.pref_speed = 1.5  # FIXME : set it for sim as well
    return world, display


def setup_linear_peds():
    # FIXME: init objects
    world, display = init_world(world_dim=[[0, 8], [-5, 5]], title='linear')
    line_objects = [Line([5, 0], [5, 8]), Line([6, 0], [6, 9])]
    for l_obj in line_objects:
        world.add_object(l_obj)

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
    world, display = init_world(world_dim=[[-outer_dim, outer_dim], [-outer_dim, outer_dim]], title='corridor')

    line_objects = [Line([0, 0], [10, 0]), Line([2, 7], [10, 7])]
    for l_obj in line_objects:
        world.add_object(l_obj)

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


def update_default(save=False):
    world, display = init_world(world_dim=[[-5, 5], [-5, 5]], title='circle')
    if not world.pause:
        world.step_crowd(0.02)

    toggle_pause = display.update()
    if toggle_pause: world.pause = not world.pause
    time.sleep(0.01)

    if not world.pause and save:
        display.screenshot('/home/cyrus/Videos/crowdsim/followbot/')


def run():
    # scenario = setup_corridor()
    # scenario = setup_circle()

    # scenario = RoundTrip()
    # scenario.setup('powerlaw', flow_2d=True)

    scenario = CorridorCrowd()
    scenario.setup()

    # Todo:
    #  capture the agents around the robot and extract the pattern
    #  1. Distance to (k)-nearest agent
    #  2. Velocity vector
    #  3. You can feed these data to a network with a pooling layer to classify different cases

    # scenario = StaticCrowd()
    # scenario.setup()

    # scenario = RealScenario()
    # scenario.setup()
    # crowd_syn = CrowdSynthesizer()
    # crowd_syn.extract_features(scenario.dataset)

    robot = scenario.world.robots[0]
    lidar = robot.lidar

    ped_detector = PedestrianDetection(robot.lidar.range_max, np.deg2rad(1 / robot.lidar.resolution))

    # ped_detector.

    while True:
        scenario.step()
            # scenario.step_crowd()
            # scenario.step_robot()
        cur_t = scenario.cur_t


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

        scenario.update_disply()

        ## FixMe:
        # n_configs = 2
        # if scenario.display.event == pygame.K_p:
        #     inds = np.where(scenario.ped_valid[scenario.cur_t])
        #     gt_pnts = scenario.ped_poss[scenario.cur_t, np.where(scenario.ped_valid[scenario.cur_t])]
        #     crowd_syn.analyze_and_plot(np.array(robot.detected_peds),
        #                                gt_pnts.reshape((-1, 2)), n_configs)


if __name__ == '__main__':
    run()



