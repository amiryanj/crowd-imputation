# coding=utf-8
"""Given the carla trajectory file, reconstruct person walking."""

import argparse
import glob
import pygame
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla
# import skvideo.io
import numpy as np


from followbot.util.carla_utils import setup_walker_bps
from followbot.util.carla_utils import setup_static
from followbot.util.carla_utils import get_scene
from followbot.util.carla_utils import get_controls_from_traj_data
from followbot.util.carla_utils import run_sim_for_one_frame
from followbot.util.carla_multiverse_settings import static_scenes
from toolkit.loaders.loader_hermes import load_bottleneck

default_scene = {
        "fps": 25.0,
        "weather": {
            "cloudiness": 80.0,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "sun_altitude_angle": 145.0,
            "sun_azimuth_angle": 0.0,
            "wind_intensity": 80.0,
        },
        "static_cars": [],
        "map": "Town03_ethucy",
    }


def get_controls(traj_file, start_frame, end_frame, fps_, interpolate=False, z_to=None, no_offset=False):
    """Gather the trajectories and convert to control data."""
    traj_dataset = load_bottleneck(traj_file)
    traj_dataset.apply_transformation(np.array([[1, 0, 30],
                                                [0, 1, 5.],
                                                [0, 0, 1]]), inplace=True)

    # traj_dataset.data["pos_x"] += 30  # x offset
    # traj_dataset.data["pos_y"] += 5   # y offset
    traj_dataset.data["pos_z"] = 0.93
    data = traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y", "pos_z"]].to_numpy()

    control_data, total_frame_num = get_controls_from_traj_data(
        data, start_frame, end_frame, fps_, interpolate=interpolate,
        z_to=z_to, no_offset=no_offset)

    return control_data, total_frame_num


def step_dataset(traj_file, start_frame_idx, end_frame_idx, fps, robot_id):
    ped_controls, total_moment_frame_num = get_controls(traj_file, start_frame_idx, end_frame_idx, fps)
    print("Control data prepared.")

    actorid2info = {}  # carla actorId to the personId or vehicle Id
    world = None

    try:
        global_actor_list = []

        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        world = client.get_world()

        walker_bps = setup_walker_bps(world)

        # 1. set up the static env
        setup_static(world, client, static_scene, global_actor_list)

        settings = world.get_settings()
        settings.fixed_delta_seconds = 1.0 / fps
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # seems some puddle on the ground makes the scene look perceptually more real.
        realism_weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=65.0,
            precipitation_deposits=60.0,
            wind_intensity=80.0,
            sun_azimuth_angle=20.0)

        # this is the server side frame Id we start with
        baseline_frame_id = world.tick()

        client_clock = pygame.time.Clock()

        moment_frame_count = 0
        current_peds = {}  # person_id -> actor
        max_yaw_change = 45  # no sudden yaw change
        for moment_frame_count in range(total_moment_frame_num):
            # grab the control data of this frame if any
            batch_cmds, _ = run_sim_for_one_frame(
                moment_frame_count, ped_controls,
                current_peds,
                walker_bps,
                world,
                global_actor_list, actorid2info,
                robot_id=robot_id,
                show_traj=args.show_traj, verbose=True, no_collision_detector=True,
                max_yaw_change=max_yaw_change)

            if batch_cmds:
                response = client.apply_batch_sync(batch_cmds)

            # block if faster than fps
            client_clock.tick_busy_loop(fps)
            server_frame_id = world.tick()
            print("server_frame_id: ", server_frame_id)
            print("moment_frame_count: ", moment_frame_count)
            yield None

    finally:
        if world is None:
            raise Exception("Carla is not running!")
        else:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            for actor in global_actor_list:
                if actor.type_id.startswith("sensor"):
                    actor.stop()
            # finished, clean actors
            client.apply_batch(
                [carla.command.DestroyActor(x) for x in global_actor_list])

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file")
    parser.add_argument("start_frame_idx", type=int,
                        help="inclusive")
    parser.add_argument("end_frame_idx", type=int,
                        help="inclusive")
    parser.add_argument("--robot_id", type=int, default=15,  # robot_id_ = -1  # to disable robot
                        help="Id of pedestrian that carries the LiDAR sensor (considered to be robot)")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)

    parser.add_argument("--show_traj", action="store_true")

    parser.add_argument("--lidar_z", type=float, default=0.0,
                        help="the height of the lidar sensor from ground")

    args = parser.parse_args()


    filename = os.path.splitext(os.path.basename(args.traj_file))[0]
    if filename.startswith("VIRAT"):  # ActEV dataset
        scene = get_scene(filename)
        assert scene in static_scenes
        static_scene = static_scenes[scene]
    elif filename in static_scenes:
        static_scene = static_scenes[filename]
    else:
        static_scene = default_scene

    fps_ = static_scene["fps"]

    # process the traj first.
    # gather all trajectory control within the frame num
    # frame_id -> list of [person_id, xyz, direction vector, speed]

    # main loop
    for _ in step_dataset(args.traj_file, args.start_frame_idx, args.end_frame_idx, fps_, args.robot_id):
        pass

