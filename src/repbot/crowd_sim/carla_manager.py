# Author: Javad Amirian
# Email: amiryan.j@gmail.com

# coding=utf-8
"""Given the trajectory file, reconstruct person walking."""

import argparse
import glob
import pygame
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])

import carla
import numpy as np
# import skvideo.io

from repbot.util.carla_utils import setup_walker_bps
from repbot.util.carla_utils import setup_static
from repbot.util.carla_utils import get_scene
from repbot.util.carla_utils import get_controls_from_traj_data
from repbot.util.carla_settings import static_scenes
from toolkit.loaders.loader_hermes import load_bottleneck
from repbot.util.carla_utils import Camera, get_bp



def xx(data):
    print("HERERS")


class CarlaManager:
    def __init__(self, robot_id=-1):
        self.lidar_sensor = None
        self.camera_sensor = None
        self.fpv_camera = None  # for recording the images
        self.client = None
        self.world = None
        self.robot_id = robot_id
        self.walker_bps = []
        self.ped_controls = []
        self.global_actor_list = []
        self.excepts = []
        self.cur_peds = {}
        self.current_peds = {}  # person_id -> actor
        self.actorid2info = {}
        self.static_scene = {}
        self.total_moment_frame_num = 0
        self.fps = -1
        self.frame_id = -1
        self.synchronous_mode = True
        self.save_path = ""

        self.lidar_height = 0.4

    def save_lidar_data(self, data):
        lidar_output_file = os.path.join(self.save_path, "lidar", "%08d.dat" % self.frame_id)
        print("printing to: ", lidar_output_file)
        data.save_to_disk(lidar_output_file)

    def setup(self, traj_dataset, fps, start_frame_idx, end_frame_idx):
        self.ped_controls, self.total_moment_frame_num = get_controls(traj_dataset, start_frame_idx, end_frame_idx, fps,
                                                                      no_offset=True)
        print("Control data prepared.")

        self.world = None
        self.fps = fps

        try:
            self.client = carla.Client(args.host, args.port)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.walker_bps = setup_walker_bps(self.world)

            # seems some puddle on the ground makes the scene look perceptually more real.
            self.static_scene['weather'] = {
                "cloudiness": 20.0,
                "precipitation": 0.0,
                "sun_altitude_angle": 65.0,
                "precipitation_deposits": 60.0,
                "wind_intensity": 80.0,
                "sun_azimuth_angle": 20.0}

            # 1. set up the static env
            setup_static(self.world, self.client, self.static_scene, self.global_actor_list)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1.0 / self.fps
            settings.synchronous_mode = self.synchronous_mode
            self.world.apply_settings(settings)

            # this is the server side frame Id we start with
            baseline_frame_id = self.world.tick()
            self.client_clock = pygame.time.Clock()
        except:
            if self.world is None:
                raise Exception("Make sure Carla is running!")
            else:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            pygame.quit()

    def step(self, show_traj):
        try:
            for moment_frame_count in range(self.total_moment_frame_num):
                # grab the control data of this frame if any
                batch_cmds, _ = self.run_sim_for_one_frame(moment_frame_count, show_traj=show_traj,
                                                           verbose=True, no_collision_detector=True)
                if batch_cmds:
                    response = self.client.apply_batch_sync(batch_cmds)

                # block if faster than fps
                self.client_clock.tick_busy_loop(self.fps)
                server_frame_id = self.world.tick()
                print("server_frame_id: ", server_frame_id)
                print("moment_frame_count: ", moment_frame_count)
                yield None

        finally:
            if self.world is None:
                raise Exception("Carla is not running!")
            else:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                for actor in self.global_actor_list:
                    if actor.type_id.startswith("sensor"):
                        actor.stop()
                # finished, clean actors
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in self.global_actor_list])

            pygame.quit()

    def run_sim_for_one_frame(self, frame_id, show_traj=False, verbose=False,
                              exit_if_spawn_fail=False, no_collision_detector=False,
                              pid2actor={}):
        """Given the controls and the current frame_id, exe the simulation. Return
         the batch command to execute, return None if the any spawning failed
        """
        stats = {}
        batch_cmds = []

        # if ped_controls.has_key(frame_id):
        if frame_id in self.ped_controls:
            self.frame_id = frame_id
            this_control_data = self.ped_controls[frame_id]
            for person_id, _, xyz, direction_vector, speed, time_elasped, is_static in this_control_data:
                if person_id in self.excepts:
                    continue

                if person_id == self.robot_id:
                    robot_tf = carla.Transform(location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))

                    # if self.robot_id not in cur_peds:
                    if self.lidar_sensor is None:
                        # kid_bp = world.get_blueprint_library().find("walker.pedestrian.0011")
                        # kid_walker = world.try_spawn_actor(kid_bp, carla.Transform(
                        #     location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]),
                        #     rotation=carla.Rotation(yaw=np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])),
                        #                             pitch=0, roll=0)))
                        # cur_peds[person_id] = kid_walker
                        # global_actor_list.append(kid_walker)
                        # actorid2info[kid_walker.id] = ("Person", person_id)
                        # pid2actor[person_id] = kid_walker

                        lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
                        lidar_bp.set_attribute('range', '8.0')
                        lidar_bp.set_attribute('channels', '3')
                        lidar_bp.set_attribute('upper_fov', '0.1')
                        lidar_bp.set_attribute('lower_fov', '-0.1')
                        lidar_bp.set_attribute('dropoff_general_rate', '0.1')
                        lidar_bp.set_attribute('sensor_tick', '0.1')
                        self.lidar_sensor = self.world.try_spawn_actor(lidar_bp, carla.Transform(
                                                                            robot_tf.location + carla.Location(z=.4))
                                                                       # ,attach_to=kid_walker
                                                                       )
                        if self.save_path:
                            self.lidar_sensor.listen(lambda data: self.save_lidar_data(data))
                            # self.lidar_sensor.listen(lambda data: xx(data))

                        # configure the rgb camera
                        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
                        camera_bp.set_attribute("image_size_x", "720")
                        camera_bp.set_attribute("image_size_y", "560")
                        camera_bp.set_attribute("fov", "80")
                        # Set the time in seconds between sensor captures
                        camera_bp.set_attribute("sensor_tick", "0.1")
                        camera_bp.set_attribute("enable_postprocess_effects", "true")
                        # no motion blur
                        camera_bp.set_attribute("motion_blur_intensity", "0.0")
                        # 2.2 is default, 1.5 is the carla default spectator gamma (darker)
                        camera_bp.set_attribute("gamma", "1.9")

                        self.camera_sensor = self.world.try_spawn_actor(camera_bp, carla.Transform())
                        self.fpv_camera = Camera(self.camera_sensor, width=720, height=560,
                                                 fov=80, save_path=self.save_path,
                                                 camera_type="rgb", recording=True)

                        print("created robot as ped=", self.robot_id)
                        # continue
                    # ---- last location reached ----
                    elif direction_vector is None:
                        self.lidar_sensor.stop()
                        self.camera_sensor.stop()

                    else:  # Update sensor locations/rotation
                        sensor_loc = robot_tf.location + carla.Location(z=self.lidar_height - xyz[2])
                        sensor_rot = carla.Rotation(yaw=np.rad2deg(np.arctan2(direction_vector[1],direction_vector[0])))
                        spectator = self.world.get_spectator()
                        spectator.set_transform(carla.Transform(sensor_loc, sensor_rot))
                        self.fpv_camera.set_frame_id(self.frame_id)
                        self.lidar_sensor.set_transform(carla.Transform(sensor_loc, sensor_rot))
                        self.camera_sensor.set_transform(carla.Transform(sensor_loc, sensor_rot))
                    continue

                # ------ last location reached, so delete this guy ------
                if direction_vector is None:
                    if person_id in self.cur_peds:
                        batch_cmds.append(carla.command.DestroyActor(self.cur_peds[person_id]))
                        del self.cur_peds[person_id]

                else:
                    walker_control = carla.WalkerControl(direction=carla.Vector3D(x=direction_vector[0],
                                                                                  y=direction_vector[1],
                                                                                  z=direction_vector[2]),
                                                         speed=speed, jump=False)
                    # carla.command.applyTransform()

                    # new person, need to spawn
                    # if not cur_peds.has_key(person_id):
                    if person_id not in self.cur_peds:
                        walker_bp = get_bp(self.walker_bps)
                        new_walker = self.world.try_spawn_actor(walker_bp, carla.Transform(
                            location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]),
                            rotation=carla.Rotation(yaw=np.rad2deg(np.arctan2(direction_vector[1],direction_vector[0])),
                                                    pitch=0, roll=0)))
                        if new_walker is None:
                            if verbose:
                                print("%s: person %s failed to spawn." % (frame_id, person_id))
                            if exit_if_spawn_fail:
                                return None, stats
                            else:
                                continue
                        if verbose:
                            print("%s: Spawned person id %s." % (frame_id, person_id))
                        new_walker.set_simulate_physics(True)
                        self.cur_peds[person_id] = new_walker
                        self.global_actor_list.append(new_walker)

                        # add a collision sensor
                        self.actorid2info[new_walker.id] = ("Person", person_id)
                        pid2actor[person_id] = new_walker

                        # if show_traj:
                        # show the track Id
                        # world.debug.draw_string(carla.Location(
                        #     x=xyz[0], y=xyz[1], z=0), "# %s" % person_id,
                        #     draw_shadow=False,
                        #     color=carla.Color(r=255, g=0, b=0),
                        #     life_time=30.0)

                    this_walker_actor = self.cur_peds[person_id]

                    if show_traj:
                        delay = 1.5  # delay before removing traj
                        p1 = carla.Location(x=xyz[0], y=xyz[1], z=0)
                        next_xyz = [xyz[i] + direction_vector[i] * speed * time_elasped for i in range(3)]
                        p2 = carla.Location(x=next_xyz[0], y=next_xyz[1], z=0)
                        self.world.debug.draw_arrow(p1, p2, thickness=0.1, arrow_size=0.001,
                                                    color=carla.Color(b=200, a=100),
                                                    life_time=time_elasped + delay)

                    if is_static:
                        # stop the walker
                        batch_cmds.append(carla.command.ApplyWalkerControl(
                            this_walker_actor, carla.WalkerControl()))
                        continue
                    batch_cmds.append(carla.command.ApplyWalkerControl(
                        this_walker_actor, walker_control))

                    sim_xyz = this_walker_actor.get_location()
                    # print("disp err (sim2real)= %3f" % np.linalg.norm(np.array(xyz[:2]) - np.array([sim_xyz.x, sim_xyz.y])))
                    # compensate the sim error
                    this_walker_actor.set_location(carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))
                    # batch_cmds.append(carla.command.ApplyTransform(this_walker_actor,
                    #                                                carla.Transform(location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))))

        return batch_cmds, stats


def get_controls(traj_dataset, start_frame, end_frame, fps_, interpolate=False, z_to=None, no_offset=False):
    """Gather the trajectories and convert to control data."""
    traj_dataset.data["pos_z"] = 0.93
    data = traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y", "pos_z"]].to_numpy()

    control_data, total_frame_num = get_controls_from_traj_data(
        data, start_frame, end_frame, fps_, interpolate=interpolate,
        z_to=z_to, no_offset=no_offset)

    return control_data, total_frame_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file")
    parser.add_argument("start_frame_idx", type=int,
                        help="inclusive")
    parser.add_argument("end_frame_idx", type=int,
                        help="inclusive")
    parser.add_argument("--robot_id", type=int, default=60,  # robot_id_ = -1  # to disable robot
                        help="Id of pedestrian that carries the LiDAR sensor (considered to be robot)")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)

    parser.add_argument("--show_traj", action="store_true")

    parser.add_argument("--lidar_z", type=float, default=0.4,
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
        static_scene = {  # default scene
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

    fps_ = static_scene["fps"]

    # process the traj first.
    # gather all trajectory control within the frame num
    # frame_id -> list of [person_id, xyz, direction vector, speed]

    traj_dataset = load_bottleneck(args.traj_file, use_kalman=False)
    traj_dataset.apply_transformation(np.array([[-1, 0, -45],
                                                [0, 1, -98.],
                                                [0, 0, 1]]), inplace=True)

    # traj_dataset.data["pos_x"] += 30  # x offset
    # traj_dataset.data["pos_y"] += 5   # y offset

    carla_manager = CarlaManager(args.robot_id)
    carla_manager.static_scene = static_scene
    carla_manager.setup(traj_dataset, fps_, args.start_frame_idx, args.end_frame_idx)
    carla_manager.save_path = "/home/cyrus/Videos/carla"
    # main loop
    for _ in carla_manager.step(args.show_traj):
        pass
