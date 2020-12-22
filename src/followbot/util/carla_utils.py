# coding=utf-8
"""Utils for constructing moments."""

import cv2
import math
import os
import sys
import glob
import operator
import pygame

import numpy as np
import carla

# script_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(glob.glob("%s/carla*.egg" % script_path)[0])


def get_sensor_data(data):
    output_file = "/home/cyrus/Music/lidar-data/%05d.dat" % data.frame
    print("printing to: ", output_file)
    data.save_to_disk(output_file)


def make_moment_id(scene, moment_idx, x_agent_pid, dest_idx, annotator_id):
    return "%s_%d_%d_%d_%s" % (scene, moment_idx, x_agent_pid, dest_idx,
                               annotator_id)


def setup_walker_bps(world):
    # walker model that we use
    # walker_indexes = [1, 2, 3, 4, 5, 6, 7, 8]
    # 9-14 are kids
    # walker_indexes = [9, 10, 11, 12, 13, 14]
    walker_indexes = [2]
    # the last item is the current cycled index of the bps
    walker_bps = [["walker.pedestrian.%04d" % o for o in walker_indexes], 0]
    walker_bps_list = [
        world.get_blueprint_library().find(one) for one in walker_bps[0]]
    walker_bps = [walker_bps_list, 0]
    return walker_bps


def setup_vehicle_bps(world):
    vehicle_bps = [
        [
            "vehicle.audi.a2",
            "vehicle.audi.etron",
            "vehicle.bmw.grandtourer",
            "vehicle.chevrolet.impala",
            "vehicle.citroen.c3",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.lincoln.mkz2017",
            "vehicle.nissan.micra",
            "vehicle.nissan.patrol",
        ], 0]
    vehicle_bps_list = [
        world.get_blueprint_library().find(one) for one in vehicle_bps[0]]
    vehicle_bps = [vehicle_bps_list, 0]
    return vehicle_bps


def get_bp(bps_):
    """Cycle through all available models."""
    bp_list, cur_index = bps_
    new_index = cur_index + 1
    if new_index >= len(bp_list):
        new_index = 0
    bps_[1] = new_index
    return bp_list[cur_index]


def reset_x_agent_key(moment_data):
    # reset all x_agents dict's key to int, since stupid json
    for i in range(len(moment_data)):
        this_moment_data = moment_data[i]
        new_x_agents = {}
        for key in this_moment_data["x_agents"]:
            new_key = int(float(key))
            new_x_agents[new_key] = this_moment_data["x_agents"][key]
        moment_data[i]["x_agents"] = new_x_agents


def interpolate_data_between(p1, p2):
    # p1, p2 is [frame_id, person_id, x, y, ..]
    data_points = []
    num_frames = int(p2[0] - p1[0])
    for i in range(num_frames - 1):
        new_data_point = [p1[0] + i + 1, p1[1]]
        for coor1, coor2 in zip(p1[2:], p2[2:]):
            inc = (coor2 - coor1) / num_frames
            this_coor = coor1 + inc * (i + 1)
            new_data_point.append(this_coor)
        data_points.append(new_data_point)
    return data_points


def interpolate_controls(controls, fps_):
    """Given low frame rate controls, interpolate."""
    # first, get the traj data
    # [frame_id, person_id, x, y, z]
    data = []
    for frame_id in controls:
        for pid, _, (x, y, z), _, _, _, is_stationary in controls[frame_id]:
            # json BS
            int_frame_id = int(float(frame_id))
            # need the is_stationary to keep things the same
            data.append([int_frame_id, int(pid), x, y, z, is_stationary])

    if len(data) == 0:
        return {}
    data.sort(key=operator.itemgetter(0))
    data = np.array(data, dtype="float64")

    person_ids = np.unique(data[:, 1]).tolist()
    # the frame_id in control data should be offset to the start of
    # the actual moment
    # frame_id always start from 0
    control_data = {}  # frame_id ->
    for person_id in person_ids:
        this_data = data[data[:, 1] == person_id, :]
        is_stationaries = this_data[:, -1]
        this_data = this_data[:, :-1]
        # here this_data should already be sorted by frame_id ASC
        if this_data.shape[0] <= 1:
            continue
        # interpolate the points in between
        # assuming constant velocity
        # don't interpolate if the second point is already stationary
        if is_stationaries[1] != 1.0:
            new_data = []
            new_stationaries = []
            for i in range(this_data.shape[0] - 1):
                j = i + 1
                # add the start point
                this_new = [this_data[i]]
                this_new += interpolate_data_between(this_data[i], this_data[j])
                new_data += this_new
                new_stationaries += [is_stationaries[i]] * len(this_new)
            new_data.append(this_data[-1])
            new_stationaries.append(is_stationaries[-1])
            this_data = np.array(new_data, dtype="float64")
            is_stationaries = np.array(new_stationaries, dtype="float64")

        for i in range(this_data.shape[0] - 1):
            frame_id = int(this_data[i, 0])
            j = i + 1
            is_stationary = is_stationaries[i]
            # direction vector
            direction_vector, speed, time_elasped = get_direction_and_speed(
                this_data[j], this_data[i], fps_)
            # if not control_data.has_key(frame_id):
            if frame_id not in control_data:
                control_data[frame_id] = []
            control_data[frame_id].append(
                [person_id, this_data[i, 0], this_data[i, 2:].tolist(),
                 direction_vector, speed,
                 time_elasped, is_stationary])

        last_frame_id = int(this_data[-1, 0])
        # if not control_data.has_key(last_frame_id):
        if last_frame_id not in control_data:
            control_data[last_frame_id] = []
        # signaling stop
        control_data[last_frame_id].append(
            [person_id, this_data[i, 0], this_data[-1, 2:].tolist(),
             None, None, None, None])

    # json bs
    new_control_data = {}
    for frame_id in control_data:
        string_frame_id = "%s" % frame_id
        new_control_data[string_frame_id] = control_data[frame_id]
    return new_control_data


def reset_bps(bps_):
    bps_[1] = 0


def get_controls(traj_file, start_frame, end_frame, fps_, interpolate=False,
                 z_to=None, no_offset=False):
    """Gather the trajectories and convert to control data."""
    data = [o.strip().split("\t") for o in open(traj_file).readlines()]
    data = np.array(data, dtype="float64")  # [frame_id, person_id, x, y, z]

    control_data, total_frame_num = get_controls_from_traj_data(
        data, start_frame, end_frame, fps_, interpolate=interpolate,
        z_to=z_to, no_offset=no_offset)

    return control_data, total_frame_num


def get_controls_from_traj_data(data, start_frame, end_frame, fps_,
                                interpolate=False, z_to=None, no_offset=False):
    if z_to is not None:
        # for car traj, set all z coordinates to 0
        data[:, -1] = z_to

    frame_ids = np.unique(data[:, 0]).tolist()
    frame_ids.sort()
    if start_frame == -1:
        target_frame_ids = frame_ids
    else:
        if start_frame not in frame_ids:
            return {}, 0
        start_idx = frame_ids.index(start_frame)
        end_idx = frame_ids.index(end_frame)
        target_frame_ids = frame_ids[start_idx:end_idx]
    total_frame_num = int(target_frame_ids[-1] - target_frame_ids[0])

    filtered_data = data[np.isin(data[:, 0], target_frame_ids), :]
    # compute the direction vector and speed at each timestep
    # per person
    person_ids = np.unique(filtered_data[:, 1]).tolist()
    # the frame_id in control data should be offset to the start of
    # the actual moment
    # frame_id always start from 0
    control_data = {}  # frame_id ->
    # compute the absolute change between points so we can identify when
    # the traj is stationary like a parked car
    traj_change_future_seconds = 2.0  # for each frame, look at change in future
    traj_change_future_frames = fps_ * traj_change_future_seconds
    stationary_thres = 0.08  # from experience
    for person_id in person_ids:
        this_data = filtered_data[filtered_data[:, 1] == person_id, :]
        # here this_data should already be sorted by frame_id ASC
        if this_data.shape[0] <= 1:
            continue
        if interpolate:
            # interpolate the points in between
            # assuming constant velocity
            new_data = []
            for i in range(this_data.shape[0] - 1):
                j = i + 1
                # add the start point
                new_data.append(this_data[i])
                new_data += interpolate_data_between(this_data[i], this_data[j])
            new_data.append(this_data[-1])
            this_data = np.array(new_data, dtype="float64")

        is_stationary_before_end = False  # use this for last few frames
        for i in range(this_data.shape[0] - 1):
            # start from zero
            frame_id = int(this_data[i, 0] - target_frame_ids[0])
            if no_offset:
                frame_id = int(this_data[i, 0])
            j = i + 1
            # compute the future changes
            future_i = None
            for t in range(j, this_data.shape[0]):
                if this_data[t, 0] - this_data[i, 0] >= traj_change_future_frames:
                    future_i = t
                    break
            is_stationary = False
            if future_i is not None:
                diff = np.linalg.norm(this_data[future_i, 2:] - this_data[i, 2:])
                if diff <= stationary_thres:
                    is_stationary = True
                    is_stationary_before_end = True
            else:
                is_stationary = is_stationary_before_end
            # direction vector
            direction_vector, speed, time_elasped = get_direction_and_speed(
                this_data[j], this_data[i], fps_)
            # if not control_data.has_key(frame_id):
            if frame_id not in control_data:
                control_data[frame_id] = []
            control_data[frame_id].append(
                [person_id, this_data[i, 0], this_data[i, 2:].tolist(),
                 direction_vector, speed,
                 time_elasped, is_stationary])

        last_frame_id = int(this_data[-1, 0] - target_frame_ids[0])
        if no_offset:
            last_frame_id = int(this_data[-1, 0])
        # if not control_data.has_key(last_frame_id):
        if last_frame_id not in control_data:
            control_data[last_frame_id] = []
        # signaling stop
        control_data[last_frame_id].append(
            [person_id, this_data[i, 0], this_data[-1, 2:].tolist(),
             None, None, None, None])
    return control_data, total_frame_num


def cleanup_actors(actor_list, client):
    for actor in actor_list:
        if actor.type_id.startswith("sensor") and actor.is_alive:
            actor.stop()
    # finished, clean actors
    if actor_list:
        client.apply_batch(
            [carla.command.DestroyActor(x) for x in actor_list])


def control_data_to_traj(control_data):
    """Convert the control data back to trajectory data."""
    # person/vehicle ID -> a list of [frame_id, xyz, is_stationary]
    traj_data = {}
    frame_ids = {}
    for frame_id in control_data:
        for one in control_data[frame_id]:
            frame_id = int(frame_id)
            p_id, ori_frame_id, xyz, _, speed, time_elasped, is_stationary = \
                one
            if p_id not in traj_data:
                traj_data[p_id] = []
            traj_data[p_id].append({
                "frame_id": frame_id,
                "xyz": xyz,
                "is_stationary": is_stationary,
                "speed": speed})
            frame_ids[frame_id] = 1
    for p_id in traj_data:
        traj_data[p_id].sort(key=operator.itemgetter("frame_id"))
    return traj_data, sorted(frame_ids.keys())


speed_calibration = 1.0  # used to account for the acceleration period


# speed_calibration = 1.22  # used to account for the acceleration period
# speed_calibration = 0.5  # to account for mismatch between sim and ground truth


def get_direction_and_speed(destination, current, fps_):
    """destination.xyz - current.xyz then normalize. also get the speed."""
    direction_vector = [
        destination[2] - current[2],
        destination[3] - current[3],
        0.0]
    vector_length = math.sqrt(sum([x ** 2 for x in direction_vector])) + \
                    np.finfo(float).eps
    direction_vector = [x / vector_length for x in direction_vector]
    direction_vector = [float(x) for x in direction_vector]

    time_elasped = (destination[0] - current[0]) / fps_
    speed = vector_length / time_elasped * speed_calibration  # meter per second

    return direction_vector, speed, time_elasped


def get_scene(videoname_):
    """ActEV scene extractor from videoname."""
    s = videoname_.split("_S_")[-1]
    s = s.split("_")[0]
    return s[:4]


class CollisionSensor(object):
    def __init__(self, parent_actor, actorid2info, world, verbose=False):
        self.world = world
        self.verbose = verbose
        self.parent = parent_actor
        self.actorid2info = actorid2info
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(),
                                        attach_to=self.parent)
        self.parent_tag = actorid2info[parent_actor.id]
        self.history = []
        self.sensor.listen(self.on_collision)

    def on_collision(self, event):
        frame_id = event.frame
        other_actor_id = event.other_actor.id
        other_actor_carla_tag = event.other_actor.type_id
        other_actor_tag = None
        # if not self.actorid2info.has_key(other_actor_id):
        if other_actor_id not in self.actorid2info:
            if self.verbose:
                print("%s: %s collide with %s." % (
                    frame_id,
                    self.parent_tag, other_actor_carla_tag))
        else:
            other_actor_tag = self.actorid2info[other_actor_id]
            if self.verbose:
                print("%s: %s collide with %s." % (
                    frame_id,
                    self.parent_tag, other_actor_tag))

        self.history.append((frame_id, self.parent.id, other_actor_id,
                             self.parent_tag, other_actor_tag,
                             other_actor_carla_tag))


def setup_static(world, client, scene_elements, actor_list):
    this_weather = scene_elements["weather"]
    weather = carla.WeatherParameters(
        cloudiness=this_weather["cloudiness"],
        precipitation=this_weather["precipitation"],
        precipitation_deposits=this_weather["precipitation_deposits"],
        sun_altitude_angle=this_weather["sun_altitude_angle"],
        sun_azimuth_angle=this_weather["sun_azimuth_angle"],
        wind_intensity=this_weather["wind_intensity"])
    world.set_weather(weather)

    spawn_cmds = []
    for car in scene_elements["static_cars"]:
        car_location = carla.Location(x=car["location_xyz"][0],
                                      y=car["location_xyz"][1],
                                      z=car["location_xyz"][2])
        car_rotation = carla.Rotation(pitch=car["rotation_pyr"][0],
                                      yaw=car["rotation_pyr"][1],
                                      roll=car["rotation_pyr"][2])
        car_bp = world.get_blueprint_library().find(car["bp"])
        assert car_bp is not None, car_bp
        # the static car can be walked though
        spawn_cmds.append(
            carla.command.SpawnActor(
                car_bp, carla.Transform(
                    location=car_location, rotation=car_rotation)).then(
                carla.command.SetSimulatePhysics(
                    carla.command.FutureActor, False)))

    # spawn the actors needed for the static scene setup
    if spawn_cmds:
        response = client.apply_batch_sync(spawn_cmds)
        all_actors = world.get_actors([x.actor_id for x in response])
        actor_list += all_actors


lidar_sensor = None
def run_sim_for_one_frame(frame_id, ped_controls, cur_peds, walker_bps,
                          world,
                          global_actor_list, actorid2info, robot_id=-1,
                          show_traj=False, verbose=False, max_yaw_change=60,
                          exit_if_spawn_fail=False,
                          no_collision_detector=False,
                          pid2actor={},
                          excepts=[]):
    """Given the controls and the current frame_id, run the simulation. Return
     the batch command to execute, return None if the any spawning failed
  """
    stats = {}
    batch_cmds = []

    # if ped_controls.has_key(frame_id):
    if frame_id in ped_controls:
        this_control_data = ped_controls[frame_id]
        for person_id, _, xyz, direction_vector, speed, time_elasped, is_static in this_control_data:
            if person_id in excepts:
                continue

            if robot_id == person_id:
                if robot_id not in cur_peds:
                    kid_bp = world.get_blueprint_library().find("walker.pedestrian.0011")
                    kid_walker = world.try_spawn_actor(kid_bp, carla.Transform(
                        location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]),
                        rotation=carla.Rotation(yaw=np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])),
                                                pitch=0, roll=0)))
                    lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
                    lidar_bp.set_attribute('range', '10.0')
                    lidar_bp.set_attribute('channels', '64')
                    lidar_bp.set_attribute('lower_fov', '-20.0')
                    lidar_bp.set_attribute('sensor_tick', '0.2')
                    global lidar_sensor
                    lidar_sensor = world.try_spawn_actor(lidar_bp,
                                                  carla.Transform(location=carla.Location(x=0.2, y=0, z=+1.4)),
                                                  attach_to=kid_walker)
                    lidar_sensor.listen(lambda data: get_sensor_data(data))
                    cur_peds[person_id] = kid_walker
                    global_actor_list.append(kid_walker)
                    actorid2info[kid_walker.id] = ("Person", person_id)
                    pid2actor[person_id] = kid_walker
                    print("created robot as ped=", robot_id)
                    continue

                # ---- last location reached ----
                if direction_vector is None:
                    lidar_sensor.stop()

            # ------ last location reached, so delete this guy ------
            if direction_vector is None:
                if person_id in cur_peds:
                    batch_cmds.append(carla.command.DestroyActor(cur_peds[person_id]))
                    del cur_peds[person_id]

            else:
                walker_control = carla.WalkerControl(direction=carla.Vector3D(x=direction_vector[0],
                                                                              y=direction_vector[1],
                                                                              z=direction_vector[2]),
                                                     speed=speed, jump=False)
                # carla.command.applyTransform()

                # new person, need to spawn
                # if not cur_peds.has_key(person_id):
                if person_id not in cur_peds:
                    walker_bp = get_bp(walker_bps)
                    new_walker = world.try_spawn_actor(walker_bp, carla.Transform(
                        location=carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]),
                        rotation=carla.Rotation(yaw=np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])),
                                                pitch=0, roll=0)))
                    if new_walker is None:
                        if verbose:
                            print("%s: person %s failed to spawn." % (frame_id, person_id))
                        if exit_if_spawn_fail:
                            return None, stats
                        else:
                            continue
                    if verbose:
                        print("%s: Spawned person id %s." % (
                            frame_id, person_id))
                    new_walker.set_simulate_physics(True)
                    cur_peds[person_id] = new_walker
                    global_actor_list.append(new_walker)

                    # add a collision sensor
                    actorid2info[new_walker.id] = ("Person", person_id)
                    pid2actor[person_id] = new_walker

                    # if not no_collision_detector:
                    #     collision_sensor = CollisionSensor(new_walker, actorid2info, world, verbose=verbose)
                    #     global_actor_list.append(collision_sensor.sensor)
                    #     cur_ped_collisions[person_id] = collision_sensor
                    if show_traj:
                        # show the track Id
                        world.debug.draw_string(carla.Location(
                            x=xyz[0], y=xyz[1], z=xyz[2]), "# %s" % person_id,
                            draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0),
                            life_time=30.0)

                this_walker_actor = cur_peds[person_id]

                if show_traj:
                    delay = 0.0  # delay before removing traj
                    p1 = carla.Location(x=xyz[0], y=xyz[1], z=xyz[2])
                    next_xyz = [xyz[i] + direction_vector[i] * speed * time_elasped
                                for i in range(3)]
                    p2 = carla.Location(x=next_xyz[0], y=next_xyz[1], z=next_xyz[2])
                    world.debug.draw_arrow(p1, p2, thickness=0.1, arrow_size=0.1,
                                           color=carla.Color(r=255),
                                           life_time=time_elasped + delay)

                if is_static:
                    # stop the walker
                    batch_cmds.append(carla.command.ApplyWalkerControl(
                        this_walker_actor, carla.WalkerControl()))
                    continue
                batch_cmds.append(carla.command.ApplyWalkerControl(
                    this_walker_actor, walker_control))

                sim_xyz = this_walker_actor.get_location()
                print(
                    "disp err (sim, real)= %3f" % np.linalg.norm(np.array(xyz[:2]) - np.array([sim_xyz.x, sim_xyz.y])))
                # compensate the sim error
                this_walker_actor.set_location(carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))

    return batch_cmds, stats


def cross(carla_vector3d_1, carla_vector3d_2):
    """Cross product."""
    return carla.Vector3D(
        x=carla_vector3d_1.y * carla_vector3d_2.z -
          carla_vector3d_1.z * carla_vector3d_2.y,
        y=carla_vector3d_1.z * carla_vector3d_2.x -
          carla_vector3d_1.x * carla_vector3d_2.z,
        z=carla_vector3d_1.x * carla_vector3d_2.y -
          carla_vector3d_1.y * carla_vector3d_2.x)


def get_degree_of_two_vectors(vec1, vec2):
    x1, y1 = vec1
    x2, y2 = vec2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle_rad = np.arctan2(det, dot)
    return np.rad2deg(angle_rad)


def parse_carla_depth(depth_image):
    """Parse Carla depth image."""
    # 0.9.6: The image codifies the depth in 3 channels of the RGB color space,
    # from less to more significant bytes: R -> G -> B.
    # depth_image is [h, w, 3], last dim is RGB order
    depth_image = depth_image.astype("float32")
    normalized = (depth_image[:, :, 0] + depth_image[:, :, 1] * 256 + \
                  depth_image[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
    return 1000 * normalized


def compute_intrinsic(img_width, img_height, fov):
    """Compute intrinsic matrix."""
    intrinsic = np.identity(3)
    intrinsic[0, 2] = img_width / 2.0
    intrinsic[1, 2] = img_height / 2.0
    intrinsic[0, 0] = intrinsic[1, 1] = img_width / (2.0 * np.tan(fov *
                                                                  np.pi / 360.0))
    return intrinsic


def compute_extrinsic_from_transform(transform_):
    """
  Creates extrinsic matrix from carla transform.
  This is known as the coordinate system transformation matrix.
  """

    rotation = transform_.rotation
    location = transform_.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))  # matrix is needed
    # 3x1 translation vector
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    # 3x3 rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    # [3, 3] == 1, rest is zero
    return matrix


def save_rgb_image(rgb_np_img, save_file):
    """Convert the RGB numpy image for cv2 to save."""
    # RGB np array
    image_to_save = rgb_np_img
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    save_path = os.path.dirname(save_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_file, image_to_save)


def to_xyz(carla_vector3d):
    """Return xyz coordinates as list from carla vector3d."""
    return [carla_vector3d.x, carla_vector3d.y, carla_vector3d.z]


def make_text_surface(text, offset):
    """Put text on a black bg pygame surface."""
    font = pygame.font.Font(pygame.font.get_default_font(), 20)
    text_width, text_height = font.size(text)

    surface = pygame.Surface((text_width, text_height))
    surface.fill((0, 0, 0, 0))  # black bg
    text_texture = font.render(text, True, (255, 255, 255))  # white color
    surface.blit(text_texture, (0, 0))  # upper-left corner

    return surface, offset + surface.get_height()


def get_2d_bbox(bbox_3d, max_w, max_h):
    """Given the computed [8, 3] points with depth, get the one bbox."""
    if all(bbox_3d[:, 2] > 0):
        # make one 2d bbox from 8 points
        x1 = round(np.min(bbox_3d[:, 0]), 3)
        y1 = round(np.min(bbox_3d[:, 1]), 3)
        x2 = round(np.max(bbox_3d[:, 0]), 3)
        y2 = round(np.max(bbox_3d[:, 1]), 3)
        if x1 > max_w or y1 > max_h:
            return None
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > max_w:
            x2 = max_w
        if y2 > max_h:
            y2 = max_h
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    else:
        return None


def get_3d_bbox(actor_, camera_actor):
    """Get the 8 point coordinates of the actor in the camera view."""
    # 1. get the 8 vertices of the actor box
    vertices = np.zeros((8, 4), dtype="float")
    extent = actor_.bounding_box.extent  # x, y, z extension from the center
    vertices[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    vertices[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    vertices[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    vertices[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    vertices[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    vertices[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    vertices[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    vertices[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    # vertices coordinates are relative to the actor (0, 0, 0)
    # now get the world coordinates
    # the bounding_box.location is all 0?
    center_transform = carla.Transform(actor_.bounding_box.location)
    center_rt = compute_extrinsic_from_transform(center_transform)
    # actor_ is in the world coordinates
    actor_rt = compute_extrinsic_from_transform(actor_.get_transform())
    # dont know why
    bbox_rt = np.dot(actor_rt, center_rt)

    # vertices relative to the bbox center
    # bbox center relative to the parent actor
    # so :
    # [4, 8] # these are in the world coordinates
    world_vertices = np.dot(bbox_rt, np.transpose(vertices))

    # now we transform vertices in world
    # to the camera 3D coordinate system
    camera_rt = compute_extrinsic_from_transform(camera_actor.get_transform())
    camera_rt_inv = np.linalg.inv(camera_rt)
    # [3, 8]
    x_y_z = np.dot(camera_rt_inv, world_vertices)[:3, :]
    # wadu hek? why?, unreal coordinates problem?
    # email me (junweil@cs.cmu.edu) if you know why
    y_minus_z_x = np.concatenate(
        [x_y_z[1, :], - x_y_z[2, :], x_y_z[0, :]], axis=0)

    # then we dot the intrinsic matrix then we got the pixel coordinates and ?
    # [8, 3]
    actor_bbox = np.transpose(np.dot(camera_actor.intrinsic, y_minus_z_x))
    # last dim keeps the scale factor?
    actor_bbox = np.concatenate(
        [actor_bbox[:, 0] / actor_bbox[:, 2],
         actor_bbox[:, 1] / actor_bbox[:, 2], actor_bbox[:, 2]], axis=1)

    return actor_bbox


# -------------- visualization

def draw_boxes(im, boxes, labels=None, colors=None, font_scale=0.6,
               font_thick=1, box_thick=1, bottom_text=False):
    """Draw boxes with labels on an image."""

    # boxes need to be x1, y1, x2, y2
    if not boxes:
        return im

    boxes = np.asarray(boxes, dtype="int")

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = font_scale

    if labels is not None:
        assert len(labels) == len(boxes)
    if colors is not None:
        assert len(labels) == len(colors)

    im = im.copy()

    for i in range(len(boxes)):
        box = boxes[i]

        color = (218, 218, 218)
        if colors is not None:
            color = colors[i]

        lineh = 2  # for box enlarging, replace with text height if there is label
        if labels is not None:
            label = labels[i]

            # find the best placement for the text
            ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, font_thick)
            bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
            top_left = [box[0] + 1, box[1] - 1.3 * lineh]
            if top_left[1] < 0:  # out of image
                top_left[1] = box[3] - 1.3 * lineh
                bottom_left[1] = box[3] - 0.3 * lineh

            textbox = [int(top_left[0]), int(top_left[1]),
                       int(top_left[0] + linew), int(top_left[1] + lineh)]

            if bottom_text:
                cv2.putText(im, label, (box[0] + 2, box[3] - 4),
                            FONT, FONT_SCALE, color=color)
            else:
                cv2.putText(im, label, (textbox[0], textbox[3]),
                            FONT, FONT_SCALE, color=color)  # , lineType=cv2.LINE_AA)

        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                      color=color, thickness=box_thick)
    return im
