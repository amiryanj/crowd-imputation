import os
import numpy as np
from transforms3d.euler import euler2mat

class Joint:
    def __init__(self, name, direction, length, axis, dof, limits):
        """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
        self.name = name
        self.direction = np.reshape(direction, [3, 1])
        self.length = length
        axis = np.deg2rad(axis)
        self.C = euler2mat(*axis)
        self.Cinv = np.linalg.inv(self.C)
        self.limits = np.zeros([3, 2])
        for lm, nm in zip(limits, dof):
            if nm == 'rx':
                self.limits[0] = lm
            elif nm == 'ry':
                self.limits[1] = lm
            else:
                self.limits[2] = lm
        self.parent = None
        self.children = []
        self.coordinate = None
        self.matrix = None

    def set_motion(self, motion):
        if self.name == 'root':
            self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
            rotation = np.deg2rad(motion['root'][3:])
            self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
        else:
            idx = 0
            rotation = np.zeros(3)
            for axis, lm in enumerate(self.limits):
                if not np.array_equal(lm, np.zeros(2)):
                    rotation[axis] = motion[self.name][idx]
                    idx += 1
            rotation = np.deg2rad(rotation)
            self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
            self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
        for child in self.children:
            child.set_motion(motion)

    def draw(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        joints = self.to_dict()
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_xlim3d(-50, 10)
        ax.set_ylim3d(-20, 40)
        ax.set_zlim3d(-20, 40)

        xs, ys, zs = [], [], []
        for joint in joints.values():
            xs.append(joint.coordinate[0, 0])
            ys.append(joint.coordinate[1, 0])
            zs.append(joint.coordinate[2, 0])
        plt.plot(zs, xs, ys, 'b.')

        for joint in joints.values():
            segment_color = 'r'
            child = joint
            if child.parent is not None:
                parent = child.parent
                xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
                ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
                zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]

                if joint.name in ['lfemur', 'ltibia', 'rfemur', 'rtibia']:
                    segment_color = 'g'

                plt.plot(zs, xs, ys, segment_color)
        plt.show()

    def to_dict(self):
        ret = {self.name: self}
        for child in self.children:
            ret.update(child.to_dict())
        return ret

    def pretty_print(self):
        print('===================================')
        print('joint: %s' % self.name)
        print('direction:')
        print(self.direction)
        print('limits:', self.limits)
        print('parent:', self.parent)
        print('children:', self.children)


def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx


def parse_asf(file_path):
    '''read joint data only'''
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        # meta infomation is ignored
        if line == ':bonedata':
            content = content[idx + 1:]
            break

    # read joints
    joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
    idx = 0
    while True:
        # the order of each section is hard-coded

        line, idx = read_line(content, idx)

        if line[0] == ':hierarchy':
            break

        assert line[0] == 'begin'

        line, idx = read_line(content, idx)
        assert line[0] == 'id'

        line, idx = read_line(content, idx)
        assert line[0] == 'name'
        name = line[1]

        line, idx = read_line(content, idx)
        assert line[0] == 'direction'
        direction = np.array([float(axis) for axis in line[1:]])

        # skip length
        line, idx = read_line(content, idx)
        assert line[0] == 'length'
        length = float(line[1])

        line, idx = read_line(content, idx)
        assert line[0] == 'axis'
        assert line[4] == 'XYZ'

        axis = np.array([float(axis) for axis in line[1:-1]])

        dof = []
        limits = []

        line, idx = read_line(content, idx)
        if line[0] == 'dof':
            dof = line[1:]
            for i in range(len(dof)):
                line, idx = read_line(content, idx)
                if i == 0:
                    assert line[0] == 'limits'
                    line = line[1:]
                assert len(line) == 2
                mini = float(line[0][1:])
                maxi = float(line[1][:-1])
                limits.append((mini, maxi))

            line, idx = read_line(content, idx)

        assert line[0] == 'end'
        joints[name] = Joint(
            name,
            direction,
            length,
            axis,
            dof,
            limits
        )

    # read hierarchy
    assert line[0] == ':hierarchy'

    line, idx = read_line(content, idx)

    assert line[0] == 'begin'

    while True:
        line, idx = read_line(content, idx)
        if line[0] == 'end':
            break
        assert len(line) >= 2
        for joint_name in line[1:]:
            joints[line[0]].children.append(joints[joint_name])
        for nm in line[1:]:
            joints[nm].parent = joints[line[0]]

    return joints


def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx + 1:]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames


class MocapGaitSimulator:
    def __init__(self):
        # load mc data
        mocap_data_dir = os.path.abspath(os.path.join(__file__, '..', 'cmu-mocap', 'walk'))
        asf_path = os.path.join(mocap_data_dir, '07.asf')
        amc_path = os.path.join(mocap_data_dir, '07_01.amc')
        print('parsing mocap file [%s]' % asf_path)

        self.joints = parse_asf(asf_path)
        self.motions = parse_amc(amc_path)
        # joints['root'].draw()

        # ============================
        M2CM = 0.01
        self.fixed_height = 30 * M2CM  # the weight at which Lidar is installed
        self.leg_radius = 4 * M2CM
        self.scale = 0.07  # Fixme

        self.fps = 120
        self.one_period_duration = 126

        pr = [0, 0]
        pl = [0, 0]

        left_x_y_pool = []
        counter = 0
        self.progress_time = 0

    def step(self, dt):
        self.progress_time += dt  # * self.one_period_duration
        if self.progress_time > self.one_period_duration / self.fps:
            self.progress_time = 0
        self.joints['root'].set_motion(self.motions[int(self.progress_time * self.fps)])

        # Left leg
        root_coord = np.array(self.joints['root'].coordinate).reshape((1, 3)) * self.scale
        root_coord[0, 1] = 0  # the heights sounds Ok.
        left_leg = np.array([self.joints['lfemur'].coordinate, self.joints['ltibia'].coordinate]).squeeze() * self.scale
        right_leg = np.array([self.joints['rfemur'].coordinate, self.joints['rtibia'].coordinate]).squeeze() * self.scale

        # compensate character movement
        left_leg -= root_coord
        right_leg -= root_coord

        # I feel the height of root is reasonable. #Fixme: double check it

        # find the joints
        # find the intercept of lidar and bone
        t_left = (self.fixed_height - left_leg[0, 1]) / (np.diff(left_leg[:, 1]))
        left_leg_coincide_lidar = left_leg[0, :] + t_left * np.diff(left_leg, axis=0)

        t_right = (self.fixed_height - right_leg[0, 1]) / (np.diff(right_leg[:, 1]))
        right_leg_coincide_lidar = right_leg[0, :] + t_right * np.diff(right_leg, axis=0)

        self.left_leg = left_leg_coincide_lidar[0, ::2]
        self.right_leg = right_leg_coincide_lidar[0, ::2]

        # collect walking frames in a list to play afterward
        # left_x_y_pool.append(left_leg_coincide_lidar)
        
        # left_x_y_pool = np.stack(self.left_x_y_pool)
        # plt.plot(left_x_y_pool[:, 0, 0]*3, 'b')
        # plt.plot(left_x_y_pool[:, 0, 2], 'r')
        # plt.grid()
        # plt.show()


if __name__ == '__main__':
    print(__file__)
    exit(1)
    mocap_walk = MocapGaitSimulator()
    while True:
        mocap_walk.step()