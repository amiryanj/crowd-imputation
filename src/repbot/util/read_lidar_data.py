#!/usr/bin/env python

import os
import glob
import numpy as np

# Raw Data directory
LiDAR_DATA_DIR = "/home/cyrus/Videos/carla/lidar"
WITH_ROS = False
if WITH_ROS:
    import rospy
    from sensor_msgs.msg import PointCloud, PointCloud2
    from geometry_msgs.msg import Point32, TransformStamped
    from tf2_msgs.msg import TFMessage
    from std_msgs.msg import Header


def read_lidar_data(data_dir=LiDAR_DATA_DIR):
    lidar_files = sorted(glob.glob(os.path.join(data_dir, '*.dat')))

    point_cloud_data = {}
    for lidar_file in lidar_files:
        base_name = int(os.path.basename(lidar_file)[:-4])
        # if base_name not in frame_range: continue
        point_cloud_data[base_name] = []
        with open(lidar_file, 'r') as fp:
            for cnt, line in enumerate(fp):
                if cnt < 8: continue
                point_cloud_data[base_name].append(line.split())
        if len(point_cloud_data[base_name]):
            point_cloud_data[base_name] = np.array(point_cloud_data[base_name]).astype(np.float64)
        else:
            del point_cloud_data[base_name]
    return point_cloud_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk

    # Optionally, specify the frame range to load
    # frame_range = range(20545, 20546)
    frame_range = range(0, 100000000)

    lidar_data = read_lidar_data(LiDAR_DATA_DIR)

    _break = False
    _pause = False


    def p(event):
        global _break, _pause
        if event.key == 'escape':
            _break = True
        if event.key == ' ':
            _pause = not _pause


    if WITH_ROS:
        rospy.init_node('plot_lidar')
        rate = rospy.Rate(10)
        ros_pc_pub = rospy.Publisher("/point_cloud", PointCloud, queue_size=10, latch=False)
        ros_tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=1)

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', p)

    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    for frame_id, scan in sorted(lidar_data.items()):
        if frame_id not in frame_range:
            continue

        skip = 1  # plot one in every `skip` points
        data_range = range(0, scan.shape[0], skip)  # skip points to prevent crash

        plt.cla()
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])
        # ax.set_zlim([0, 5])
        ax.scatter(scan[data_range, 0],  # x
                   -scan[data_range, 1],  # y
                   # scan[data_range, 2],  # z
                   c=scan[data_range, 3],  # reflectance
                   s=2, cmap='viridis')
        ax.set_title('Lidar scan %s' % frame_id)
        plt.grid()

        if WITH_ROS and not rospy.is_shutdown():
            pc_msg = PointCloud()
            pc_msg.points = [Point32(xyz[0], xyz[1], xyz[2]) for xyz in scan]
            ros_pc_pub.publish(pc_msg)

            t = TransformStamped()
            t.header.frame_id = "turtle1"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "carrot1"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 2.0
            t.transform.translation.z = 0.0

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            tfm = TFMessage([t])
            ros_tf_pub.publish(tfm)

        plt.pause(0.25)

        if _break:
            break
        while _pause:
            plt.pause(0.25)

    if WITH_ROS:
        rospy.spin()
    else:
        plt.show()
