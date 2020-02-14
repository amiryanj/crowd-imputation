#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from followbot.followbot_sim import RoundTrip, RealScenario
from scipy.spatial.transform import Rotation


def run_sim():
    scan_topic = rospy.get_param("~publisher/scan/topic")
    sim_model = rospy.get_param("~simulation/sim_model")
    flow_2d = rospy.get_param("~simulation/flow_2d")
    n_pedestrians = rospy.get_param("~simulation/n_pedestrians")

    # scenario = RoundTrip(n_pedestrians)
    # scenario.setup(sim_model, flow_2d=flow_2d)

    scenario = RealScenario()
    scenario.setup()

    scan_pub = rospy.Publisher(scan_topic, LaserScan, queue_size=1)
    transform_pub = rospy.Publisher('/transform/robot', TransformStamped, queue_size=1)
    rate = rospy.Rate(10)  # Hz

    counter = -1
    while not rospy.is_shutdown():
        counter += 1
        scenario.step(save=False)

        scan_msg = LaserScan()
        # msg_time = rospy.get_rostime()
        # print(msg_time)

        robot = scenario.world.robots[0]
        transform_msg = TransformStamped()

        transform_msg.transform.translation.x = robot.pos[0]
        transform_msg.transform.translation.y = robot.pos[1]
        transform_msg.transform.translation.z = 0

        robot_rot = Rotation.from_euler('z', robot.orien, degrees=False).as_quat()
        print('robot pos = ', robot.pos, '| orien =', robot_rot)
        transform_msg.transform.rotation.x = robot_rot[0]
        transform_msg.transform.rotation.y = robot_rot[1]
        transform_msg.transform.rotation.z = robot_rot[2]
        transform_msg.transform.rotation.w = robot_rot[3]

        scan_msg.header.stamp = Clock()
        scan_msg.header.stamp = rospy.Time(0, 0)
        scan_msg.header.seq = counter
        scan_msg.header.frame_id = "frontLaser"  # FIXME: what is this?

        scan_msg.range_min = robot.lidar.range_min  # 0.05m
        scan_msg.range_max = robot.lidar.range_max  # 8m => SICK TiM571
        scan_msg.angle_min = robot.lidar.angle_min_radian()
        scan_msg.angle_max = robot.lidar.angle_max_radian()
        scan_msg.angle_increment = robot.lidar.angle_increment_radian()
        scan_msg.time_increment = robot.lidar.time_increment()

        scan_msg.scan_time = 0.0
        scan_msg.ranges = robot.lidar.last_range_data
        scan_msg.intensities = robot.lidar.last_intensities

        transform_pub.publish(transform_msg)
        scan_pub.publish(scan_msg)
        rate.sleep()


def robot_nav_callback():
    pass


def clock_callback():
    pass


if __name__ == '__main__':
    rospy.init_node('followbot_sim')
    print('followbot simulator node is running ...')
    try:
        run_sim()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

