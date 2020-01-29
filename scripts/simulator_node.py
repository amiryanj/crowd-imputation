#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from followbot.followbot_sim import RoundTrip


def run_sim():
    scan_topic = rospy.get_param("~publisher/scan/topic")
    sim_model = rospy.get_param("~simulation/sim_model")
    flow_2d = rospy.get_param("~simulation/flow_2d")
    n_pedestrians = rospy.get_param("~simulation/n_pedestrians")

    scenario = RoundTrip(n_pedestrians)
    scenario.setup(sim_model, flow_2d=flow_2d)

    pub = rospy.Publisher(scan_topic, LaserScan, queue_size=1)
    rate = rospy.Rate(10)  # Hz

    counter = -1
    while not rospy.is_shutdown():
        counter += 1
        scenario.step(save=False)

        msg = LaserScan()
        # msg_time = rospy.get_rostime()
        # print(msg_time)

        robot = scenario.world.robots[0]

        msg.header.stamp = Clock()
        msg.header.stamp = rospy.Time(0, 0)
        msg.header.seq = counter
        msg.header.frame_id = "frontLaser"  # FIXME: what is this?

        msg.range_min = robot.lidar.range_min  # 0.05m
        msg.range_max = robot.lidar.range_max  # 8m => SICK TiM571
        msg.angle_min = robot.lidar.angle_min_radian()
        msg.angle_max = robot.lidar.angle_max_radian()
        msg.angle_increment = robot.lidar.angle_increment_radian()
        msg.time_increment = robot.lidar.time_increment()

        msg.scan_time = 0.0
        msg.ranges = robot.lidar.last_range_data
        msg.intensities = robot.lidar.last_intensities

        pub.publish(msg)
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

