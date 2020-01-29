#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from followbot.followbot_sim import RoundTrip


def run_sim():
    scenario = RoundTrip()
    # TODO: read the topic from rospy.load_param('~xxx_topic')
    scenario.setup('powerlaw', flow_2d=True)

    pub = rospy.Publisher('/laser_front/scan', LaserScan, queue_size=1)
    rate = rospy.Rate(10)  # 10hz

    counter = -1
    while not rospy.is_shutdown():
        counter += 1
        scenario.step(save=False)

        msg = LaserScan()
        print(msg)
        print('**********************')

        msg_time = rospy.get_rostime()
        print(msg_time)

        # FIXME: values just copy pasted from crowdbot_sim
        robot = scenario.world.robots[0]

        msg.header.stamp = Clock()
        msg.header.stamp = rospy.Time(0, 0)
        msg.header.seq = counter
        msg.header.frame_id = "frontLaser"

        msg.range_min = robot.lidar.range_max  # 0.05m
        msg.range_max = robot.lidar.range_max  # 8m => SICK TiM571
        msg.angle_min = robot.lidar.min_angle_radian
        msg.angle_max = robot.lidar.max_angle_radian
        msg.angle_increment = robot.lidar.angle_increment_radian
        msg.time_increment = 2.31481481023e-05
        msg.scan_time = 0.0
        msg.ranges = robot.range_data
        msg.intensities = []

        print(msg)
        print('**********************')
        # exit(1)

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

