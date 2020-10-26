#!/usr/bin/env python

import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from frame_msgs.msg import DetectedPersons, TrackedPersons
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from scipy.spatial.transform import Rotation
from followbot.scenarios.real_scenario import RealScenario
from followbot.gui.visualizer import *


class SimulationNode:
    def __init__(self):
        scan_topic = rospy.get_param("~publisher/scan/topic")
        sim_model = rospy.get_param("~simulation/sim_model")
        flow_2d = rospy.get_param("~simulation/flow_2d")
        n_pedestrians = rospy.get_param("~simulation/n_pedestrians")
        biped = rospy.get_param("~simulation/biped")
        robot_tf_topic = rospy.get_param("~publisher/robot_tf/topic")
        robot_vis_topic = rospy.get_param("~publisher/robot_vis/topic")
        objects_vis_topic = rospy.get_param("~publisher/objects_vis/topic")

        # scenario = RoundTrip(n_pedestrians)
        # scenario.setup(sim_model, flow_2d=flow_2d)

        self.scenario = RealScenario()
        self.scenario.setup(biped=biped)
        robot = self.scenario.world.robots[0]

        self.scan_pub = rospy.Publisher(scan_topic, LaserScan, queue_size=1)
        self.robot_transform_pub = rospy.Publisher(robot_tf_topic, TransformStamped, queue_size=1)
        self.robot_vis_pub = rospy.Publisher(robot_vis_topic, Marker, queue_size=1)
        self.objects_vis_pub = rospy.Publisher(objects_vis_topic, MarkerArray, queue_size=1)
        self.detection_sub = rospy.Subscriber('/fusion/detected_persons_synchronized', DetectedPersons, self.callback_detection)
        self.tracking_sub = rospy.Subscriber('/rwth_tracker/tracked_persons', TrackedPersons, self.callback_tracking)
        self.rate = rospy.Rate(10)  # Hz

        world_objects = MarkerArray()
        for ii, obj in enumerate(self.scenario.world.obstacles):
            obj_i_marker = Marker()
            obj_i_marker.color.a = 1.0
            obj_i_marker.color.r = 1.0
            obj_i_marker.color.g = 0.0
            obj_i_marker.color.b = 0.0
            obj_i_marker.id = ii
            obj_i_marker.ns = "Objects"
            obj_i_marker.header.frame_id = "myFixedFrame"
            obj_i_marker.header.stamp = rospy.Time.now()
            obj_i_marker.action = Marker.ADD
            if isinstance(obj, Line):
                obj_i_marker.type = Marker.LINE_STRIP
                pt1, pt2 = Point(), Point()
                pt1.x, pt1.y = obj.line[0]
                pt2.x, pt2.y = obj.line[1]
                obj_i_marker.points.append(pt1)
                obj_i_marker.points.append(pt2)
                obj_i_marker.scale.x = 0.1
                world_objects.markers.append(obj_i_marker)
            elif isinstance(obj, Circle):
                obj_i_marker.type = Marker.CYLINDER
                obj_i_marker.scale.x = obj.radius * 2
                obj_i_marker.scale.y = obj.radius * 2
                obj_i_marker.scale.z = 1
                obj_i_marker.pose.position.x = obj.center[0]
                obj_i_marker.pose.position.y = obj.center[1]
                obj_i_marker.pose.position.z = obj_i_marker.scale.z / 2.
                world_objects.markers.append(obj_i_marker)

        counter = -1
        while not rospy.is_shutdown():
            counter += 1
            self.scenario.step_crowd(dt=0.1)
            self.scenario.step_robots(dt=0.1)
            self.scenario.update_disply()

            # for ped_i in self.scenario.world.crowds:
            #     print(ped_i.pos)

            scan_msg = LaserScan()
            # msg_time = rospy.get_rostime()

            transform_msg = TransformStamped()

            transform_msg.transform.translation.x = robot.pos[0]
            transform_msg.transform.translation.y = robot.pos[1]
            transform_msg.transform.translation.z = 0

            robot_rot = Rotation.from_euler('z', robot.orien, degrees=False).as_quat()
            transform_msg.transform.rotation.x = robot_rot[0]
            transform_msg.transform.rotation.y = robot_rot[1]
            transform_msg.transform.rotation.z = robot_rot[2]
            transform_msg.transform.rotation.w = robot_rot[3]

            scan_msg.header.stamp = Clock()
            scan_msg.header.stamp = rospy.Time(0, 0)
            scan_msg.header.seq = counter
            scan_msg.header.frame_id = "myFixedFrame"  # FIXME: what is this?

            scan_msg.range_min = robot.lidar.range_min  # 0.05m
            scan_msg.range_max = robot.lidar.range_max  # 8m => SICK TiM571
            scan_msg.angle_min = robot.lidar.angle_min_radian()
            scan_msg.angle_max = robot.lidar.angle_max_radian()
            scan_msg.angle_increment = robot.lidar.angle_increment_radian()
            scan_msg.time_increment = robot.lidar.time_increment()

            scan_msg.scan_time = 0.0
            scan_msg.ranges = robot.lidar.last_range_data
            scan_msg.intensities = robot.lidar.last_intensities  # not important

            robot_marker = Marker()
            robot_marker.type = Marker.CUBE
            robot_marker.scale.x = 0.3  # depth
            robot_marker.scale.y = 0.6  # width
            robot_marker.scale.z = 1.20  # height
            robot_marker.pose.position.x = robot.pos[0]
            robot_marker.pose.position.y = robot.pos[1]
            robot_marker.pose.position.z = robot_marker.scale.z / 2.
            robot_marker.pose.orientation.x = robot_rot[0]
            robot_marker.pose.orientation.y = robot_rot[1]
            robot_marker.pose.orientation.z = robot_rot[2]
            robot_marker.pose.orientation.w = robot_rot[3]
            robot_marker.color.a = 1.0
            robot_marker.color.r = 1.0
            robot_marker.color.g = 0.0
            robot_marker.color.b = 0.7
            robot_marker.id = 0
            robot_marker.ns = "robot"
            robot_marker.header.frame_id = "myFixedFrame"
            robot_marker.header.stamp = rospy.Time.now()
            robot_marker.action = Marker.MODIFY
            self.robot_vis_pub.publish(robot_marker)

            self.robot_transform_pub.publish(transform_msg)
            self.scan_pub.publish(scan_msg)
            self.objects_vis_pub.publish(world_objects)
            self.rate.sleep()

    def callback_detection(self, detection_msg):
        for track in detection_msg.detections:
            p = [track.pose.pose.position.x, track.pose.pose.position.y]
            self.scenario.visualizer.draw_circle(p, 12, BLUE_LIGHT, 5)
        pygame.display.update()

    def callback_tracking(self, track_msg):
        # print(track_msg)
        for track in track_msg.tracks:
            p = [track.pose.pose.position.x, track.pose.pose.position.y]
            self.scenario.visualizer.draw_circle(p, 12, RED_COLOR, 3)
        pygame.display.update()

    def robot_nav_callback(self):
        pass

    def clock_callback(self):
        pass


if __name__ == '__main__':
    rospy.init_node('followbot_sim')
    print('followbot simulator node is running ...')
    try:
        SimulationNode()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

