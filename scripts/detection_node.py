#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from frame_msgs.msg import DetectedPerson, DetectedPersons
from repbot.tracking.tracking import PedestrianDetection

import numpy as np


class SimpleDetection:
    def __init__(self):
        self.detector = PedestrianDetection(range_max_=8.0)
        # self.tracker = MultiObjectTracking()
        self.detection_counter = 0
        self.scan_sub = []
        self.detection_pub = []

    def run_ros(self):
        scan_topic = rospy.get_param("~publisher/scan/topic")
        detection_topic = rospy.get_param("~subscriber/detections0/topic")
        queue_size = rospy.get_param("~publisher/detections/queue_size")
        latch = rospy.get_param("~publisher/detections/latch")
        self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.callback_scan)
        self.detection_pub = rospy.Publisher(detection_topic, DetectedPersons, queue_size=queue_size, latch=latch)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def callback_scan(self, scan_msg):
        print('detection received new data ...')
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        segments = self.detector.cluster_range_data(scan_msg.ranges, angles)
        detections, walls = self.detector.detect(segments, [0, 0])
        detections_msg = self.encode_msg(detections)
        self.detection_pub.publish(detections_msg)

    def encode_msg(self, detections):
        detections_msg = DetectedPersons()
        detections_msg.header.frame_id = "myFixedFrame"
        for det in detections:
            det_msg = DetectedPerson()

            self.detection_counter += 1
            det_msg.detection_id = self.detection_counter
            det_msg.confidence = 0.9  # at the moment there is no confidence information

            det_msg.pose.pose.position.x = det[0]
            det_msg.pose.pose.position.y = det[1]
            det_msg.pose.pose.position.z = 0.3  # fixed: Height of detection
            det_msg.pose.pose.orientation.x = 0
            det_msg.pose.pose.orientation.y = 0
            det_msg.pose.pose.orientation.z = 0
            det_msg.pose.pose.orientation.w = 1

            cov = np.zeros([6, 6])
            cov[0, 0] = cov[1, 1] = cov[2, 2] = 0.05
            cov[3, 3] = cov[4, 4] = cov[5, 5] = 1000000.0
            det_msg.pose.covariance = cov.flatten().tolist()
            det_msg.height = 1.85
            det_msg.modality = "laser2d"
            det_msg.embed_vector = []

            detections_msg.detections.append(det_msg)
        return detections_msg


if __name__ == '__main__':
    rospy.init_node('simple_detection')
    print('***** detection node started *****')
    try:
        sd = SimpleDetection()
        sd.run_ros()
    except rospy.ROSInterruptException:
        pass
