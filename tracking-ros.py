#!/usr/bin/env python

class RosTracking:
    def __init__(self):
        rospy.init_node('crowd_prediction', anonymous=True)
        self.detection_sub = rospy.Subscriber(
            '/drow_detected_persons', DetectedPersons, self.detection_callback, queue_size=10)

        # self.visual_sub = rospy.Subscriber(
        #     '/visualization_marker', MarkerArray, self.vis_callback, queue_size=10)

        rospy.spin()

    def detection_callback(self, data):
        print('New detection message received')
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
        print('num of detections = ', len(data.detections), '\n***********************')
        for ii, det in enumerate(data.detections):
            print('id = ', det.detection_id)

    def vis_callback(self, data):
        print('New Visualization message received')
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)


if __name__ == '__main__':
    import rospy
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray

    # Customized Messages
    from frame_msgs.msg import DetectedPerson, DetectedPersons

    ros_tracker = RosTracking()

