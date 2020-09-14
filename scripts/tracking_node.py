#!/usr/bin/env python


class RosTracking:
    def __init__(self):
        rospy.init_node('crowd_prediction', anonymous=True)
        self.detection_sub = rospy.Subscriber(
            '/drow_detected_persons', DetectedPersons, self.detection_callback, queue_size=10)
        self.detection_sub = rospy.Subscriber(
            '/clock', Clock, self.clock_callback, queue_size=10)

        # self.visual_sub = rospy.Subscriber(
        #     '/visualization_marker', MarkerArray, self.vis_callback, queue_size=10)

        rospy.spin()

    def detection_callback(self, data):
        for ii, det in enumerate(data.detections):
            print('id = ', det.detection_id)

    def vis_callback(self, data):
        print('Received Visualization msg')
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)

    def clock_callback(self, data):
        print('times = ', data.clock.secs)


if __name__ == '__main__':
    import rospy
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
    from rosgraph_msgs.msg import Clock

    # Customized Messages
    from frame_msgs.msg import DetectedPerson, DetectedPersons

    ros_tracker = RosTracking()

