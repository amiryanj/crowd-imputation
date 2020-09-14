#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from frame_msgs.msg import DetectedPerson, DetectedPersons, TrackedPersons
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from crowd_prediction.srv \
    import GetSynthesizedCrowd, GetSynthesizedCrowdResponse, GetPredictedTrajectories, GetPredictedTrajectoriesResponse


class RobotNode:
    def __init__(self):
        self.nav_pub = rospy.Publisher('/navigation', Twist, queue_size=1)
        rate = rospy.Rate(10)

        self.detection_subscriber = rospy.Subscriber('/rwth_tracker/tracked_persons', TrackedPersons, self.callback_tracking)
        # self.tracking_subscriber = rospy.Subscriber('/rwth_tracker/tracked_persons', TrackedPersons, self.callback_tracking)
        # /rwth_tracker/pedestrian_array [rwth_perception_people_msgs/PedestrianTrackingArray]
        # /rwth_tracker/tracked_persons [frame_msgs/TrackedPersons]

        self.tracks = dict()

        while not rospy.is_shutdown():
            rate.sleep()

    def callback_tracking(self, track_msg):
        for track in track_msg.tracks:
            if not track.track_id in self.tracks:
                self.tracks[track.track_id] = []
            self.tracks[track.track_id].append([track.pose.pose.position.x, track.pose.pose.position.y])
            # print(self.tracks)
        return

        try:
            # TODO: call crowd_synthesis()
            rospy.wait_for_service('crowd_synthesis')
            crowd_synthesis_srv = rospy.ServiceProxy('crowd_synthesis', GetSynthesizedCrowd)
            all_peds = crowd_synthesis_srv(track_msg)

            # TODO: call trajec_prediction()
            obsvs_msg = track_msg.obsvs
            n_next = 12
            n_samples = 1

            rospy.wait_for_service('trajec_prediction')
            trajec_prediction_srv = rospy.ServiceProxy('trajec_prediction', GetPredictedTrajectories)
            predicted_trajecs = trajec_prediction_srv(obsvs_msg, n_next, n_samples)

            # TODO:  Run Navigation and send command to sim
            twist_msg = Twist()
            self.nav_pub.publish(twist_msg)

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


if __name__ == '__main__':
    rospy.init_node('follow_bot')
    print('follow_bot node is running ...')
    try:
        RobotNode()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
