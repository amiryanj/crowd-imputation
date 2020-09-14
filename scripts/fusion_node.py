#!/usr/bin/env python
import numpy as np
import rospy
from frame_msgs.msg import DetectedPerson, DetectedPersons
from geometry_msgs.msg import TransformStamped
# import tf2_geometry_msgs
from followbot.util.transform import Transform, Rotation


class SimpleFusion:
    def __init__(self):
        detection_topic = rospy.get_param("~subscriber/detections0/topic")
        transform_topic = rospy.get_param("~subscriber/transform/topic")
        fusion_topic = rospy.get_param("~publisher/detections/topic")
        # print('detection_topic')
        # print('transform_topic')
        # print('fusion_topic')

        queue_size = rospy.get_param("~publisher/detections/queue_size")
        latch = rospy.get_param("~publisher/detections/latch")
        rate = rospy.Rate(10)

        self.detection_sub = rospy.Subscriber(detection_topic, DetectedPersons, self.callback_detection)
        self.transform_sub = rospy.Subscriber(transform_topic, TransformStamped, self.callback_transform)
        self.fusion_pub = rospy.Publisher(fusion_topic, DetectedPersons, queue_size=queue_size, latch=latch)

        self.tf = Transform()

        while not rospy.is_shutdown():
            rate.sleep()

    def callback_detection(self, msg):
        print('fusion node received detection msg')
        fusion_msg = DetectedPersons()
        fusion_msg.header = msg.header
        for det in msg.detections:
            out_det = DetectedPerson()
            out_det.detection_id = det.detection_id
            out_det.confidence = det.confidence
            out_det.height = det.height
            out_det.bbox_x = det.bbox_x
            out_det.bbox_y = det.bbox_y
            out_det.bbox_w = det.bbox_w
            out_det.bbox_h = det.bbox_h
            out_det.modality = det.modality
            out_det.embed_vector = det.embed_vector

            det_rot = Rotation.from_quat([det.pose.pose.orientation.x,
                                          det.pose.pose.orientation.y,
                                          det.pose.pose.orientation.z,
                                          det.pose.pose.orientation.w])
            det_trans = np.array([det.pose.pose.position.x, det.pose.pose.position.y, det.pose.pose.position.z])

            out_trans, out_orien = self.tf.apply(det_trans, det_rot)

            out_det.pose.pose.orientation.x = out_orien[0]
            out_det.pose.pose.orientation.y = out_orien[1]
            out_det.pose.pose.orientation.z = out_orien[2]
            out_det.pose.pose.orientation.w = out_orien[3]
            out_det.pose.pose.position.x = out_trans[0]
            out_det.pose.pose.position.y = out_trans[1]
            out_det.pose.pose.position.z = out_trans[2]

            out_det.pose.covariance = det.pose.covariance
            fusion_msg.detections.append(out_det)
        self.fusion_pub.publish(fusion_msg)

    def callback_transform(self, msg):
        # print('fusion node received transform msg')
        self.tf.translation = np.array([msg.transform.translation.x,
                                        msg.transform.translation.y,
                                        msg.transform.translation.z])

        self.tf.rotation = Rotation.from_quat([msg.transform.rotation.x,
                                               msg.transform.rotation.y,
                                                   msg.transform.rotation.z,
                                                   msg.transform.rotation.w])


if __name__ == '__main__':
    rospy.init_node('simple_fusion')
    print('******************* fusion node started ...')
    try:
        SimpleFusion()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()