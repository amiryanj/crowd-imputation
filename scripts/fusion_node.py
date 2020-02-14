#!/usr/bin/env python
import numpy as np
import rospy
from frame_msgs.msg import DetectedPerson, DetectedPersons, TrackedPersons
from geometry_msgs.msg import TransformStamped
# import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(self):
        self.translation = None
        self.rotation = None


class SimpleFusion:
    def __init__(self):
        detection_topic = rospy.get_param("~subscriber/detections0/topic")
        transform_topic = rospy.get_param("~subscriber/transform")
        fusion_topic = rospy.get_param("~publisher/detections/topic")
        queue_size = rospy.get_param("~publisher/detections/queue_size")
        latch = rospy.get_param("~publisher/detections/latch")

        self.detection_sub = rospy.Subscriber(detection_topic, DetectedPersons, self.callback_detection)
        self.transform_sub = rospy.Subscriber(transform_topic, TransformStamped, self.callback_transform)
        self.fusion_pub = rospy.Publisher(fusion_topic, DetectedPersons, queue_size=queue_size, latch=latch)

        self.tf = Transform()

    def callback_detection(self, msg):
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

            det_orien = det.pos.pos.orientation
            det_orien = Rotation.from_quat([det_orien.x, det_orien.y, det_orien.z, det_orien.w])
            det_trans = det.pos.pos.position
            det_trans = np.array([det_trans.x, det_trans.y, det_trans.z])

            out_orien = det_orien.__mul__(self.tf.rotation).as_quat()
            out_trans = self.tf.translation + self.tf.rotation.apply(det_trans)

            out_det.pos.pos.orientation.x = out_orien[0]
            out_det.pos.pos.orientation.y = out_orien[1]
            out_det.pos.pos.orientation.z = out_orien[2]
            out_det.pos.pos.orientation.w = out_orien[3]
            out_det.pos.pos.position.x = out_trans[0]
            out_det.pos.pos.position.y = out_trans[1]
            out_det.pos.pos.position.z = out_trans[2]

            out_det.pos.covariance = det.pos.covariance
            fusion_msg.detections.append(out_det)


def callback_transform(self, msg):
        self.tf.translation = np.array([msg.transform.translation.x,
                                        msg.transform.translation.y,
                                        msg.transform.translation.z])

        self.tf.rotation = Rotation.from_quat([msg.transform.rotation.x,
                                               msg.transform.rotation.y,
                                               msg.transform.rotation.z,
                                               msg.transform.rotation.w])


