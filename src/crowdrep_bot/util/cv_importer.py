import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)  # in order to import cv2 under python3
import cv2
sys.path.append(ros_path)  # append back in order to import rospy
