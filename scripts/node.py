#!/usr/bin/env python

import rospy
from followbot.test_ros import talker
from followbot.followbot_sim_ros import run_sim

print(11)

if __name__ == '__main__':
    rospy.init_node('followbot_sim')
    try:
        run_sim()
        # talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
