# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from followbot.crowdsim.pedestrian import Pedestrian


class RobotWorld:
    """
    This class holds an instance of robot beliefs about the world
    """

    def __init__(self):
        # crowds: pedestrians (actual or imaginary)
        self.crowds = {}
