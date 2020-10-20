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

        self.walkable = []  # would be a constant-matrix that is determined by the scenario maker
        self.POM = []  # probabilistic occupancy map
        self.mapping_to_grid = []

    def update(self, lidar_data, tracked_peds):
        # TODO: data should be lidar_data or ...
        for p in tracked_peds:
            pass
        pass
