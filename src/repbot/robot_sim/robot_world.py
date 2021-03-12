# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from repbot.crowd_sim.pedestrian import Pedestrian


class RobotWorld:
    """
    This class holds an instance of robot hypothesis about the world
    """

    def __init__(self):
        # crowds: pedestrians (both actual or imaginary)
        self.crowds = []  # a list of `Pedestrian`s

        self.walkable_map = []  # would be a constant-matrix that is determined by the scenario maker
        self.occupancy_map = []  # probabilistic occupancy map

    def update(self, lidar_data, tracked_peds):
        # TODO: data should be lidar_data or ...
        for p in tracked_peds:
            pass
        pass
