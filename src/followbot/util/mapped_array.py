# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import numpy as np


class MappedArray:
    """
    This class is a simple data container, which holds a numpy array,
    together with the mapping between the matrix elements and world coordinates
    """
    def __init__(self, min_x, max_x, min_y, max_y, resolution, n_channels : int=1, dtype=None):
        self.data = np.array(((max_x - min_x)*resolution, (max_y - min_y*resolution), n_channels), dtype)

        x_range = (max_x - min_x)
        y_range = (max_y - min_y)

        # Warning: map/inv_map dont check if the given values are in range or not
        self.map = lambda x, y: ((x - min_x) / x_range, (y - min_y) / y_range)
        self.inv_map = lambda u, v: (u * x_range + min_x, v * y_range + min_y)

    def at(self, x, y):
        u, v = self.map(x, y)
        return self.data[u, v]

