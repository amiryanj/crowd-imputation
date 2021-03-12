# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import copy
from math import ceil

import numpy as np


class MappedArray:
    """
    This class is a simple data container, which holds a numpy array,
    together with the mapping between the matrix elements and world coordinates
    """
    def __init__(self, min_x, max_x, min_y, max_y, resolution, n_channels: int = 1, dtype=None):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.resolution = resolution
        self.n_channels = n_channels

        self.data = np.zeros((int(ceil((max_x - min_x) * resolution)),
                              int(ceil((max_y - min_y) * resolution)), n_channels), dtype).squeeze()

        x_range = (max_x - min_x)
        y_range = (max_y - min_y)
        self.x_edges = np.linspace(min_x, max_x, int(ceil(x_range * resolution)))
        self.y_edges = np.linspace(min_y, max_y, int(ceil(y_range * resolution)))

        # Warning: map/inv_map dont check if the given values are in range or not

        # takes location in world-coord and returns the pixel coord
        self.map = lambda x, y: (int(round((x - min_x) / x_range * self.data.shape[0])),
                                 int(round((y - min_y) / y_range * self.data.shape[1])))
        # takes location in pixel-coord and returns the world-coord
        self.inv_map = lambda u, v: (u / self.data.shape[0] * x_range + min_x,
                                     v / self.data.shape[1] * y_range + min_y)

    def copy_constructor(self):
        return copy.deepcopy(self)

    def meshgrid(self):
        xx, yy = np.meshgrid(np.arange(self.min_x, self.max_x, 1 / self.resolution),
                             np.arange(self.min_y, self.max_y, 1 / self.resolution))
        return xx, yy

    def set(self, pos, val):
        u, v = self.map(pos[0], pos[1])
        if 0 <= u < self.data.shape[0] and 0 <= v < self.data.shape[1]:
            self.data[u, v] = val

    def get(self, pos):
        u, v = self.map(pos[0], pos[1])
        if 0 <= u < self.data.shape[0] and 0 <= v < self.data.shape[1]:
            return self.data[u, v]
        return 0

    def fill(self, val):
        self.data.fill(val)

    def sample_random_pos(self):
        rand_num = np.random.random(1)
        data_cum = np.cumsum(self.data.ravel()) / np.sum(self.data)
        value_bins = np.searchsorted(data_cum, rand_num)
        u_idx, v_idx = np.unravel_index(value_bins, (len(self.x_edges), len(self.y_edges)))
        u_idx, v_idx = u_idx[0], v_idx[0]

        # find local maximum
        padding_width = 8
        padded_data = np.pad(self.data, ((padding_width, padding_width), (padding_width, padding_width)))
        data_roi = padded_data[u_idx:u_idx+2*padding_width+1, v_idx:v_idx+2*padding_width+1]
        du, dv = np.unravel_index(np.argmax(data_roi), data_roi.shape)
        du, dv = du - padding_width, dv - padding_width
        if self.data[u_idx, v_idx] < self.data[u_idx+du, v_idx+dv]:
            u_idx, v_idx = u_idx + du, v_idx + dv

        return self.inv_map(u_idx, v_idx)
