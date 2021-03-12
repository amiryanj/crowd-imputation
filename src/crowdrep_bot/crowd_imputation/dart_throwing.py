import numpy as np
import matplotlib.pyplot as plt


# class Rect:
#     def __init__(self, left, top, width, height):
#         self.left = left
#         self.top = top
#         self.width = width
#         self.height = height
#
#     def contains(self, x, y):
#         return self.left < x < self.left + self.width and self.top < y < self.top + self.height


class DartThrowing:
    """
        # summary: Dart Throwing algorithm
                   Construct a set of particles, so that each pair is not closer than a min_dist threshold
        # CrowdBox  : width, height of the crowd,
        # minDist   : dist minimal / or a distribution
        # K         : Choose up to K points around each reference point as candidates for a new sample point
    """

    def __init__(self, crowdRect, pcf_vals, pcf_radius_vals, k=30):

        self.CrowdRect = crowdRect
        self.PcfAccum = np.add.accumulate(pcf_vals / sum(pcf_vals))  # Todo: a distribution
        self.PcfRadia = pcf_radius_vals
        self.k = k

        # Cell side length
        # self.a = minDist / np.sqrt(2)  # Todo: how to calc this?
        # Number of cells in the x- and y-directions of the grid
        # self.nx, self.ny = int(self.CrowdSize[0] / self.a) + 1, int(self.CrowdSize[0] / self.a) + 1
        self.samples = []   # Todo: And this?
        # self.cells = {}
        # self.init_cells_coord()

    # # def init_cells_coord(self):
    #     # A list of coordinates in the grid of cells
    #     coords_list = [(ix, iy) for ix in range(self.nx) for iy in range(self.ny)]
    #
    #     # Initilalize the dictionary of cells: each key is a cell's coordinates, the
    #     # corresponding value is the index of that cell's point's coordinates in the
    #     # samples list (or None if the cell is empty).
    #     self.cells = {coords: None for coords in coords_list}

    # # Get the coordinates of the cell that pt = (x,y) falls in
    # def get_cell_coords(self, pt):
    #
    #     return int(pt[0] // self.a), int(pt[1] // self.a)

    # def get_neighbours(self, coords):
    #     """Return the indexes of points in cells neighbouring cell at coords.
    #
    #             For the cell at coords = (x,y), return the indexes of points in the cells
    #             with neighbouring coordinates illustrated below: ie those cells that could
    #             contain points closer than r.
    #
    #                                              ooo
    #                                             ooooo
    #                                             ooXoo
    #                                             ooooo
    #                                              ooo
    #
    #             """
    #
    #     dxdy = [(-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
    #             (-2, 0), (-1, 0), (1, 0), (2, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
    #             (-1, 2), (0, 2), (1, 2), (0, 0)]
    #     neighbours = []
    #     for dx, dy in dxdy:
    #         neighbour_coords = coords[0] + dx, coords[1] + dy
    #         if not (0 <= neighbour_coords[0] < self.nx and
    #                 0 <= neighbour_coords[1] < self.ny):
    #             # We're off the grid: no neighbours here.
    #             continue
    #         neighbour_cell = self.cells[neighbour_coords]
    #         if neighbour_cell is not None:
    #             # This cell is occupied: store this index of the contained point.
    #             neighbours.append(neighbour_cell)
    #     return neighbours

    # def point_valid(self, pt):
    #     """Is pt a valid point to emit as a sample?
    #
    #     It must be no closer than r from any other point: check the cells in its
    #     immediate neighbourhood.
    #
    #     """
    #
    #     cell_coords = self.get_cell_coords(pt)
    #     for idx in self.get_neighbours(cell_coords):
    #         nearby_pt = self.samples[idx]
    #         # Squared distance between or candidate point, pt, and this nearby_pt.
    #         distance2 = (nearby_pt[0] - pt[0]) ** 2 + (nearby_pt[1] - pt[1]) ** 2
    #         if distance2 < self.MinDistPcf ** 2:
    #             # The points are too close, so pt is not a candidate.
    #             return False
    #     # All points tested: if we're here, pt is valid
    #     return True

    def check_sample_points(self, pt, min_dist):
        for p1 in self.samples:
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (p1[0] - pt[0]) ** 2 + (p1[1] - pt[1]) ** 2
            if distance2 < min_dist ** 2:
                # The points are too close, so pt is not a candidate.
                return False

        # All points tested: if we're here, pt is valid
        return True

    def get_point(self, refpt, min_dist):
        """Try to find a candidate point relative to refpt to emit in the sample.

        We draw up to K points from the annulus of inner radius r, outer radius 2r
        around the reference point, refpt. If none of them are suitable (because
        they're too close to existing points in the sample), return False.
        Otherwise, return the pt.

        """
        i = 0
        while i < self.k:
            rho, theta = min_dist, np.random.uniform(0, 2 * np.pi)  # FixMe
            pt = [refpt[0] + rho * np.cos(theta), refpt[1] + rho * np.sin(theta)]
            if not self.CrowdRect.collidepoint(pt[0], pt[1]):
            # if not (0 <= pt[0] < self.AreaSize[0] and 0 <= pt[1] < self.AreaSize[1]):
                # This point falls outside the domain, so try again.
                continue
            if self.check_sample_points(pt, min_dist):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def get_random_mindist(self):
        urand = np.random.rand()
        inds = np.where(self.PcfAccum > urand)
        return self.PcfRadia[inds[0][0]]


    # Pick a random point to start with.
    def create_samples(self, init_pts):
        self.samples = init_pts.tolist()  # FixMe: to get input points
        # Our first sample is indexed at 0 in the samples list...
        # for pt in init_pts:
        #     self.cells[self.get_cell_coords(pt)] = 0
        # ... and it is active, in the sense that we're going to look for more points
        # in its neighbourhood.
        active = list(range(len(self.samples)))

        nsamples = len(self.samples)
        # As long as there are points in the active list, keep trying to find samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = np.random.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.

            min_dist = self.get_random_mindist()
            pt = self.get_point(refpt, min_dist)
            if pt:
                # Point pt is valid: add it to the samples list and mark it as active
                self.samples.append(pt)
                nsamples += 1
                active.append(len(self.samples) - 1)
            else:
                # We had to give up looking for valid points near refpt, so remove it
                # from the list of "active" points.
                active.remove(idx)

        self.samples = np.array(self.samples)
        return self.samples

    def plot_samples(self):
        plt.scatter(self.samples[:, 0], self.samples[:, 1], color='r', s=250, alpha=0.6, lw=0)
        print(self.samples.shape[0])
        plt.xlim(0, self.AreaSize[0])
        plt.ylim(0, self.AreaSize[1])
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    crowdSize = (10, 10)  # size of the crowd
    minDist = 0.6  # minimal distance between agents
    k = 30  # number of attempt when trying to add a new agent
    Distrib = DartThrowing(crowdSize=crowdSize, minDist=minDist, k=k)
    init_pt = [np.random.uniform(0, self.CrowdSize[0]), np.random.uniform(0, self.CrowdSize[1])]
    samples = Distrib.create_samples(plot=True)
