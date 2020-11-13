# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np

def norm(x):
    return np.linalg.norm(x)


class PedestrianDetection:
    def __init__(self, range_max_, lidar_resolution):
        self.range_max = range_max_
        self.segmentation_threshold = 0.2
        self.lidar_resolution = lidar_resolution
        self.min_n_pixel = 5

    def cluster_points(self, scan_pnts_rel):
        segments = []
        last_x = np.array([self.range_max, self.range_max])
        for x in scan_pnts_rel:
            range_x = np.linalg.norm(x)
            if range_x < (self.range_max - 0.1):
                if np.linalg.norm(x - last_x) > range_x * self.lidar_resolution * 2.5:
                    segments.append([])

                if not np.isnan(x[0]):
                    segments[-1].append(x)
                last_x = x
        return segments

    def cluster_range_data(self, range, angles):
        coss = np.cos(angles)
        sins = np.sin(angles)
        xs = np.multiply(coss, range)
        ys = np.multiply(sins, range)
        scan_pnts_rel = np.stack([xs, ys]).T
        return self.cluster_points(scan_pnts_rel)

    def detect(self, segments, sensor_pos):
        detections = []
        walls = []
        for seg in segments:
            if len(seg) < self.min_n_pixel: continue  # noise
            seg_cntr = (seg[0] + seg[-1]) * 0.5  # center of seg
            seg_vctr = seg[0] - seg[-1]
            seg_prepend = np.array([seg_vctr[1], -seg_vctr[0]])
            if np.dot(seg_prepend, seg_cntr - sensor_pos) < 0:  # find concavity of the segment
                seg_prepend = -seg_prepend

            # TODO: check regression error using polyfit()
            segment_len = norm(seg_vctr)
            if segment_len > 1.5:
                walls.append(seg)
                continue
            else:
                seg_np = np.array(seg)
                z, res, _, _, _ = np.polyfit(seg_np[:, 0], seg_np[:, 1], deg=2, full=True)
                if res < 0.001:
                    walls.append(seg)
                    continue

            detections.append(seg_cntr + seg_prepend * (1/(np.linalg.norm(seg_prepend)+1e-7) * 0.1))
        return detections, walls
