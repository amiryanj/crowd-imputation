# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
import lidar_clustering
import visdom


def norm(x):
    return np.linalg.norm(x)


class PedestrianDetection:
    def __init__(self, range_max_, lidar_resolution):
        self.range_max = range_max_
        self.clustering_threshold = 0.1  # meter
        self.lidar_resolution = lidar_resolution
        self.min_n_pixel = 5
        self.debug_viz = visdom.Visdom()


    def cluster_points_daniel(self, scan, angles):
        scan[scan > self.range_max - 0.1] = np.nan
        clusters, a, b = lidar_clustering.euclidean_clustering(scan, angles, self.clustering_threshold)
        cluster_sizes = lidar_clustering.cluster_sizes(len(scan), clusters)
        return clusters

    def cluster_points(self, scan_pnts_rel):
        clusters = []
        last_x = np.array([self.range_max, self.range_max])
        range_x = np.linalg.norm(scan_pnts_rel, axis=1)

        for ii, x in enumerate(scan_pnts_rel):
            if range_x[ii] < (self.range_max - 0.02):
                if np.linalg.norm(x - last_x) > range_x[ii] * self.lidar_resolution * 2.5:
                    clusters.append([])

                if not np.isnan(x[0]):
                    clusters[-1].append(x)
                last_x = x
        return clusters

    def cluster_range_data(self, range, angles):
        coss = np.cos(angles)
        sins = np.sin(angles)
        xs = np.multiply(coss, range)
        ys = np.multiply(sins, range)
        scan_pnts_rel = np.stack([xs, ys]).T
        clusters = self.cluster_points_daniel(range.astype(np.float32), angles.astype(np.float32))
        return [scan_pnts_rel[c] for c in clusters]
        return self.cluster_points(scan_pnts_rel)

    def detect(self, clusters, sensor_pos):
        detections = []
        walls = []
        for cluster in clusters:
            if len(cluster) < self.min_n_pixel: continue  # noise
            seg_cntr = (cluster[0] + cluster[-1]) * 0.5  # center of seg
            seg_vctr = cluster[0] - cluster[-1]
            seg_prepend = np.array([seg_vctr[1], -seg_vctr[0]])
            if np.dot(seg_prepend, seg_cntr - sensor_pos) < 0:  # find concavity of the segment
                seg_prepend = -seg_prepend

            # TODO: check regression error using polyfit()
            segment_len = norm(seg_vctr)
            if segment_len > 1.5:
                walls.append(cluster)
                continue
            else:
                seg_np = np.array(cluster)
                z, res, _, _, _ = np.polyfit(seg_np[:, 0], seg_np[:, 1], deg=2, full=True)
                if res < 0.001:
                    walls.append(cluster)
                    # self.debug_viz.scatter(cluster, win='detection')
                    continue

            detections.append(seg_cntr + seg_prepend * (1/(np.linalg.norm(seg_prepend)+1e-7) * 0.1))
        return detections, walls
