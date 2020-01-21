import numpy as np
# from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter


class ObjectTracker:
    def __init__(self, dt=0.01):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.zeros((4, 1), dtype=float)  # position & velocity

        # FIXME: Define the state transition matrix:
        self.kf.F = np.eye(4, dtype=float)
        self.kf.F[0, 2], self.kf.F[1, 3] = dt, dt

        # FIXME: Define the measurement function:
        self.kf.H = np.zeros((2, 4), dtype=float)
        self.kf.H[0, 0], self.kf.H[1, 1] = 1, 1

        self.kf.P *= 1000
        # process noise
        self.kf.Q = np.array([[dt ** 5 / 20, 0, dt ** 4 / 8, 0],
                              [0, dt ** 5 / 20, 0, dt ** 4 / 8],
                              [dt ** 4 / 8, 0, dt ** 3 / 3, 0],
                              [0, dt ** 4 / 8, 0, dt ** 3 / 3]]) * 100

        # measurement noise
        self.kf.R *= np.eye(2, dtype=float) * 0.1
        self.tentative = True
        self.recent_detections_bool = []
        self.recent_detections = []
        self.coasted = False  # decayed

    def init(self, measurement):
        self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
        self.tentative = True

    def update_recent_detections(self, val):
        self.recent_detections_bool.append(val)
        if len(self.recent_detections_bool) > 10:
            self.recent_detections_bool.pop(0)
            if sum(self.recent_detections_bool) < 5:
                self.coasted = True

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)
        self.tentative = True

    def position(self):
        return self.kf.x[:2]

    def velocity(self):
        return self.kf.x[2:]


class MultiObjectTracking:
    def __init__(self, max_dist):
        self.max_dist = max_dist
        self.segmentation_threshold = 0.5
        self.detection_matching_threshold = 1.5
        self.tracks = []

    def segment(self, scan, sensor_pos):
        segments = []
        last_x = np.array([1000, 1000])
        for x in scan:
            if np.linalg.norm(x - sensor_pos) < (self.max_dist - 1):
                if np.linalg.norm(x - last_x) > self.segmentation_threshold:
                    segments.append([])

                if np.isnan(x[0]):
                    dummy_test = 1
                else:
                    segments[-1].append(x)
            last_x = x
        return segments

    def detect(self, segments, sensor_pos):
        detections = []
        walls = []
        for seg in segments:
            if len(seg) < 2: continue
            seg_np = np.array(seg)

            p = (seg[0] + seg[-1]) * 0.5
            q = seg[0] - seg[-1]
            q_prepend = np.array([q[1], -q[0]])

            if np.linalg.norm(q) > 1:
                # TODO: check line by using polyfit
                # z, res, _, _, _ = np.polyfit(seg_np[:, 0], seg_np[:, 1], deg=2, full=True)
                # if res < 0.001:
                #     walls.append(seg)
                walls.append(seg)
                continue

            if np.dot(q_prepend, p - sensor_pos) < 0:
                q_prepend = -q_prepend
            detections.append(p + q_prepend * (1/(np.linalg.norm(q_prepend)+1e-7) * 0.1))
        return detections, walls

    def track(self, detections):
        # TODO| from Mathworks: www.mathworks.com/help/driving/examples/multiple-object-tracking-tutorial.html
        # 1. Assigning detections to tracks.
        # 2. Initializing new tracks based on unassigned detections. All tracks are initialized as 'Tentative',
        #    accounting for the possibility that they resulted from a false detection.
        # 3. Confirming tracks if they have more than M assigned detections in N frames.
        # 4. Updating existing tracks based on assigned detections.
        # 5. Coasting (predicting) existing unassigned tracks.
        # 6. Deleting tracks if they have remained unassigned (coasted) for too long.

        unassigned_detections = detections.copy()
        for track_i in self.tracks:
            if track_i.coasted: continue

            # print('before predict = ', track_i.kf.x)
            track_i.predict()
            # print('after predict =', track_i.kf.x)
            if len(unassigned_detections) == 0:
                break
            # find the closest to the prediction
            dists = np.stack(unassigned_detections) - track_i.kf.x[:2]
            dists = np.linalg.norm(dists, axis=1)
            min_ind = np.argmin(dists, axis=0)
            detection_is_matched = dists[min_ind] < self.detection_matching_threshold
            track_i.update_recent_detections(detection_is_matched)
            if detection_is_matched:
                # FIXME: assign detection to track_i
                track_i.update(unassigned_detections[min_ind])

                # if len(unassigned_detections) == len(detections):
                #     print('*** dist = ', np.linalg.norm(unassigned_detections[min_ind] - track_i.kf.x[:2]))
                del unassigned_detections[min_ind]

            track_i.recent_detections.append(track_i.kf.x[:2])

        for ii, det in enumerate(unassigned_detections):
            # print('create new track')
            track_i = ObjectTracker(0.1)
            track_i.init(det)
            track_i.tentative = True
            track_i.recent_detections_bool.append(True)
            self.tracks.append(track_i)

        # for track_i in self.tracks:
        #     track_i.update()

        return self.tracks


if __name__ == '__main__':
    tracker = ObjectTracker(dt=0.1)
    tracker.init([0, 0])

    for i in range(10):
        tracker.kf.predict()
        x, y = i * 0.1, i * 0.1
        tracker.update([x, y])
        print(tracker.kf.x)
