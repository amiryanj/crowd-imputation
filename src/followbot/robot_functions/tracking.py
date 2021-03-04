from abc import abstractmethod

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


class TrackBase:
    def __init__(self, id):
        """
        :param id: Track id should be unique
        """
        self.id = id
        self.last_obsv_timestamp = -1
        self.state = np.array([0, 0, 0, 0])

    @abstractmethod
    def predict(self, t):
        pass

    @abstractmethod
    def update(self, detection, t):
        self.state[2:4] = (detection - self.state[:2]) / (t - self.last_obsv_timestamp)  # update velocity
        self.state[:2] = detection      # update position
        self.last_obsv_timestamp = t

    def get_position(self):
        return self.state[:2]

    def get_velocity(self):
        return self.state[2:4]


class KalmanBasedTrack(TrackBase):
    def __init__(self, id, dt):
        self.id = id
        # ======== Kalman Filter ================
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.zeros((4, 1), dtype=float)  # position & velocity

        # state transition matrix:
        self.kf.F = np.eye(4, dtype=float)
        self.kf.F[0, 2], self.kf.F[1, 3] = dt, dt

        # measurement function:
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
        # =======================================

        self.tentative = True  # not confirmed
        self.recent_assigned_detections = []  # this variable remembers assignment/un-assignment during last 10 frames
        self.recent_detections = []  # Todo: should be a mapping from time
        self.coasted = False  # decayed
        self.last_obsv_timestamp = -1

    def init_track_state(self, measurement):
        self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
        self.tentative = True

    def update_recent_assignments(self, assigned_det: np.ndarray):
        self.recent_assigned_detections.append(assigned_det)
        if len(self.recent_assigned_detections) > 6:
            self.recent_assigned_detections.pop(0)
            if sum([not np.isnan(det[0]) for det in self.recent_assigned_detections]) < 3:
                self.coasted = True

    def predict(self, t):
        self.kf.predict()

    def update(self, detection, t):
        self.kf.update(detection)
        self.tentative = True

    def get_position(self):
        return self.kf.x[:2]

    def get_velocity(self):
        return self.kf.x[2:]


class MultiObjectTracking:
    """
        ref: https://mathworks.com/help/driving/ug/multiple-object-tracking-tutorial.html
        ***************************************
        1. Assigning detections to tracks.
        2. Initializing new tracks based on unassigned detections. All tracks are initialized as 'Tentative',
           accounting for the possibility that they resulted from a false detection.
        3. Confirming tracks if they have more than M assigned detections in N frames.
        4. Updating existing tracks based on assigned detections.
        5. Coasting (predicting) existing unassigned tracks.
        6. Deleting tracks if they have remained unassigned (coasted) for too long.
        ***************************************
    """
    def __init__(self, sensor_fps):
        self._tracks = []
        self._track_id_counter = 0  # track ids start from 1
        self.last_timestamp = 0
        self.sensor_dt = 1. / sensor_fps

        # parameters
        # =======================================
        # How far detections may fall from tracks
        self.assignmentThreshold = 1.0

        # How long a track is maintained before deletion.
        # a reasonable value is about 0.75 seconds (7.5 frames).
        # TODO
        # self.deletionThreshold = 0.75

        # The parameters controlling track confirmation.
        # A track is initialized with every unassigned detection. Some of these detections might be false, so initially,
        # all tracks are 'Tentative'. To confirm a track, it has to be detected at least M out of N frames.
        # The choice of M and N depends on the visibility of the objects. By default, we assume a visibility
        # of 6 out of 10 frames.
        # TODO
        # self.confirmationThreshold = [6, 10]
        # -----------------------------------------

    def _get_new_track_id_(self):
        self._track_id_counter += 1
        return self._track_id_counter

    def track(self, detections, t: float):
        """
        :param detections: new detections provided by object detector
        :param t: new timestamp
        :return: _tracks
        """

        for track_i in self._tracks:
            # 1) ****** Predict the state of each track ******
            track_i.predict(t)  # todo: Warning: remove coasted tracks

        active_tracks = [tr for tr in self._tracks if not tr.coasted]
        if len(active_tracks) == 0 and len(detections) > 0:
            track_0 = KalmanBasedTrack(self._get_new_track_id_(), self.sensor_dt)
            track_0.init_track_state(detections[0])
            track_0.predict(t)
            # to assign this track to first detection
            self._tracks.append(track_0)
            active_tracks = self._tracks

        tracks_prior_positions = np.array([tr.kf.x_prior[:2] for tr in active_tracks]).reshape((-1, 2))
        list_of_unassigned_tracks_idx = list(range(len(active_tracks)))


        # probability matrix between each new detection and each track
        if len(detections):
            dist_d2t = pairwise_distances(detections, tracks_prior_positions, metric="euclidean")

            # Hungarian algorithm, also known as the Munkres or Kuhn-Munkres algorithm.
            # https://brc2.com/the-algorithm-workshop/
            row_ind, col_ind = linear_sum_assignment(dist_d2t)
            row_ind = row_ind.tolist()
            col_ind = col_ind.tolist()
            for ii, det in enumerate(detections):
                if ii in row_ind and dist_d2t[ii, col_ind[row_ind.index(ii)]] < self.assignmentThreshold:
                    # assign detection to track
                    track_idx = col_ind[row_ind.index(ii)]
                    track_i = active_tracks[track_idx]  # Fixme: of course it points to the right track. no?
                    list_of_unassigned_tracks_idx.remove(track_idx)
                    track_i.recent_detections.append(track_i.kf.x[:2])
                else:
                    # for detections without any track in a `epsilon` distance: create new track
                    track_i = KalmanBasedTrack(self._get_new_track_id_(), self.sensor_dt)
                    self._tracks.append(track_i)
                    track_i.init_track_state(det)
                track_i.update_recent_assignments(det)
                track_i.update(det, t)

        # Todo: tracks that are not assigned with any detections
        unassigned_tracks = [active_tracks[i] for i in list_of_unassigned_tracks_idx]
        for track_i in unassigned_tracks:
            track_i.update_recent_assignments(np.array([np.nan, np.nan]))

        # unassigned_detections = detections.copy()

        # for track_i in self._tracks:
        #     if track_i.coasted: continue
        #     # print('after predict =', track_i.kf.x)
        #     if len(unassigned_detections) == 0:
        #         break
        #     # find the closest to the prediction
        #     dists = np.stack(unassigned_detections) - track_i.kf.x[:2]
        #     dists = np.linalg.norm(dists, axis=1)
        #     min_ind = np.argmin(dists, axis=0)
        #     detection_is_matched = dists[min_ind] < self.assignmentThreshold
        #     if detection_is_matched:
        #         # assign detection to track_i
        #         track_i.update(unassigned_detections[min_ind])
        #         del unassigned_detections[min_ind]
        #     track_i.update_recent_detections(detection_is_matched)
        #     track_i.recent_detections.append(track_i.kf.x[:2])

        # for ii, det in enumerate(unassigned_detections):
        #     # print('create new track')
        #     track_i = KalmanBasedTrack(0.1)
        #     track_i.initialize_track_state(det)
        #     track_i.tentative = True
        #     track_i.recent_detections_bool.append(True)
        #     self._tracks.append(track_i)

        self.last_timestamp = t
        return self._tracks


# @deprecated
class BlindObjectTracking:
    """
            ref: https://mathworks.com/help/driving/ug/multiple-object-tracking-tutorial.html
            ***************************************
            1. Assigning detections to tracks.
            2. Initializing new tracks based on unassigned detections. All tracks are initialized as 'Tentative',
               accounting for the possibility that they resulted from a false detection.
            3. Confirming tracks if they have more than M assigned detections in N frames.
            4. Updating existing tracks based on assigned detections.
            5. Coasting (predicting) existing unassigned tracks.
            6. Deleting tracks if they have remained unassigned (coasted) for too long.
            ***************************************
        """

    def __init__(self):
        self._tracks = []

        # parameters
        # =======================================
        # How far detections may fall from tracks
        self.assignmentThreshold = 1.0

        # How long a track is maintained before deletion.
        # a reasonable value is about 0.75 seconds (7.5 frames).
        self.deletionThreshold = 0.75

        # The parameters controlling track confirmation.
        # A track is initialized with every unassigned detection. Some of these detections might be false, so initially,
        # all tracks are 'Tentative'. To confirm a track, it has to be detected at least M out of N frames.
        # The choice of M and N depends on the visibility of the objects. By default, we assume a visibility
        # of 6 out of 10 frames.
        self.confirmationThreshold = [6, 10]
        # -----------------------------------------

    def track(self, detections, t: float):
        """
        :param detections: new detections provided by object detector
        :param t: new timestamp
        :return: _tracks
        """

        for track_i in self._tracks:
            # 1) ****** Predict the state of each track ******
            # print('before predict = ', track_i.kf.x)
            track_i.predict(t)

        active_tracks = [tr for tr in self._tracks if not tr.coasted]
        if len(active_tracks) == 0 and len(detections) > 0:
            track_0 = KalmanBasedTrack(dt=0.1)
            track_0.init_track_state(detections[0])
            track_0.predict(t)
            # to assign this track to first detection
            self._tracks.append(track_0)
            active_tracks = self._tracks

        tracks_prior_positions = np.array([tr.kf.x_prior[:2] for tr in active_tracks]).reshape((-1, 2))
        list_of_unassigned_tracks_idx = list(range(len(active_tracks)))

        # probability matrix between each new detection and each track
        dist_d2t = pairwise_distances(detections, tracks_prior_positions, metric="euclidean")

        # Hungarian algorithm, also known as the Munkres or Kuhn-Munkres algorithm.
        # https://brc2.com/the-algorithm-workshop/
        row_ind, col_ind = linear_sum_assignment(dist_d2t)
        row_ind = row_ind.tolist()
        col_ind = col_ind.tolist()
        for ii, det in enumerate(detections):
            if ii in row_ind and dist_d2t[ii, col_ind[row_ind.index(ii)]] < self.assignmentThreshold:
                # assign detection to track
                track_idx = col_ind[row_ind.index(ii)]
                track_i = active_tracks[track_idx]  # Fixme: of course it points to the right track. no?
                list_of_unassigned_tracks_idx.remove(track_idx)
                track_i.recent_detections.append(track_i.kf.x[:2])
            else:
                # for detections without any track in a `epsilon` distance: create new track
                track_i = KalmanBasedTrack(dt=0.1)
                self._tracks.append(track_i)
                track_i.init_track_state(det)
            track_i.update(det, t)
            track_i.update_recent_assignments(det)

        # Todo: tracks that are not assigned with any detections
        unassigned_tracks = [active_tracks[i] for i in list_of_unassigned_tracks_idx]
        for track_i in unassigned_tracks:
            track_i.update_recent_assignments(np.array([np.nan, np.nan]))

        return self._tracks


if __name__ == '__main__':
    tracker = KalmanBasedTrack(0, dt=0.1)
    tracker.init_track_state([0, 0])

    for i in range(10):
        tracker.kf.predict()
        x, y = i * 0.1, i * 0.1
        tracker.update([x, y], i/10.)
        print(tracker.get_position(), tracker.get_velocity())
