import numpy as np
from filterpy.kalman import KalmanFilter


class Track:
    def __init__(self, dt=0.01):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.zeros((4, 1), dtype=float)  # position & velocity

        # FIXME: Define the state transition matrix:
        self.kf.F = np.eye(4, dtype=float)
        self.kf.F[0, 2], self.kf.F[1, 3] = dt, dt

        # FIXME: Define the measurement function:
        self.kf.H = np.zeros((2, 4), dtype=float)
        self.kf.H[0, 0], self.kf.H[1, 1] = 1, 1

        self.kf.P *= 1
        # process noise
        self.kf.Q = [[dt ** 5 / 20, 0, dt ** 4 / 8, 0],
                     [0, dt ** 5 / 20, 0, dt ** 4 / 8],
                     [dt ** 4 / 8, 0, dt ** 3 / 3, 0],
                     [0, dt ** 4 / 8, 0, dt ** 3 / 3]]

        # measurement noise
        self.kf.R *= np.eye(2, dtype=float) * 1
        self.tentative = True
        self.recent_detections = []
        self.coasted = False  # decayed

    def init(self, value):
        if len(value) == 4:
            self.kf.x = np.array(value)
        else:
            self.kf.x = np.array([value[0], value[1], 0, 0])
        self.tentative = True

    def update_recent_detections(self, val):
        self.recent_detections.append(val)
        if len(self.recent_detections) > 10:
            self.recent_detections.pop(0)
            if sum(self.recent_detections) < 5:
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



