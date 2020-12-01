# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from dr_spaam.detector import Detector
import dr_spaam.utils.utils as u
default_ckpts = "/home/cyrus/workspace2/DROW/DR-SPAAM-Detector/ckpts/dr_spaam_e40.pth"


class DrSpaamDetector:
    def __init__(self, num_pts, ang_inc_degree=0.5, tracking=False, gpu=True, ckpt=default_ckpts):
        # Detector class wraps up preprocessing, inference, and postprocessing for DR-SPAAM.
        # Checkout the comment in the code for meanings of the parameters.
        self.detector = Detector(
            model_name="DR-SPAAM",
            ckpt_file=ckpt,
            gpu=gpu,
            stride=1,
            tracking=tracking
        )

        # set angular grid (this is only required once)
        ang_inc = np.radians(ang_inc_degree)  # angular increment of the scanner
        # num_pts = 450  # number of points in a scan
        self.detector.set_laser_spec(ang_inc, num_pts)

    def detect(self, scan):
        """
        :param scan: 1D numpy array with positive values
        :return:
        """
        dets_xy, dets_cls, instance_mask = self.detector(scan)  # get detection

        if len(dets_xy):
            dets_xy = np.stack([dets_xy[:, 1], dets_xy[:, 0]]).T

        # confidence threshold
        cls_thresh = 0.2
        cls_mask = dets_cls.squeeze() > cls_thresh
        dets_xy = dets_xy[cls_mask]
        dets_cls = dets_cls[cls_mask]

        return dets_xy, dets_cls

    def get_tracklets(self):
        tracklets = self.detector.get_tracklets()
        return [[np.stack(tr)[:, ::-1] for tr in tracklets[0]], tracklets[1]]


def play_sequence(seq_name, tracking=False):
    # scans
    scans_data = np.genfromtxt(seq_name, delimiter=',')
    scans_t = scans_data[:, 1]
    scans = scans_data[:, 2:]
    scan_phi = u.get_laser_phi()

    # odometry, used only for plotting
    odo_name = seq_name[:-3] + 'odom2'
    odos = np.genfromtxt(odo_name, delimiter=',')
    odos_t = odos[:, 1]
    odos_phi = odos[:, 4]

    # detector
    detector = DrSpaamDetector(num_pts=450, ang_inc_degree=0.5, tracking=tracking, gpu=True, ckpt=default_ckpts)
    # detector.set_laser_spec(angle_inc=np.radians(0.5), num_pts=450)

    # scanner location
    rad_tmp = 0.5 * np.ones(len(scan_phi), dtype=np.float)
    xy_scanner = u.rphi_to_xy(rad_tmp, scan_phi)
    xy_scanner = np.stack(xy_scanner[::-1], axis=1)

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    _break = False
    _pause = False

    def p(event):
        nonlocal _break, _pause
        if event.key == 'escape':
            _break = True
        if event.key == ' ':
            _pause = not _pause


    fig.canvas.mpl_connect('key_press_event', p)

    # video sequence
    odo_idx = 0
    for i in range(len(scans)):
        # for i in range(0, len(scans), 20):
        plt.cla()

        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        # ax.set_title('Frame: %s' % i)
        ax.set_title('Press escape key to exit.')
        ax.axis("off")

        # find matching odometry
        while odo_idx < len(odos_t) - 1 and odos_t[odo_idx] < scans_t[i]:
            odo_idx += 1
        odo_phi = odos_phi[odo_idx]
        odo_rot = np.array([[np.cos(odo_phi), -np.sin(odo_phi)],
                            [np.sin(odo_phi), np.cos(odo_phi)]], dtype=np.float32)

        # plot scanner location
        xy_scanner_rot = np.matmul(xy_scanner, odo_rot.T)
        ax.plot(xy_scanner_rot[:, 0], xy_scanner_rot[:, 1], c='black')
        ax.plot((0, xy_scanner_rot[0, 0] * 1.0), (0, xy_scanner_rot[0, 1] * 1.0), c='black')
        ax.plot((0, xy_scanner_rot[-1, 0] * 1.0), (0, xy_scanner_rot[-1, 1] * 1.0), c='black')

        # plot points
        scan = scans[i]
        scan_y, scan_x = u.rphi_to_xy(scan, scan_phi + odo_phi)
        ax.scatter(scan_x, scan_y, s=1, c='blue')

        # inference
        dets_xy, dets_cls = detector.detect(scan)

        # plot detection
        dets_xy_rot = np.matmul(dets_xy, odo_rot.T)
        cls_thresh = 0.3
        for j in range(len(dets_xy)):
            if dets_cls[j] < cls_thresh:
                continue
            # c = plt.Circle(dets_xy_rot[j], radius=0.5, color='r', fill=False)
            c = plt.Circle(dets_xy_rot[j], radius=0.5, color='r', fill=False, linewidth=2)
            ax.add_artist(c)

        # plot track
        if tracking:
            cls_thresh = 0.2
            tracks, tracks_cls = detector.get_tracklets()
            for t, tc in zip(tracks, tracks_cls):
                if tc >= cls_thresh and len(t) > 1:
                    t_rot = np.matmul(t, odo_rot.T)
                    ax.plot(t_rot[:, 0], t_rot[:, 1], color='g', linewidth=2)

        # plt.savefig('/home/dan/tmp/det_img/frame_%04d.png' % i)

        plt.pause(0.001)

        if _break:
            break
        if _pause:
            plt.pause(1)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # seq_name = '/home/cyrus/workspace2/DROW/DROWv2-data/test/run_t_2015-11-26-11-55-45.bag.csv'
    seq_name = '/home/cyrus/workspace2/DROW/DROWv2-data/train/lunch_2015-11-26-12-04-23.bag.csv'
    tracking = True

    play_sequence(seq_name, tracking=tracking)
