# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import numpy as np
from followbot.crowdsim.drow_utils import load_scan, laserFoV, load_dets
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/n-1r_2015-11-24-18-11-57.bag.csv"
# person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/n-1r_2015-11-24-18-11-57.bag"
# scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/n-1w1p_2015-11-24-17-32-30.bag.csv"
# person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/n-1w1p_2015-11-24-17-32-30.bag"
# scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/r-1r_2015-11-24-18-17-51.bag.csv"
# person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/r-1r_2015-11-24-18-17-51.bag"
# scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/rot-1r_2015-11-24-18-26-38.bag.csv"
# person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/rot-1r_2015-11-24-18-26-38.bag"
# scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/rot_2015-11-26-16-35-03.bag.csv"
# person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/rot_2015-11-26-16-35-03.bag"
# scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/run_2015-11-25-11-18-12-a.bag.csv"
# person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/train/run_2015-11-25-11-18-12-a.bag"
scan_file = "/home/cyrus/workspace2/DROW/DROWv2-data/test/run_t_2015-11-26-11-55-45.bag.csv"
person_annot_file = "/home/cyrus/workspace2/DROW/DROWv2-data/test/run_t_2015-11-26-11-55-45.bag"


scans = load_scan(scan_file)
dets = load_dets(person_annot_file)

_break = False
_pause = False
def press(event):
    global _break, _pause
    if event.key == 'escape':
        _break = True
    if event.key == ' ':
        _pause = not _pause

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)

for ii, scan in enumerate(scans[1]):
    frame_id = scans[0][ii]
    ped_dets = []
    if frame_id in dets[0]:
        wc_dets = dets[1][dets[0].index(frame_id)]
        ped_dets = dets[2][dets[0].index(frame_id)]
        ped_dets = np.array(ped_dets)
        print(ped_dets)
    # print("frame_id = ", frame_id)
    angles = np.linspace(-laserFoV/2, laserFoV/2, len(scan))
    scan_pnts = np.stack([np.cos(angles), np.sin(angles)] * scan).T
    plt.cla()
    ax.set_aspect('equal')
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    plt.scatter(scan_pnts[:, 0], scan_pnts[:, 1], s=1)
    if len(ped_dets):
        plt.scatter(ped_dets[:, 0], ped_dets[:, 1], color='r')
    plt.title("%d" % frame_id)
    plt.grid()
    plt.pause(0.02)
    if _break:
        break
    if _pause:
        plt.pause(1)

