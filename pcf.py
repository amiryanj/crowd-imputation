import os
import numpy as np
from math import sqrt
from sklearn.metrics.pairwise import euclidean_distances

from followbot.temp.parser_eth import ParserETH


def pcf(points, rng=np.arange(0.2, 10, 0.05), sigma=0.3):
    N = len(points)
    dists = euclidean_distances(points)
    dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N-1)
    # return dists

    pcf_lut = dict()  # PCF look-up table
    for rr in rng:
        A_r = np.pi * (rr + 0.2) ** 2 - np.pi * max((rr - 0.2), 0) ** 2

        dists_sqr = np.power(dists_wo_diag - rr, 2)
        dists_exp = np.exp(-dists_sqr / (sigma ** 2)) / (sqrt(np.pi) * sigma)
        pcf_r = np.sum(dists_exp) / N**2 / 2. /A_r # because each dist is repeated twice

        pcf_lut[rr] = pcf_r

    return pcf_lut


def sample_pom(pdf, points):
    return np.random.rand(2)


def dart_with_social_space(points, thresh=0.5):
    for kk in range(100):  # number of tries
        new_point = np.random.rand(2) * np.array([3.5, 7]) + np.array([0.25, 0.5])
        if len(points) == 0:
            return new_point
        new_point_repeated = np.repeat(new_point.reshape((1, 2)), len(points), axis=0)
        dists = np.linalg.norm(np.array(points) - new_point_repeated, axis=1)
        if min(dists) > thresh:
            return new_point

    # FIXME: otherwise ignore the distance and return sth
    return np.random.uniform(2) * np.array([3.5, 7]) + np.array([0.25, 0.5])


def dart_throwing(n, target_pcf, prior_pom):
    error_eps = 0.5
    points = []
    counter = 0
    while len(points) < n:
        new_pt = sample_pom(pdf=prior_pom)
        points_tmp = [points[:], new_pt]
        new_pcf_lut = pcf(points_tmp)
        new_pcf_vals = np.array(sorted(new_pcf_lut.items()))[:, 1]
        err = np.clip(new_pcf_vals - target_pcf, 0, 1000)  # skip values smaller than target values
        max_err = max(err)
        if max_err < error_eps:
            points.append(new_pt)

    return points


def get_rand_n_agents(hist):
    hist_cum = np.cumsum(hist) / np.sum(hist)
    rnd = np.random.random(1)
    for ii, val in enumerate(hist_cum):
        if rnd < val:
            return ii


def test1():
    # FIXME: change Number of points, Range & SIGMA to see the difference
    # when density increases (number of points in the same area) => the peak of the curve moves toward zero
    # range should be big enough!
    # sigma directly controls the smoothness of the curve

    rng = np.arange(0.5, 20, 0.5)
    sigma = 0.2

    # points = [[0, 0], [1, 1], [2, 2], [3, 3]]
    points = np.random.rand(100, 2) * 10

    # pcf_lookup_table = pcf(points, rng, sigma)
    # pcf_values = np.array(sorted(pcf_lookup_table.items()))[:, 1]
    # plt.plot(pcf_values)
    # plt.grid()
    # plt.figure()
    # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()

    parser = ParserETH()
    filename = '/home/cyrus/workspace2/OpenTraj/ETH/seq_hotel/obsmat.txt'
    # filename = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/obsmat.txt'
    parser.load(filename)

    resol = 0.05  # meter
    all_trajs = parser.get_all_points()
    # dim = [5, 5]
    dim = [max(all_trajs[:, 0]) - min(all_trajs[:, 0]),
           max(all_trajs[:, 1]) - min(all_trajs[:, 1])]

    prior_pom = np.ones((int(round(dim[0] / resol)), int(round(dim[1] / resol))), dtype=float)

    parser_data = sorted(parser.t_p_dict.items())
    pcf_cumsum = np.zeros(len(rng), dtype=float)
    n_agents_stats = []
    for item in parser_data:
        t = item[0]
        points_t = np.array(item[1])
        pcf_lut_t = pcf(points_t, rng, sigma)
        pcf_values_t = np.array(sorted(pcf_lut_t.items()))[:, 1]
        pcf_cumsum = pcf_cumsum + pcf_values_t

        n_agents_stats.append(len(points_t))

        # plt.plot(pcf_values_t)
        # plt.grid()
        # plt.show()

    pcf_avg = pcf_cumsum / len(parser_data)

    # draw a random number (number of agents)
    hist, bins = np.histogram(n_agents_stats, bins=range(0, max(n_agents_stats) + 2))
    N_agents = get_rand_n_agents(hist)

    est_points = dart_throwing(N_agents, pcf_avg, prior_pom)

    plt.plot(rng, pcf_avg)
    plt.grid()

    plt.figure()
    _ = plt.hist(n_agents_stats, bins=range(0, 50))  # arguments are passed to np.histogram
    plt.show()


def image_click(event, x, y, flags, param):
    # grab references to the global variables
    global image, points

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 8, (55, 255, 55), -1)
        print(x, y)
        points.append(np.array([x, y])/100)


def crowd_by_click():
    global image, points
    points = []
    # uniformly random crowd
    # points = np.random.rand(20, 2).tolist() * np.array([3.5, 7]) + np.array([0.25, 0.5])
    for kk in range(8):
        new_pnt = dart_with_social_space(points, 1.5)
        points.append(new_pnt)

    if os.path.exists('./temp/points.npy'):
        print('points loaded ..')
        points = np.load('./temp/points.npy').tolist()
    image = cv2.imread('/home/cyrus/Pictures/DSC01887.JPG')
    print(image.dtype)
    image = np.zeros((800, 400, 3), np.uint8)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", image_click)

    cv2.line(image, (10, 100), (10, 700), (0, 0, 250), 3)
    cv2.line(image, (390, 100), (390, 700), (0, 0, 250), 3)

    for pnt in points:
        cv2.circle(image, (int(pnt[0] * 100), int(pnt[1] * 100)), 8, (55, 255, 55), -1)

    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'c' key is pressed, break from the loop
        if key == 27:
            points_np = np.array(points)
            np.save('./temp/points.npy', points_np)
            exit(1)
        elif key == ord('p'):
            rng = np.arange(0.1, 10, 0.01)
            pcf_lut = pcf(points, rng, sigma=0.5)
            # pcf_lut = pcf(points)
            pcf_values = np.array(sorted(pcf_lut.items()))[:, 1]

            plt.figure(figsize=(8, 4), dpi=140, facecolor='w', edgecolor='k')
            plt.subplot(1, 2, 1)
            points_np = np.array(points)
            plt.scatter(points_np[:, 0], points_np[:, 1], label='Agents')
            plt.legend()
            plt.xlim([-2, 6])
            plt.ylim([0, 8])

            # walls
            x1, y1 = [0, 0], [1, 7]
            x2, y2 = [4, 4], [1, 7]
            plt.plot(x1, y1, 'r', marker='')
            plt.plot(x2, y2, 'r', marker='')

            plt.title('Crowd')

            plt.subplot(1, 2, 2)
            plt.plot(rng, pcf_values)
            plt.ylim([0, 0.1])
            plt.grid()
            plt.title('PCF')

            plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    crowd_by_click()





