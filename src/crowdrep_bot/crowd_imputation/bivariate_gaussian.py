# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy import cos, sin, exp
from collections import Counter


class BivariateGaussian:
    """
    refs: [1] www.wikipedia.org/wiki/Gaussian_function
          [2] imkbemu.physik.uni-karlsruhe.de/~eisatlas/covariance_ellipses.pdf
    """

    def __init__(self, x0, y0, sigma_x, sigma_y, theta, A=1.):
        """
        :param x0: center of the ellipse along X axis
        :param y0: center of the ellipse along Y axis
        :param sigma_x: variance along X axis
        :param sigma_y: variance along Y axis
        :param theta: rotation angle (degree) Counter-Clock-Wise
        :param A: The function coefficient
        """
        self.x0 = x0
        self.y0 = y0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.theta = theta
        self.A = A
        self.a = cos(theta) ** 2 / (2 * sigma_x ** 2) + sin(theta) ** 2 / (2 * sigma_y ** 2)
        self.b = sin(2 * theta) / (4 * sigma_x ** 2) - sin(2 * theta) / (4 * sigma_y ** 2)
        self.c = sin(theta) ** 2 / (2 * sigma_x ** 2) + cos(theta) ** 2 / (2 * sigma_y ** 2)

    def pdf(self, x, y):
        return self.A * exp(- (self.a * (x - self.x0) ** 2
                               + 2 * self.b * (x - self.x0) * (y - self.y0)
                               + self.c * (y - self.y0) ** 2))


class BivariateGaussianMixtureModel:
    def __init__(self):
        self.components = []
        self.weights = []
        self.targets = []

    def add_component(self, component: BivariateGaussian, weight, target):
        self.components.append(component)
        self.weights.append(weight)
        self.targets = np.append(self.targets, target)

    def classify_kNN(self, x, y, K=-1):
        # FixMe: Debug
        # print('Mixture of %d components:' % len(self.components), self.targets)

        # TODO: check if there is any component?
        if not len(self.components):
            return np.zeros(np.array(x).shape, dtype=int)

        # find the shape of inputs => shape of prob matrix
        probs_shape = [len(self.components)]

        # check if x is a list/matrix
        if np.array(x).shape:
            probs_shape.extend(list(np.array(x).shape))

        # compute the probs for each component
        probabilities = np.zeros(probs_shape)
        class_sum_probs = {}
        for i, component_i in enumerate(self.components):
            probabilities[i] = component_i.pdf(x, y) * self.weights[i]
            if self.targets[i] not in class_sum_probs:
                class_sum_probs[self.targets[i]] = np.zeros_like(probabilities[i])
            class_sum_probs[self.targets[i]] += probabilities[i]

        if K == -1:
            class_probs_keys = np.stack(list(class_sum_probs.keys()))
            class_probs_values = np.stack(list(class_sum_probs.values()))
            return class_probs_keys[np.argmax(class_probs_values, axis=0)]

        else:
            raise Exception('Not implemented: please set K to default value (-1)')

        # Todo
        #  if K != -1:

        probs_copy = probabilities.copy()
        if not np.array(x).shape:  # if given query is a single point
            votes = []
            probs_sorted = []
            for kk in range(K):
                index = np.argmax(probs_copy, axis=0)
                votes.append(self.targets[index])
                probs_sorted.append(probs_copy[index])
                probs_copy[index] = 0

            counter = Counter(votes)
            # m = [[mode(rr[i, j, :]) for i in range(rr.shape[0])] for j in range(rr.shape[1])]
            # x = m[0][0]
            return counter.most_common(1)[0][0]

            # Todo: what if we have more than one mode
            if len(counter) == 1 and counter.most_common(2)[0][0] == counter.most_common(2)[1][0]:
                pass  # Todo

        else:  # if given query is an array
            votes = np.zeros([K] + list(np.array(x).shape), dtype=self.targets.dtype)
            probs_sorted = np.zeros([K] + list(np.array(x).shape), dtype=float)

            for kk in range(K):
                index = np.argmax(probs_copy, axis=0)
                for ii in range(x.shape[0]):
                    for jj in range(x.shape[1]):
                        probs_copy[index[ii, jj], ii, jj] = 0
                        votes[kk, ii, jj] = self.targets[index[ii, jj]]
                        probs_sorted[kk, ii, jj] = probs_copy[index[ii, jj], ii, jj]

            output = np.zeros_like(x, dtype=self.targets.dtype)
            for ii in range(x.shape[0]):
                for jj in range(x.shape[1]):
                    output[ii, jj] = Counter(votes[:, ii, jj]).most_common(1)[0][0]

            return output

    # TODO
    def regression(self, x, y):
        pass


def draw_bgmm(mm: BivariateGaussianMixtureModel, query_x, query_y):
    from crowdrep_bot.crowd_imputation.crowd_communities import CommunityHandler
    import matplotlib.pyplot as plt
    predicted_class_matrix = mm.classify_kNN(query_x, query_y)

    plt.figure("bgmm", figsize=((np.max(query_x)-np.min(query_x)) * 1,
                           (np.max(query_y)-np.min(query_y)) * 1))
    # draw contour for each gaussian component (gray)
    for ii in range(len(mm.components)):
        comp_mean = [mm.components[ii].x0, mm.components[ii].y0]
        comp_cov = np.array([[mm.components[ii].sigma_x, 0], [0, mm.components[ii].sigma_y]]) * 2
        samples = np.random.multivariate_normal(comp_mean, comp_cov, 5000)
        p_k = mm.components[ii].pdf(samples[:, 0], samples[:, 1])
        samples = samples[p_k > 0.4]
        plt.scatter(samples[:, 0], samples[:, 1], c='gray', alpha=0.2)
        plt.scatter(mm.components[ii].x0, mm.components[ii].y0, c='grey')

    plt.scatter(query_x.reshape(-1), query_y.reshape(-1), alpha=0.4,
                c=CommunityHandler().id2color(predicted_class_matrix).reshape(-1))

    plt.xlim([np.min(query_x), np.max(query_x)])
    plt.ylim([np.min(query_y), np.max(query_y)])
    plt.pause(0.001)


# test
if __name__ == "__main__":
    x_min = 0
    x_max = 5
    y_min = 0
    y_max = 2
    sample_2d = lambda: (np.random.rand() * x_max, np.random.rand() * y_max)

    # setup 8 random agents
    n_points = 8
    bgmm = BivariateGaussianMixtureModel()
    points = np.zeros((n_points, 2))
    targets = np.random.choice(['b', 'r'], n_points)  # labels are chosen randomly either 0 or 1
    for i in range(n_points):
        pos = sample_2d()
        vel = [np.random.uniform(0.8, 1.5), np.random.uniform(-0.4, 0.4)]
        points[i] = pos
        theta = np.arctan2(vel[1], vel[0])
        A = 1

        comp_i = BivariateGaussian(pos[0], pos[1], sigma_x=np.linalg.norm(vel)/5, sigma_y=0.1, theta=theta, A=A)
        bgmm.add_component(comp_i, 1, targets[i])

    # test one single point
    test_single_point = sample_2d()
    predicted_class_single_point = bgmm.classify_kNN(test_single_point[0], test_single_point[1])
    print('classification of a single point: ', predicted_class_single_point)

    # test an grid
    resol = 10   # pixel per meter
    xx, yy = np.meshgrid(np.arange(0, x_max, 1/resol), np.arange(y_min, y_max, 1/resol))
    draw_bgmm(bgmm, xx, yy)

