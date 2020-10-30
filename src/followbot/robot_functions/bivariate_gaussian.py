# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
from numpy import cos, sin, exp
from collections import Counter
from scipy.stats import mode


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
        # TODO: check if there is any component?

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
            class_probs_keys = np.stack(class_sum_probs.keys())
            class_probs_values = np.stack(class_sum_probs.values())
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


# test
if __name__ == "__main__":
    # rr = np.random.randint(low=1, high=10, size=(10, 20, 3))
    # m = [[mode(rr[i, j, :]) for i in range(rr.shape[0])] for j in range(rr.shape[1])]
    # x = m[0][0]
    # exit(1)

    import matplotlib.pyplot as plt
    x_min = 0
    x_max = 5
    y_min = 0
    y_max = 2
    sample_2d = lambda: (np.random.rand() * x_max, np.random.rand() * y_max)

    # setup 8 random agents
    n_center_points = 8
    bgm = BivariateGaussianMixtureModel()
    center_points = np.zeros((n_center_points, 2))
    targets = np.random.choice(['b', 'r'], n_center_points)  # labels are chosen randomly either 0 or 1
    for i in range(n_center_points):
        pos = sample_2d()
        vel = [np.random.uniform(0.8, 1.5), np.random.uniform(-0.2, 0.2)]
        center_points[i] = pos
        theta = np.arctan2(vel[1], vel[0])
        A = 1

        comp_i = BivariateGaussian(pos[0], pos[1], sigma_x=np.linalg.norm(vel)/5, sigma_y=0.1, theta=theta, A=A)
        bgm.add_component(comp_i, 1, targets[i])

    # test one single point
    test_single_point = sample_2d()
    predicted_class_single_point = bgm.classify_kNN(test_single_point[0], test_single_point[1])
    print('classification of a single point: ', predicted_class_single_point)

    # test an grid
    resol = 10   # pixel per meter
    xx, yy = np.meshgrid(np.arange(0, x_max, 1/resol), np.arange(y_min, y_max, 1/resol))
    predicted_class_matrix = bgm.classify_kNN(xx, yy)

    # draw contour for each gaussian component (gray)
    for ii in range(len(bgm.components)):
        samples = []
        for kk in range(5000):
            s_k = sample_2d()
            samples.append(s_k)
        samples = np.stack(samples)
        p_k = bgm.components[ii].pdf(samples[:, 0], samples[:, 1])
        samples = samples[p_k > 0.5]
        plt.scatter(samples[:, 0], samples[:, 1], c='gray', alpha=0.2)
    plt.scatter(center_points[:, 0], center_points[:, 1], c=targets)

    plt.plot(test_single_point[0], test_single_point[1], c='g')

    plt.scatter(xx.reshape(-1), yy.reshape(-1), c=predicted_class_matrix.reshape(-1), alpha=0.4)

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.show()

