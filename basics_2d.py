import numpy as np


def intersect_lineseg_lineseg(line1, line2):
    p0_x = line1[0, 0]
    p0_y = line1[0, 1]
    p1_x = line1[1, 0]
    p1_y = line1[1, 1]
    p2_x = line2[0, 0]
    p2_y = line2[0, 1]
    p3_x = line2[1, 0]
    p3_y = line2[1, 1]

    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
    t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

    if 0 <= s <= 1 and 0 <= t <= 1:  # Collision detected
        i_x = p0_x + (t * s1_x)
        i_y = p0_y + (t * s1_y)
        return True, (i_x, i_y)

    return False, [np.nan, np.nan]  # No collision


def intersect_lineseg_linesegs(line1, lines):
    p0_x = line1[0, 0]
    p0_y = line1[0, 1]
    p1_x = line1[1, 0]
    p1_y = line1[1, 1]
    p2_x = lines[:, 0, 0]
    p2_y = lines[:, 0, 1]
    p3_x = lines[:, 1, 0]
    p3_y = lines[:, 1, 1]

    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y + np.finfo(float).eps)
    t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y + np.finfo(float).eps)

    results = np.all([0 <= s, s <= 1, 0 <= t, t <= 1], axis=0)  # Collision detected
    i_x = p0_x + (t * s1_x)
    i_y = p0_y + (t * s1_y)
    return results, np.stack([i_x, i_y]).transpose()


def intersect_line_line(line1, line2):
    # xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    # ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    diff = np.array([line1[0] - line1[1], line2[0] - line2[1]])  # 2(l1, l2) x 2(x, y)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(diff[:, 0], diff[:, 1])
    if div == 0:
        return False, [np.nan, np.nan]

    d = (det(*line1), det(*line2))
    x = det(d, diff[:, 0]) / div
    y = det(d, diff[:, 1]) / div
    return True, [x, y]


def intersect_line_lines(line1, lines):
    ## lines => N x 2(p1,p2) x 2(x,y)
    diff1 = line1[0] - line1[1]  # 2(x, y)
    diffs = lines[:, 0] - lines[:, 1]  # Nx2

    def det(a0, a1, b0, b1):
        if a0.shape == b0.shape:
            return np.multiply(a0, b1) - np.multiply(a1, b0)
        else:
            return a0 * b1 - a1 * b0

    div = det(diff1[0], diff1[1], diffs[:, 0], diffs[:, 1])  # Nx1

    D1 = det(line1[0, 0], line1[0, 1], line1[1, 0], line1[1, 1])  # 1
    D2 = det(lines[:, 0, 0], lines[:, 0, 1], lines[:, 1, 0], lines[:, 1, 1])  # Nx1
    X = np.divide(det(D1, D2, diff1[0], diffs[:, 0]), div)
    Y = np.divide(det(D1, D2, diff1[1], diffs[:, 1]), div)

    results = (div != 0)
    return results, np.stack([X, Y])


def intersect_circle_line(circle, line):
    Q = circle.center  # Centre of circle
    r = circle.radius  # Radius of circle
    P1 = line[0]  # Start of line segment
    V = line[1] - P1  # Vector along line segment

    a = V.dot(V)
    b = 2 * V.dot(P1 - Q)
    c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r ** 2

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return False, None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        return False, None

    t = min(t1, t2)  # closest intersection to starting point of line
    return True, P1 + t * V


def intersect_circle_lines(circle, lines):
    Q = circle.center  # Centre of circle
    r = circle.radius  # Radius of circle
    P1 = lines[:, 0, :]  # Start of line segment
    V = lines[:, 1, :] - P1  # Vector along line segment

    a = np.multiply(V, V).sum(axis=1)
    b = 2 * np.multiply(V, (P1 - Q)).sum(axis=1)
    c = np.multiply(P1, P1).sum(axis=1) + np.dot(Q, Q) - 2 * np.matmul(P1, Q.reshape([2, 1])).squeeze() - r ** 2

    disc = b ** 2 - 4 * a * c
    sqrt_disc = np.sqrt(np.abs(disc))
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    t = np.minimum(t1, t2)  # closest intersection to starting point of line

    results = np.all([disc > 0, 0 <= t1, t1 <= 1, 0 <= t2, t2 <= 1], axis=0)
    return results, P1 + np.stack((np.multiply(V[:, 0], t), np.multiply(V[:, 1], t))).transpose()


class Line:
    def __init__(self, p1, p2):
        self.line = np.array([p1, p2], dtype=np.float)

    def intersect(self, other_line):
        # return intersect_line_line(self.line, other_line)
        return intersect_lineseg_lineseg(self.line, other_line)

    def intersect_many(self, other_lines):
        # return intersect_line_lines(self.line, other_lines)
        return intersect_lineseg_linesegs(self.line, other_lines)


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, line):
        return intersect_circle_line(self, line)

    def intersect_many(self, lines):
        return intersect_circle_lines(self, lines)
