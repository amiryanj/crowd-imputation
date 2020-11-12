import numpy as np


def rotate(p, angle, origin=(0, 0)):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


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


def intersect_circle_line(circle_center, circle_radius, line):
    Q = circle_center  # Centre of circle
    r = circle_radius  # Radius of circle
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


def intersect_circle_lines(circle_center, circle_radius, lines):
    """
    calculates all the intersection center_points between a given circle and a set of line segments

    :param circle_center: the center point of given circle
    :param circle_radius: the radius of given circle
    :param lines: the coordinates of the line segments (Nx2x2)
    :return: three variables
            1. results: list<bool> shows if each line intersects with the circle or not
            2. intersects: list<point> shows the intersection center_points
            3. the normalized distance of intersection center_points from beginning of the line segments
    """
    Q = circle_center  # Centre of circle
    r = circle_radius  # Radius of circle
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
    return results, P1 + np.stack((np.multiply(V[:, 0], t), np.multiply(V[:, 1], t))).transpose(), t


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
        self.center = np.array(center, dtype=np.float)
        self.radius = radius

    def translate(self, d):
        self.center = self.center + d

    def intersect(self, line):
        return intersect_circle_line(self.center, self.radius, line)

    def intersect_many(self, lines):
        circle1_ress, circle1_intpnts, _ = intersect_circle_lines(self.center, self.radius, lines)
        return circle1_ress, circle1_intpnts


class DoubleCircle:  # to represent two legs of a pedestrian
    def __init__(self, center1, center2, radius):
        self.center1 = center1
        self.center2 = center2
        self.radius = radius

    def translate(self, d):
        self.center1 = self.center1 + d
        self.center2 = self.center2 + d

    def intersect(self, line):
        circle1_res, circle1_intpnt = intersect_circle_line(self.center1, self.radius, line)
        circle2_res, circle2_intpnt = intersect_circle_line(self.center2, self.radius, line)
        if circle1_res and np.linalg.norm(circle1_intpnt - line[0]) < np.linalg.norm(circle2_intpnt - line[0]):
            return True, circle1_intpnt
        else:
            return circle2_res, circle2_intpnt

    # @deprecated
    def intersect_many_deprecated(self, lines):
        circle1_ress, circle1_intpnts,_ = intersect_circle_lines(self.center1, self.radius, lines)
        circle2_ress, circle2_intpnts,_ = intersect_circle_lines(self.center2, self.radius, lines)

        final_ress, final_intpnts = circle2_ress.copy(), circle2_intpnts.copy()
        line_0 = np.array(lines[0][0])
        for ii in range(len(lines)):
            if circle1_ress[ii] and ((not circle2_ress[ii]) or (circle2_ress[ii] and
                                                                (np.linalg.norm(circle1_intpnts[ii] - line_0) < np.linalg.norm(circle2_intpnts[ii] - line_0)))):
                final_ress[ii] = True
                final_intpnts[ii] = circle1_intpnts[ii]
        return final_ress, final_intpnts

    def intersect_many(self, lines):
        circle1_ress, circle1_int_pnts, t1 = intersect_circle_lines(self.center1, self.radius, lines)
        circle2_ress, circle2_int_pnts, t2 = intersect_circle_lines(self.center2, self.radius, lines)

        final_int_pnts = circle2_int_pnts.copy()

        final_ress = np.logical_or(circle1_ress, circle2_ress)
        not_int_with_c2 = np.any([t1 < t2, ~circle2_ress])
        int_with_c1_idx = np.all([not_int_with_c2, circle1_ress], axis=0)
        final_int_pnts[int_with_c1_idx] = circle1_int_pnts[int_with_c1_idx]

        return final_ress, final_int_pnts


# test functions
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.xlim([0, 10])
    plt.ylim([0, 10])

    print("Test DoubleCircle intersection with Lidar rays:")
    double_circle = DoubleCircle(np.array([5, 3]), np.array([6, 6]), radius=1)

    circle1 = plt.Circle(double_circle.center1, double_circle.radius, color='r')
    circle2 = plt.Circle(double_circle.center2, double_circle.radius, color='r')
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    # setup rays
    ray_resolution = 10
    ray_length = 10
    ray_angles = np.deg2rad(np.linspace(0, 90, 1 + 90 * ray_resolution))

    # N x 2(bg, end) x 2(x, y)
    rays = np.stack([np.zeros((len(ray_angles), 2)),
                     np.stack([np.cos(ray_angles) * ray_length, np.sin(ray_angles) * ray_length]).T]
                    , axis=2).transpose([0, 2, 1])

    does_intersect, intersect_pnts = double_circle.intersect_many(rays)
    # does_intersect_Dep, intersect_pnts_Dep = double_circle.intersect_many_deprecated(rays)

    for i in range(len(ray_angles)):
        if does_intersect[i]:
            dot = plt.Circle(intersect_pnts[i], 0.05, color='g')
            ax.add_artist(dot)
            line = plt.Line2D([0, intersect_pnts[i, 0]], [0, intersect_pnts[i, 1]], alpha=0.1)
            ax.add_artist(line)
        else:
            line = plt.Line2D([0, rays[i, 1, 0]], [0, rays[i, 1, 1]], alpha=0.3)
            # ax.add_artist(line)


    plt.show()

    dummy = 0
