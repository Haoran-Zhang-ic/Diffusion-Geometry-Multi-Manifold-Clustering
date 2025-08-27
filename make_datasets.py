import numpy as np
from sklearn.datasets import make_swiss_roll


def three_planes(n_points=1000, random_state=None):
    rng = np.random.default_rng(random_state)

    def rotation_matrix_y(theta_deg):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    x = rng.uniform(-1, 1, int(n_points / 3))
    y_ = rng.uniform(-1, 1, int(n_points / 3))
    z = np.zeros(int(n_points / 3))
    base_points = np.vstack([x, y_, z])

    angles = [0, 60, 120]
    points_list = []
    labels = []

    for i, angle in enumerate(angles):
        R = rotation_matrix_y(angle)
        rotated = R @ base_points
        points_list.append(rotated.T)
        labels.extend([i] * int(n_points / 3))

    X = np.vstack(points_list)
    y = np.array(labels)

    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def dollarsign(n_points=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros((n_points, 3))
    m = np.zeros(n_points, dtype=int)
    Y = np.zeros(n_points)

    p = (3 * np.pi) / (6 + 3 * np.pi)

    for i in range(n_points):
        if np.random.rand() > p:
            m[i] = 0
            X[i] = [0, np.random.rand(), 6 * np.random.rand() - 3]
            Y[i] = 13 + X[i, 2] + np.random.randn()
        else:
            m[i] = 1
            angle = 1.5 * np.random.rand() * np.pi
            x = np.cos(angle)
            y = np.random.rand()
            z = np.sin(angle)

            if np.random.rand() > 0.5:
                X[i] = [x, y, z + 1]
                Y[i] = angle
            else:
                X[i] = [-x, y, -z - 1]
                Y[i] = 3 * np.pi - angle

    perm = np.random.permutation(n_points)
    X = X[perm]
    Y = Y[perm]
    m = m[perm]

    return X, Y, m


def roll_and_plane(n_roll=2000, n_plane=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    swiss_X, _ = make_swiss_roll(n_samples=n_roll)
    swiss_y = np.zeros(n_roll, dtype=int)

    x_plane = rng.uniform(-20, 20, n_plane)
    y_plane = rng.uniform(-10, 30, n_plane)
    z_plane = np.full(n_plane, 0)
    plane_X = np.vstack([x_plane, y_plane, z_plane]).T
    plane_y = np.ones(n_plane, dtype=int)

    X = np.vstack([swiss_X, plane_X])
    y = np.hstack([swiss_y, plane_y])

    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def two_sphere(n_points=1500, random_state=None):
    rng = np.random.default_rng(random_state)

    u1 = rng.random(n_points)
    v1 = rng.random(n_points)
    phi1 = 2 * np.pi * u1
    theta1 = np.arccos(2 * v1 - 1)

    x1 = np.sin(theta1) * np.cos(phi1)
    y1 = np.sin(theta1) * np.sin(phi1)
    z1 = np.cos(theta1)
    D1 = np.stack([x1, y1, z1], axis=1)
    label1 = np.zeros(n_points, dtype=int)

    u2 = rng.random(n_points)
    v2 = rng.random(n_points)
    phi2 = 2 * np.pi * u2
    theta2 = np.arccos(2 * v2 - 1)

    x2 = np.sin(theta2) * np.cos(phi2)
    y2 = np.sin(theta2) * np.sin(phi2) + 1
    z2 = np.cos(theta2)
    D2 = np.stack([x2, y2, z2], axis=1)
    label2 = np.ones(n_points, dtype=int)

    X = np.vstack([D1, D2])
    y = np.concatenate([label1, label2])

    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y


def rose_and_circle(n_rose=2000, n_circle=1000, random_state=None):
    rng = np.random.default_rng(random_state)

    n1 = n_rose
    theta1 = 2 * np.pi * rng.normal(loc=0, scale=1, size=n1)
    x1 = np.cos(theta1 / 0.5) * np.sin(theta1)
    y1 = np.cos(theta1 / 0.5) * np.cos(theta1)
    label1 = np.zeros(n1, dtype=int)

    n2 = n_circle
    theta2 = 2 * np.pi * rng.normal(loc=0, scale=1, size=n2)
    x2 = 0.5 * np.sin(theta2)
    y2 = 0.5 * np.cos(theta2)
    label2 = np.ones(n2, dtype=int)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    X = np.stack([x, y], axis=1)
    labels = np.concatenate([label1, label2])

    perm = rng.permutation(len(X))
    X = X[perm]
    labels = labels[perm]
    return X, labels


def five_affine_subspaces(n_points=500, random_state=None):
    n_points = int(n_points // 5)
    rng = np.random.default_rng(seed=random_state)

    def generate_line_points(start, end, n=100, noise=0.003):
        t = np.linspace(0, 1, n)
        points = np.outer(1 - t, start) + np.outer(t, end)
        points += rng.normal(scale=noise, size=points.shape)
        return points

    line1 = generate_line_points(np.array([-0.8, 0.8]), np.array([0.8, -0.8]), n_points)
    line2 = generate_line_points(np.array([-0.8, -0.8]), np.array([0.8, 0.8]), n_points)

    line3 = generate_line_points(np.array([-0.5, -0.7]), np.array([0.1, -0.7]), n_points)
    line4 = generate_line_points(np.array([0.2, -0.7]), np.array([0.5, -0.7]), n_points)

    line5 = generate_line_points(np.array([-0.6, 0.3]), np.array([-0.6, -0.3]), n_points)

    X = np.vstack([line1, line2, line3, line4, line5])
    y = np.array([0] * n_points + [1] * n_points + [2] * n_points + [3] * n_points + [4] * n_points)

    indices = rng.permutation(n_points * 5)
    X = X[indices]
    y = y[indices]

    return X, y
