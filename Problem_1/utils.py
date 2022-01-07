import math, pdb, os, sys

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt


def map_chunked(fn, chunk_size, n, verbose=False):
    """
    Map a function over iterates in chunks to save memory.
    You DO NOT need to use this.
    """
    ret = []
    rng = range(math.ceil(n / chunk_size))
    rng = rng if not verbose else tqdm(rng)
    for i in rng:
        i1, i2 = i * chunk_size, min((i + 1) * chunk_size, n)
        ret.append(fn(i1, i2))
    return tf.concat(ret, 0)


def visualize_value_function(V):
    """
    Visualizes the value function given in V & computes the optimal action,
    visualized as an arrow.

    You need to call plt.show() yourself.

    Args:
        V: (np.array) the value function reshaped into a 2D array.
    """
    V = np.array(V)
    assert V.ndim == 2
    m, n = V.shape
    pos2idx = np.arange(m * n).reshape((m, n))
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    u, v = [], []
    for pt in pts:
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])
    u, v = np.reshape(u, (m, n)), np.reshape(v, (m, n))

    plt.imshow(V.T, origin="lower")
    plt.quiver(X, Y, u, v, pivot="middle")


def make_transition_matrices(m, n, x_eye, sig):
    """
    Compute the transisiton matrices T, which maps a state probability vector to
    a next state probability vector.

        prob(S') = T @ prob(S)

    Args:
        n (int): the width and height of the grid
        x_eye (Sequence[int]): 2 element vector describing the storm location
        sig (float): standard deviation of the storm, increases storm size

    Returns:
        List[np.array]: 4 transition matrices for actions
                                                {right, up, left, down}
    """

    sdim = m * n

    # utility functions
    w_fn = lambda x: np.exp(-np.linalg.norm(np.array(x) - x_eye) / sig ** 2 / 2)
    xclip = lambda x: min(max(0, x), m - 1)
    yclip = lambda y: min(max(0, y), n - 1)

    # graph building
    pos2idx = np.reshape(np.arange(m * n), (m, n))
    y, x = np.meshgrid(np.arange(n), np.arange(m))
    idx2pos = np.stack([x.reshape(-1), y.reshape(-1)], -1)

    T_right, T_up, T_left, T_down = [np.zeros((sdim, sdim)) for _ in range(4)]
    for i in range(m):
        for j in range(n):
            z = (i, j)
            w = w_fn(z)
            right = (xclip(z[0] + 1), yclip(z[1] + 0))
            up = (xclip(z[0] + 0), yclip(z[1] + 1))
            left = (xclip(z[0] - 1), yclip(z[1] + 0))
            down = (xclip(z[0] + 0), yclip(z[1] - 1))

            T_right[pos2idx[i, j], pos2idx[right[0], right[1]]] += 1 - w
            T_right[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_right[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_right[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_up[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_up[pos2idx[i, j], pos2idx[up[0], up[1]]] += 1 - w
            T_up[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_up[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_left[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_left[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_left[pos2idx[i, j], pos2idx[left[0], left[1]]] += 1 - w
            T_left[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_down[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[down[0], down[1]]] += 1 - w
    return (T_right, T_up, T_left, T_down), pos2idx, idx2pos


def generate_problem():
    """
    A function that generates the problem data for Problem 1.

    Generates transition matrices for each of the four actions.
    Generates pos2idx array which allows to convert from a (i, j) grid
        coordinates to a vectorized state (1D).
    """
    n = 20
    m = n
    sdim, adim = m * n, 4

    # the parameters of the storm
    x_eye, sig = np.array([15, 7]), 1e0

    Ts, pos2idx, idx2pos = make_transition_matrices(m, n, x_eye, sig)

    Ts = [tf.convert_to_tensor(T, dtype=tf.float32) for T in Ts]
    Problem = dict(Ts=Ts, n=n, m=m, pos2idx=pos2idx, idx2pos=idx2pos)
    return Problem
