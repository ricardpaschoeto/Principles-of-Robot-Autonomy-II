import numpy as np

def wrench_size(dimensions):
    """
    Returns the size of a 2D or 3D wrench.
    """
    if dimensions == 2:
        return 3
    elif dimensions == 3:
        return 6
    raise RuntimeError("wrench_size(): points must be 2D or 3D. Received a {}D vector.".format(D))

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.

    For 3D vectors, x_cross is a 3x3 skew-symmetric matrix. For 2D vectors,
    x_cross is a 2x1 vector representing the magnitude of the cross product in
    the z direction.
    """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    elif D == 3:
        return np.array([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]])
    raise RuntimeError("cross_matrix(): x must be 2D or 3D. Received a {}D vector.".format(D))

def compute_local_transformation(n):
    """
    Returns a rotation matrix that transforms a vector in a local coordinate
    system to the global coordinate system.

    The local coordinate system is constructed with the normal n as the z-axis
    in the 3D case, or n as the y-axis in the 2D case.
    """
    D = n.shape[0]
    if D == 2:
        # Set n as the y axis
        n /= np.linalg.norm(n)
        T = np.array([[n[1], n[0]],
                      [-n[0], n[1]]])

    elif D == 3:
        # Set n as the z axis

        # Compute (n x e1), (n x e2), and (n x e3), and choose the one with the
        # largest magnitude to compute x
        n_unit = n / np.linalg.norm(n)
        n_cross = cross_matrix(n_unit)
        norm_n_cross = np.linalg.norm(n_cross, axis=0)
        idx = np.argmax(norm_n_cross)
        x = n_cross[:,idx] / norm_n_cross[idx]

        # y = z x x
        y = n_cross.dot(x)
        T = np.column_stack([x, y, n_unit])

    else:
        raise RuntimeError("compute_local_transformation(): n must be 3D or 6D. Received a {}D vector.".format(D))

    return T
