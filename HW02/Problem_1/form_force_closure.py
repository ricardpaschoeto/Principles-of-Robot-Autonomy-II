import cvxpy as cp
import numpy as np

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

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.
    Works for 2D and 3D.

    Args:
        f - 2D or 3D contact force.
        p - 2D or 3D contact point.

    Return:
        w - 3D or 6D contact wrench represented as (force, torque).    
    """
    ########## Your code starts here ##########
    # Hint: you may find cross_matrix(x) defined above helpful. This should be one line of code.
    w = np.concatenate((f, cross_matrix(p) @ f))
    ########## Your code ends here ##########

    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D or 3D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]

    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 2
        if np.where(f != 0)[0] == 0:
            edges[0] =  np.array([f[0],  f[0] * mu])
            edges[1] =  np.array([f[0], -f[0] * mu])
        else:
            edges[0] =  np.array([ f[1] * mu, f[1]])
            edges[1] =  np.array([-f[1] * mu, f[1]])            
        ########## Your code ends here ##########

    # Spatial wrenches
    elif D == 3:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 4
        if np.where(f != 0)[0] == 0:
            edges[0] = np.array([f[0], f[0] * mu,   0])
            edges[1] = np.array([f[0], -f[0] * mu,   0])
            edges[2] = np.array([f[0],   0,  f[0] * mu])
            edges[3] = np.array([f[0],   0, -f[0] * mu])
        elif np.where(f != 0)[0] == 1:
            edges[0] = np.array([ f[1] * mu, f[1],   0])
            edges[1] = np.array([-f[1] * mu, f[1],   0])
            edges[2] = np.array([0,   f[1],  f[1] * mu])
            edges[3] = np.array([0,   f[1], -f[1] * mu])
        else:
            edges[0] = np.array([ f[2] * mu, 0, f[2]])
            edges[1] = np.array([-f[2] * mu, 0, f[2]])
            edges[2] = np.array([0,  f[2] * mu, f[2]])
            edges[3] = np.array([0, -f[2] * mu, f[2]])   
        ######### Your code ends here ##########

    else:
        raise RuntimeError("cone_edges(): f must be 3D or 6D. Received a {}D vector.".format(D))

    return edges

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D or 6D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    ########## Your code starts here ##########
    # Hint: you may find np.linalg.matrix_rank(F) helpful
    # TODO: Replace the following program (check the cvxpy documentation)

    # k = cp.Variable(1)
    # objective = cp.Minimize(k)
    # constraints = [k >= 0]
    rank = np.linalg.matrix_rank(F)
    if rank == F.shape[0]:
        k = cp.Variable(shape=(F.shape[1],1), name='k')
        objective = cp.Minimize(np.ones((1, F.shape[1])) @ k)
        constraints = [F @ k == 0, k >= 1]
    else:
        return False        

    ########## Your code ends here ##########

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)
    return prob.status not in ['infeasible', 'unbounded']

def is_in_form_closure(normals, points):
    """
    Calls form_closure_program() to determine whether the given contact normals
    are in form closure.

    Args:
        normals - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.

    Return:
        True/False - whether the forces are in form closure.
    """
    ########## Your code starts here ##########
    # TODO: Construct the F matrix (not necessarily 6 x 7)
    if points[0].shape[0] == 2:
        F = np.zeros((3,4))
        for ii, (force, point) in enumerate(zip(normals, points)):
            w = wrench(force, point)
            F[0, ii] = w[0]
            F[1, ii] = w[1]
            F[2, ii] = w[2]
    elif points[0].shape[0] == 3:
        F = np.zeros((6,7))
        for ii, (force, point) in enumerate(zip(normals, points)):
            w = wrench(force, point)
            F[0, ii] = w[0]
            F[1, ii] = w[1]
            F[2, ii] = w[2]
            F[3, ii] = w[3]
            F[4, ii] = w[4]
            F[5, ii] = w[5]
    ########## Your code ends here ##########

    return form_closure_program(F)

def is_in_force_closure(forces, points, friction_coeffs):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in force closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.
        friction_coeffs - list of friction coefficients.

    Return:
        True/False - whether the forces are in force closure.
    """
    ########## Your code starts here ##########
    # TODO: Call cone_edges() to construct the F matrix (not necessarily 6 x 7)
    F = []
    for force, point, mu in zip(forces, points, friction_coeffs):
        edges = cone_edges(force, mu)
        for edge in edges:
            w = wrench(edge, point)
            F.append(w)                  
    F = np.asarray(F).T
    ########## Your code ends here ##########

    return form_closure_program(F)
