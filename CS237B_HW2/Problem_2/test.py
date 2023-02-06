#!/usr/bin/env python

import unittest

import numpy as np

from grasp_optimization import *

TOL_ERR = 1e-5

class TestGraspOptimization(unittest.TestCase):

    def test_planar_grasp_1(self):
        # Grasping a square
        grasp_normals = [np.array([1., 0.]),
                         np.array([-1., 0.])]
        points = [np.array([-1., 0.]),
                  np.array([1., 0.])]
        friction_coeffs = [1., 1.]
        wrench_ext = np.array([0., -1., 0.])

        forces = grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext)
        self.assertTrue(len(forces) == 2)
        self.assertTrue(np.linalg.norm(forces[0] - np.array([0.5, 0.5])) < TOL_ERR)
        self.assertTrue(np.linalg.norm(forces[1] - np.array([-0.5, 0.5])) < TOL_ERR)

    def test_spatial_grasp_1(self):
        # Grasping a cube
        grasp_normals = [np.array([1., 0., 0.]),
                         np.array([0., 0., 1.]),
                         np.array([0., 0., -1.])]
        points = [np.array([-1., 0., 0.]),
                  np.array([0., 0., -1.]),
                  np.array([0., 0., 1.])]
        friction_coeffs = [0., 1., 1.]
        wrench_ext = np.array([0., -1., 0., 0., 0., 0.])

        forces = grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext)
        self.assertTrue(len(forces) == 3)
        self.assertTrue(np.linalg.norm(forces[0] - np.array([0., 0., 0.])) < TOL_ERR)
        self.assertTrue(np.linalg.norm(forces[1] - np.array([0., 0.5, 0.5])) < TOL_ERR)
        self.assertTrue(np.linalg.norm(forces[2] - np.array([0., 0.5, -0.5])) < TOL_ERR)

    def test_planar_force_closure_1(self):
        # Grasping a square
        grasp_normals = [np.array([1., 0.]),
                         np.array([-1., 0.])]
        points = [np.array([-1., 0.]),
                  np.array([1., 0.])]
        friction_coeffs = [1., 1.]
        wrench_ext = np.array([0., -1., 0.])

        force_closure = precompute_force_closure(grasp_normals, points, friction_coeffs)
        forces = force_closure(wrench_ext)
        self.assertTrue(len(forces) == 2)
        self.assertTrue(np.linalg.norm(forces[0] - np.array([0.5, 0.5])) < TOL_ERR)
        self.assertTrue(np.linalg.norm(forces[1] - np.array([-0.5, 0.5])) < TOL_ERR)


if __name__ == '__main__':
    unittest.main()
