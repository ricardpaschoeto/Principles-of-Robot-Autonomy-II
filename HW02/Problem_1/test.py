#!/usr/bin/env python

import unittest

from form_force_closure import *

class TestFormClosure(unittest.TestCase):

    def test_planar_form_closure_1(self):
        # Grasping a rectangle
        forces = [np.array([1., 0.]),
                  np.array([0., 1.]),
                  np.array([-1., 0.]),
                  np.array([0., -1.])]
        points = [np.array([-1., -1.]),
                  np.array([0., -2.]),
                  np.array([1., 1.]),
                  np.array([0., 2.])]
        self.assertFalse(is_in_form_closure(forces, points))

    def test_planar_form_closure_2(self):
        # Grasping a square
        forces = [np.array([1., 0.]),
                  np.array([0., 1.]),
                  np.array([-1., 0.]),
                  np.array([0., -1.])]
        points = [np.array([-1., -2.]),
                  np.array([-1., -2.]),
                  np.array([1., 2.]),
                  np.array([1., 2.])]
        self.assertTrue(is_in_form_closure(forces, points))

    def test_spatial_form_closure_1(self):
        # Grasping a cube
        forces = [np.array([0., 0., 1.]),
                  np.array([-1., 0., 0.]),
                  np.array([0., 1., 0.]),
                  np.array([0., 0., -1.]),
                  np.array([1., 0., 0.]),
                  np.array([0., -1., 0.]),
                  np.array([0., 0., -1.])]
        points = [np.array([1., -1., -1.]),
                  np.array([1., -1., -1.]),
                  np.array([1., -1., -1.]),
                  np.array([-1., 1., 1.]),
                  np.array([-1., 1., 1.]),
                  np.array([-1., 1., 1.]),
                  np.array([1., 1., 1.])]
        self.assertFalse(is_in_form_closure(forces, points))

    def test_spatial_form_closure_2(self):
        # Grasping a cube
        forces = [np.array([0., 0., 1.]),
                  np.array([-1., 0., 0.]),
                  np.array([0., 1., 0.]),
                  np.array([0., 0., -1.]),
                  np.array([1., 0., 0.]),
                  np.array([0., -1., 0.]),
                  np.array([0., 0., -1.])]
        points = [np.array([1., 0., -1.]),
                  np.array([1., -1., -1.]),
                  np.array([1., -1., -1.]),
                  np.array([-1., 1., 1.]),
                  np.array([-1., 1., 1.]),
                  np.array([-1., 1., 1.]),
                  np.array([1., 1., 1.])]
        self.assertTrue(is_in_form_closure(forces, points))

class TestForceClosure(unittest.TestCase):

    def test_planar_force_closure_1(self):
        # Grasping a square
        forces = [np.array([1., 0.]),
                  np.array([-1., 0.])]
        points = [np.array([-1., 0.]),
                  np.array([1., 0.])]
        friction_coeffs = [1., 0.]
        self.assertFalse(is_in_force_closure(forces, points, friction_coeffs))

    def test_planar_force_closure_2(self):
        # Grasping a square
        forces = [np.array([1., 0.]),
                  np.array([-1., 0.])]
        points = [np.array([-1., 0.]),
                  np.array([1., 0.])]
        friction_coeffs = [1., 1.]
        self.assertTrue(is_in_force_closure(forces, points, friction_coeffs))

    def test_spatial_force_closure_3(self):
        # Grasping a cube
        forces = [np.array([1., 0., 0.]),
                  np.array([0., 0., 1.]),
                  np.array([0., 0., -1.])]
        points = [np.array([-1., 0., 0.]),
                  np.array([0., 0., -1.]),
                  np.array([0., 0., 1.])]
        friction_coeffs = [1., 1., 1.]
        self.assertTrue(is_in_force_closure(forces, points, friction_coeffs))

    def test_spatial_force_closure_4(self):
        # Grasping a cube
        forces = [np.array([1., 0., 0.]),
                  np.array([0., 0., 1.]),
                  np.array([0., 0., -1.])]
        points = [np.array([-1., 0., 0.]),
                  np.array([0., 0., -1.]),
                  np.array([0., 0., 1.])]
        friction_coeffs = [0., 1., 1.]
        self.assertFalse(is_in_force_closure(forces, points, friction_coeffs))

if __name__ == '__main__':
    unittest.main()
