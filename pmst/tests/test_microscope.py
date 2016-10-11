from unittest import TestCase
import cProfile
from pstats import Stats
import pmst.microscope
import pmst.source
import pmst.component
import numpy as np
from sympy import Ray3D, Point3D, Plane


class TestMicroscope(TestCase):

    def setUp(self):
        self.r = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
        self.s = pmst.source.Source()
        self.s.add_ray(self.r)

        self.d = Plane(Point3D(0, 0, 1), normal_vector=(0, 0, 1))

        self.m = pmst.microscope.Microscope(self.s)
        self.m.add_component(self.d)
        
    def test_createMicroscope(self):
        self.assertTrue(len(self.m.component_list) == 1)
        
    def test_singleIntersection(self):
        i = self.m.simulate()
        self.assertTrue(i[0][0] == Point3D(0, 0, 1))
        
    def test_noIntersections(self):
        self.m.source.rays[0] = Ray3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
        i = self.m.simulate()
        self.assertTrue(i == [[]])

    def test_isotropic(self):
        self.m.source = pmst.source.IsotropicPointSource(Point3D(0, 0, 0), 1)
        i = self.m.simulate()

