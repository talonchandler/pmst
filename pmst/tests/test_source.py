from unittest import TestCase
import pmst.source
import numpy as np
from sympy import Point3D, Ray3D


class TestSource(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        direction = np.array((0, 0))
        self.r = Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
        self.r2 = Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 2))
        #self.r = pmst.source.Ray(origin, direction, polarization=polarization)
        #self.r2 = pmst.source.Ray(origin, np.array((0,1)), polarization=polarization)
        
    def test_createRay(self):
        self.assertTrue(self.r.source == Point3D(0, 0, 0))
        self.assertTrue((self.r.direction_ratio == np.array([1, 1, 1])).all())

    def test_changeRay(self):
        self.r.direction = np.array((0, 0.1))
        self.assertTrue((self.r.direction == np.array((0, 0.1))).all())

    def test_createSource(self):
        self.s = pmst.source.Source()
        self.s.add_ray(self.r)
        self.s.add_ray(self.r2)
        self.assertTrue(len(self.s.rays) == 2)

        
class TestPointSource(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        self.nrays = 2
        self.s = pmst.source.IsotropicPointSource(origin, self.nrays)

    def test_createIsotropicPointSource(self):
        self.assertTrue(len(self.s.rays) == self.nrays)
        self.assertTrue((self.s.origin == np.array((0, 0, 0))).all())
        #print(self.s)
        #print(self.s.rays[1])


