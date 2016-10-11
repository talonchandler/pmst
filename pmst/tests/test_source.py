from unittest import TestCase
import pmst.source
import numpy as np
from pmst.geometry import Point, Ray


class TestSource(TestCase):

    def setUp(self):
        self.r = Ray(Point(0, 0, 0), Point(1, 1, 1))
        self.r2 = Ray(Point(0, 0, 0), Point(1, 1, 2))
        
    def test_createRay(self):
        self.assertTrue(self.r.origin == Point(0, 0, 0))
        self.assertTrue((self.r.direction == Point(1, 1, 1)))

    def test_createSource(self):
        self.s = pmst.source.Source()
        self.s.add_ray(self.r)
        self.s.add_ray(self.r2)
        self.assertTrue(len(self.s.rays) == 2)

        
class TestPointSource(TestCase):

    def setUp(self):
        origin = Point(0, 0, 0)
        self.nrays = 100
        self.s = pmst.source.IsotropicPointSource(origin, self.nrays)

    def test_createIsotropicPointSource(self):
        self.assertTrue(len(self.s.rays) == self.nrays)
        self.assertTrue((self.s.origin == Point(0, 0, 0)))
