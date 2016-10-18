from unittest import TestCase
import pmst.microscope
import pmst.source
import pmst.component
from pmst.detector import Detector
from pmst.geometry import Point, Ray, Plane
import sys
import inspect
import numpy as np


class TestMicroscope(TestCase):

    def setUp(self):
        self.r = Ray(Point(0, 0, 0), Point(0, 0, 1))
        self.s = pmst.source.Source()
        self.s.add_ray(self.r)
        self.d = Detector(Point(0, 0, 2), Point(1, 0, 2), Point(0, 1, 2))
        self.m = pmst.microscope.Microscope(self.s, self.d)
        
    def test_createMicroscope(self):
        self.assertTrue(len(self.m.component_list) == 1)
        
    def test_singleIntersection(self):
        i = self.m.simulate()
        self.assertTrue(i[0][0] == Point(0, 0, 2))
        
    def test_noIntersections(self):
        self.m.source.rays[0] = Ray(Point(0, 0, 0), Point(0, 1, 0))
        i = self.m.simulate()
        self.assertTrue(i == [[]])

    def test_isotropic2(self):
        self.m.source = pmst.source.IsotropicPointSource(Point(0, 0, 0), 1e1)
        i = self.m.simulate()
        # print(i)
