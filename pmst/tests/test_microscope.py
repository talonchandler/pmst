from unittest import TestCase
import pmst.microscope
import pmst.source
import pmst.component
from pmst.geometry import Point, Ray, Plane
import sys
import inspect
import numpy as np


class TestMicroscope(TestCase):

    def setUp(self):
        self.r = Ray(Point(0, 0, 0), Point(0, 0, 1))
        self.s = pmst.source.Source()
        self.s.add_ray(self.r)

        self.d = Plane(Point(0, 0, 2), Point(1, 0, 2), Point(0, 1, 2))

        self.m = pmst.microscope.Microscope(self.s)
        self.m.add_component(self.d)
        
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
        #print(i)

    def test_isotropic(self):
        from pmst.source import IsotropicPointSource
        from pmst.microscope import Microscope
        from pmst.detector import Detector

        s = IsotropicPointSource(Point(0, 0, 0), n_rays=1e6)
        m = Microscope(s)
        center = Point(0, 0, 2)
        x_edge = Point(5, 0, 2)
        y_edge = Point(0, 5, 2)
        n_pixels = 100
        d = Detector(center, x_edge, y_edge, n_pixels, n_pixels)
        m.add_component(d)
        m.simulate()
        src = inspect.getsourcelines(TestMicroscope.test_isotropic)
        name = sys._getframe().f_code.co_name
        m.plot_results('pmst/tests/output/' + name + '.png', src)


        
