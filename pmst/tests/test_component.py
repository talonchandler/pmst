import sys
sys.path.append("../../")

from unittest import TestCase
from pmst.geometry import Point, Ray
from pmst.component import Lens
from pmst.microscope import Microscope
import pmst.source

import numpy as np


class TestPixel(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        normal = np.array((0, 0))
        dimensions = np.array((.1, .2))


class TestLensIsoSourceAtFocal(TestCase):

    def setUp(self):
        self.r0 = Ray(Point(0, 0, 0), Point(0, 0, 1))
        self.r1 = Ray(Point(0, 0, 0), Point(0, 0.1, 1))
        self.r1 = Ray(Point(0, 0, 0), Point(0, 0.5, 1)) # numerical error here
        self.r2 = Ray(Point(0, 0, 0), Point(0.1, 0.1, 1))        
        self.ray_list = [self.r0, self.r1, self.r2]
        self.s = pmst.source.RayListSource(self.ray_list)
        self.s.generate_rays()
        self.m = Microscope(source=self.s)
        self.l = Lens(Point(0, 0, 1), n=1.5, normal=Point(0, 0, 2), f=1,
                      radius=1.0)
        self.m.add_component(self.l)
        self.m.simulate()

    def test_Lens(self):

        print('Focal', self.s.ray_list)
        self.assertTrue(self.s.n_rays == 3)
        
        self.assertTrue(self.s.ray_list.get_ray(0) == self.r0)
      #  self.assertTrue(self.s.ray_list.get_ray(1) == Ray(Point(0, .5, 1), Point(0, .5, 2)))
      #  self.assertTrue(self.s.ray_list.get_ray(2) == Ray(Point(.1, .1, 1), Point(.1, .1, 2)))

# Plane source converges
class TestLensPlaneSourceAtFocal(TestCase):

    def setUp(self):
        self.r0 = Ray(Point(0, 0, 0), Point(0, 0, 1))
        self.r1 = Ray(Point(0, .5, 0), Point(0, .5, 1))
        self.r2 = Ray(Point(.1, .1, 0), Point(.1, .1, 1))        
        self.ray_list = [self.r0, self.r1, self.r2]
        self.s = pmst.source.RayListSource(self.ray_list)
        self.s.generate_rays()
        self.m = Microscope(source=self.s)
        self.l = Lens(Point(0, 0, 1), n=1.5, normal=Point(0, 0, 2), f=1,
                      radius=1.0)
        self.m.add_component(self.l)
        self.m.simulate()

    def test_Lens(self):

        print('Plane', self.s.ray_list)        
        self.assertTrue(self.s.n_rays == 3)
        self.assertTrue(self.s.ray_list.get_ray(0) == Ray(Point(0, 0, 1), Point(0, 0, 2)))
        self.assertTrue(self.s.ray_list.get_ray(1) == Ray(Point(0, .5, 1), Point(0, 0, 2)))
        self.assertTrue(self.s.ray_list.get_ray(2) == Ray(Point(.1, .1, 1), Point(0, 0, 2)))
        
