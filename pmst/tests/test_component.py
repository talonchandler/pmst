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


class TestLens(TestCase):

    def setUp(self):
        self.r = Ray(Point(0, 1, 0), Point(0, 3, 1))
        self.r2 = Ray(Point(0, 0, 0), Point(0, 0, 3))
        self.ray_list = [self.r, self.r2]
        self.s = pmst.source.RayListSource(self.ray_list)
        self.s.generate_rays()
        self.m = Microscope(source=self.s)
        self.l = Lens(Point(0, 0, 1), n=1.5, normal=Point(0, 0, 2), f=1,
                      radius=0.5)
        self.m.add_component(self.l)
        self.m.simulate()

    def test_RayListSource(self):
        self.assertTrue(self.s.n_rays == 2)
        x = self.s.ray_list.get_ray(0)
        print(x)
        print(self.s.ray_list)
