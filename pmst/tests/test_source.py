import sys
sys.path.append("../../")

from unittest import TestCase
import pmst.source
import numpy as np
from pmst.geometry import Point, Ray

class TestIsotropicPointSource(TestCase):

    def setUp(self):
        origin = Point(0, 0, 0)
        self.nrays = 100
        self.s = pmst.source.IsotropicPointSource(origin, self.nrays)
        self.s.generate_rays()

    def test_createIsotropicPointSource(self):
        self.assertTrue(len(self.s.ray_list.ray_list[0]) == self.nrays)
        self.assertTrue((self.s.origin == Point(0, 0, 0)))

class TestDirectedPointSource(TestCase):

    def setUp(self):
        self.dps = pmst.source.DirectedPointSource(Point(0, 0, 0), n_rays=1, direction=Point(0, 0, 1), psi=np.pi/2)
        self.dps.generate_rays()
        
    def test_createSource(self):
        self.assertTrue(self.dps.n_rays == 1)
        # print(self.dps.ray_list)
        # print("HELLO")

class TestRayListSource(TestCase):

    def setUp(self):
        self.r = Ray(Point(0, 0, 0), Point(1, 1, 1))
        self.r2 = Ray(Point(0, 0, 0), Point(1, 1, 3))
        self.ray_list = [self.r, self.r2]
        self.s = pmst.source.RayListSource(self.ray_list)
        self.s.generate_rays()        

    def test_RayListSource(self):
        self.assertTrue(self.s.n_rays == 2)
        #print(self.s.ray_list)
    
        
