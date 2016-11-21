import sys
sys.path.append("../../")

from unittest import TestCase
import pmst.component
import numpy as np


class TestPixel(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        normal = np.array((0, 0))
        dimensions = np.array((.1, .2))
        self.p = pmst.component.Pixel(origin, normal, dimensions)
        
    def test_createPixel(self):
        self.assertTrue((self.p.origin == np.array((0, 0, 0))).all())
        self.assertTrue((self.p.normal == np.array((0, 0))).all())
        self.assertTrue((self.p.dimensions == np.array((.1, .2))).all())
        self.assertTrue(self.p.count == 0)
        #print(self.p)
