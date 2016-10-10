from unittest import TestCase
import pmst.microscope
import pmst.source
import pmst.component
import numpy as np


class TestMicroscope(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        normal = np.array((0, 0))
        dimensions = np.array((.1, .2))
        self.p = pmst.component.Pixel(origin, normal, dimensions)

        origin = np.array((0, 0, 0))
        direction = np.array((0, 0))
        polarization = np.array((1, 0))
        self.r = pmst.source.Ray(origin, direction, polarization=polarization)

        self.m = pmst.microscope.Microscope(self.r)
        self.m.add_component(self.p)
        
    def test_createMicroscope(self):
        self.assertTrue(len(self.m.component_list) == 1)
        print(self.m)
