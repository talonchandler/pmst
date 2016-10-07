from unittest import TestCase
import pmst.source
import numpy as np


class TestSource(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        direction = np.array((0, 0))
        polarization = np.array((1, 0))
        self.r = pmst.source.Ray(origin, direction, polarization=polarization)
        self.r2 = pmst.source.Ray(origin, np.array((0,1)), polarization=polarization)
        
    def test_createRay(self):
        self.assertTrue((self.r.origin == np.array((0, 0, 0))).all())
        self.assertTrue((self.r.direction == np.array((0, 0))).all())
        self.assertTrue((self.r.polarization == np.array((1, 0))).all())

    def test_changeRay(self):
        self.r.direction = np.array((0, 0.1))
        self.assertTrue((self.r.direction == np.array((0, 0.1))).all())

    def test_propagate(self):
        self.assertTrue(self.r.propagate(0) == 1)

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
