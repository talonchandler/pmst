from unittest import TestCase
from pmst.detector import Detector
from pmst.geometry import Point, Ray, Plane
import numpy as np

class TestDetector(TestCase):

    def test_detector(self):
        p1 = Point(0, 0, 2)
        p2 = Point(0, 1, 2)
        p3 = Point(1, 0, 2)
        d = Detector(p1, p2, p3, 100, 100)
        self.assertTrue(np.size(d.pixel_values) == 100*100)
