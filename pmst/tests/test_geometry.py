from unittest import TestCase
from pmst.geometry import Point, Ray, Plane
import numpy as np


class TestGeometry(TestCase):

    def setUp(self):
        self.point1 = Point(0, 0, 0)
        self.point2 = Point(0, 0, 1)
        self.point3 = Point(0, 1, 0)        
        self.ray1 = Ray(self.point1, self.point2)
        self.plane1 = Plane(self.point1, self.point2, self.point3)

    def test_point(self):
        self.assertTrue(self.point1.x == 0)
        self.assertTrue(self.point1.y == 0)
        self.assertTrue(self.point1.z == 0)
        self.assertTrue(self.point2.z == 1)
        self.assertTrue(self.point3.y == 1)
        self.assertTrue(self.point1 != self.point2)
        self.assertTrue(self.point1 == self.point1)
        self.assertFalse(self.point1 in self.point2)
        self.assertTrue(-Point(1, 1, 1) == Point(-1, -1, -1))
        self.assertTrue(Point(1, 1, 1) + Point(1, 2, 3) == Point(2, 3, 4))

    def test_pointMethod(self):
        self.assertTrue(self.point1.dot(self.point2) == 0)
        self.assertTrue(self.point2.dot(Point(1, 1, 1)) == 1)
        self.assertTrue(self.point2.cross(Point(0, 0, 0)) == Point(0, 0, 0))
        self.assertTrue(Point(1, 0, 0).cross(Point(0, 1, 0)) == Point(0, 0, 1))
        self.assertTrue(Point(1, 0, 0).cross(Point(0, 1, 0)) == Point(0, 0, 1))
        
    def test_ray(self):
        self.assertTrue(self.point1 in self.ray1)
        self.assertFalse(self.point3 in self.ray1)
        self.assertTrue(Point(0, 0, 2) in self.ray1)
        self.assertFalse(Point(0, 0, -1) in self.ray1)
