from unittest import TestCase
from pmst.geometry import Point, Ray, Plane
import numpy as np


class TestGeometry(TestCase):

    def setUp(self):
        self.p1 = Point(0, 0, 0)
        self.p2 = Point(0, 0, 1)
        self.p3 = Point(0, 1, 0)
        self.ray1 = Ray(self.p1, self.p2)
        self.plane1 = Plane(self.p1, self.p2, self.p3)

    def test_point(self):
        self.assertTrue(self.p1.x == 0)
        self.assertTrue(self.p1.y == 0)
        self.assertTrue(self.p1.z == 0)
        self.assertTrue(self.p2.z == 1)
        self.assertTrue(self.p3.y == 1)
        self.assertTrue(self.p1 != self.p2)
        self.assertTrue(self.p1 == self.p1)
        self.assertFalse(self.p1 in self.p2)
        self.assertTrue(-Point(1, 1, 1) == Point(-1, -1, -1))
        self.assertTrue(Point(1, 1, 1) + Point(1, 2, 3) == Point(2, 3, 4))

    def test_pointMethod(self):
        self.assertTrue(self.p1.dot(self.p2) == 0)
        self.assertTrue(self.p2.dot(Point(1, 1, 1)) == 1)
        self.assertTrue(self.p2.cross(Point(0, 0, 0)) == Point(0, 0, 0))
        self.assertTrue(Point(1, 0, 0).cross(Point(0, 1, 0)) == Point(0, 0, 1))
        self.assertTrue(Point(1, 0, 0).cross(Point(0, 1, 0)) == Point(0, 0, 1))
        self.assertTrue(Point(20, 0, 0).normalize() == Point(1, 0, 0))
        self.assertTrue(Point(1, 1, 1).normalize() == Point(1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)))        
        
    def test_ray(self):
        self.assertRaises(TypeError, self.ray1.__contains__, self.ray1, 1)
        self.assertTrue(self.p1 in self.ray1)
        self.assertFalse(self.p3 in self.ray1)
        self.assertTrue(Point(0, 0, 2) in self.ray1)
        self.assertFalse(Point(0, 0, -1) in self.ray1)

    def test_plane(self):
        self.assertTrue(self.plane1.normal == Point(1, 0, 0))
        self.assertTrue(Point(0, 0, 0) in Plane(Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)))
        self.assertFalse(Point(1, 1, 1) in Plane(Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)))
        self.assertTrue(Point(1, 4, 1) in Plane(Point(0, 0, 0), Point(1, 1, 1), Point(0, 1, 0)))
        self.assertTrue(Ray(Point(0, 0, 0), Point(1, 1, 1)) in Plane(Point(0, 0, 0), Point(1, 1, 1), Point(0, 1, 0)))
        self.assertTrue(Ray(Point(-3, -3, -3), Point(1, 17, 1)) in Plane(Point(0, 0, 0), Point(1, 1, 1), Point(0, 1, 0)))

    def test_intersection(self):
        p = Plane(Point(0, 0, 0), Point(1, 1, 1), Point(0, 0, 1))
        self.assertTrue(p.intersection(Point(7, 7, 7)))
        r = Ray(Point(0, 0, 0), Point(2, 2, -10)) # In plane
        self.assertTrue(p.intersection(r)[0] == r)
        p2 = Plane(Point(0, 0, 0), Point(0, 1, 0), Point(1, 1, 0))
        r2 = Ray(Point(10, 0, 1), Point(0, 0, -1)) # Single intersection
        self.assertTrue(p2.intersection(r2)[0] == Point(5, 0, 0))
        r3 = Ray(Point(0, 0, 1), Point(2, 2, 1)) # Parallel
        self.assertTrue(p2.intersection(r3) == [])
        r4 = Ray(Point(0, 0, 1), Point(0, 0, 2)) # No intersection
        self.assertTrue(p2.intersection(r4) == [])
