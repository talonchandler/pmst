import numpy as np


class Point():
    """A  point in a 3-dimensional Euclidean space."""

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return Point(other*self.x, other*self.y, other*self.z)
    
    def __contains__(self, other):
        return self.__eq__(other)

    def __str__(self):
        return 'Point(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

    def __repr__(self):
        return self.__str__()

    @property
    def length(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other):
        return Point(self.y*other.z - self.z*other.y,
                     self.z*other.x - self.x*other.z,
                     self.x*other.y - self.y*other.x)

    # TODO
    def are_collinear(self, *args):
        return False

    def normalize(self):
        return Point(self.x/self.length, self.y/self.length, self.z/self.length)


class Ray:
    """A half line in 3-dimensional Euclidean space from point1 to infinity in 
    the direction of point2."""

    def __init__(self, origin=None, direction=None):
        self.origin = origin
        self.direction = direction

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __contains__(self, other):
        if isinstance(other, Point):  # Is a point on a half line
            if (other == self.origin) or (other == self.direction):
                return True
            if self.direction.cross(self.origin - other) == Point(0, 0, 0):
                if self.direction.dot(other) >= 0:
                    return True
                else:
                    return False
        else:
            raise TypeError("Can't determine if" + str(type(other)) + "is in Ray.")
        
    def __str__(self):
        return 'Ray(' + str(self.origin) + ', ' + str(self.direction) + ')'

    def __repr__(self):
        return self.__str__()
    

class Plane:
    """A 2-dimensional plane in 3-dimensional space. """

    # TODO implement constuctor with single point and normal
    # TODO throw error if 3 collinear points are given 
    def __init__(self, p1, p2=None, p3=None, **kwargs):
        if p2 and p3:
            self.p1 = p1
            self.normal = (p2 - p1).cross(p3 - p2).normalize()
        else:
            n = kwargs.pop('normal', p2)
            if isinstance(n, Point):
                self.normal = n.normalize()
            else:
                raise ValueError("Either provide 3 3D points or a point with\
                                 a normal vector.")
            
    def __contains__(self, other):
        if isinstance(other, Point):  # Is a point on the plane?
            if (other - self.p1).dot(self.normal) == 0:
                return True
            else:
                return False
        elif isinstance(other, Ray):  # Is a ray in the plane?
            if (other.origin in self) and (other.direction in self):
                return True
            else:
                return False
        else:
            raise TypeError("Can't determine if" + str(type(other)) +
                            "is in Plane.")
        
    def __str__(self):
        return 'Plane(' + str(self.p1) + ', ' + str(self.p2) + ', ' +\
            str(self.p3) + ')'

    def __repr__(self):
        return self.__str__()
    
    def intersection(self, o):
        """ Returns a list of intersections with a Ray or Point."""
        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []
        if isinstance(o, Ray):
            # If ray is entirely in the plane
            if o in self:
                return [o]
            # If ray is parallel to the plane
            if (o.direction - o.origin).dot(self.normal) == 0:
                return []
            # If ray line has a single intersection with the plane
            else:
                p0 = self.p1
                l0 = o.origin
                n = self.normal
                l = o.direction - l0
                d = ((p0 - l0).dot(n))/(l.dot(n))
                if d >= 0:
                    return [l*d + l0]
                # If ray is in wrong direction to intersect plane
                else:
                    return []
                
        if isinstance(o, Plane):
            # TODO
            return []
