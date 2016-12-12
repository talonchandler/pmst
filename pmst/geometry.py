import numpy as np

util = """        
#include <stdio.h>
#define PI 3.14159265

typedef struct Point {
    double x;
    double y;
    double z;
} Point;

typedef struct Plane {
    Point n;
    Point c;
} Plane;

typedef struct Ray {
    Point r0;
    Point r1;
} Ray;


__device__ Point add(struct Point p1, struct Point p2) {
    Point out = {p1.x + p2.x, p1.y + p2.y, p1.z + p2.z};
    return out;
}

__device__ Point subtract(struct Point p1, struct Point p2) {
    Point out = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
    return out;
}

__device__ Point scale(struct Point p1, double s) {
    Point out = {s*p1.x, s*p1.y, s*p1.z};
    return out;
}

__device__ double dot(struct Point p1, struct Point p2) {
    double out = p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
    return out;
}

__device__ Point cross(struct Point p1, struct Point p2) {
    Point out = {p1.y*p2.z - p1.z*p2.y,
                 p1.z*p2.x + p1.x*p2.z,
                 p1.x*p2.y - p1.y*p2.x};
    return out;
}

__device__ double len(struct Point p1) {
    double out = sqrt(p1.x*p1.x + p1.y*p1.y + p1.z*p1.z);
    return out;
}

__device__ Point intersect(struct Ray r, struct Plane p) {
    Point l = subtract(r.r1, r.r0);
    double d = dot(subtract(p.c, r.r0), p.n)/dot(l, p.n);
    Point out = add(scale(l, d), r.r0);  // Intersection point
    return out;
}

__device__ Point projectPointToPlane(struct Point r, struct Plane p) {
    // Double check that p.n is the unit normal (from origin not c)
    // http://math.stackexchange.com/a/445015/357869
    Point no = subtract(p.n, p.c);
    
    double d = dot(subtract(r, p.c), p.n);
    printf("XXX: %lf \\n", d);
    Point out = subtract(r, scale(p.n, d));
    return out;
}

"""


class Point():
    """A  point in a 3-dimensional Euclidean space."""

    def __init__(self, x=0, y=0, z=0):
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.z = np.float64(z)

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
        return 'Point(' + '{:.2f}'.format(self.x) + ', ' + '{:.2f}'.format(self.y) + ', ' + '{:.2f}'.format(self.z) + ')'

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

    def is_collinear(self, p2, p3):
        if (self - p2).cross(self - p3).length <= 1e-6:
            return True
        else:
            return False

    def normalize(self):
        return Point(self.x/self.length,
                     self.y/self.length,
                     self.z/self.length)


class Ray:
    """A half line in 3-dimensional Euclidean space from point1 to infinity in 
    the direction of point2."""

    def __init__(self, origin=None, direction=None):
        if origin == direction:
           raise ValueError("Provide two unique points for the origin and direction.")
        else:
            self.origin = origin
            self.direction = direction

    # TODO: Test that direction is the same too
    # TODO: Clean this method
    def __eq__(self, other):
        if self.origin.is_collinear(other.origin, other.direction):
            if self.direction.is_collinear(other.origin, other.direction):
                return True
        if self.__dict__ == other.__dict__:
            return True
        return False

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
            raise TypeError("Can't determine if" + str(type(other)) +
                            "is in Ray.")
        
    def __str__(self):
        return 'Ray(' + str(self.origin) + ', ' + str(self.direction) + ')'

    def __repr__(self):
        return self.__str__()
    

class Plane:
    """A 2-dimensional plane in 3-dimensional space. """

    def __init__(self, p1, p2=None, p3=None, **kwargs):
        if p2 and p3:
            if p1.is_collinear(p2, p3):
                raise ValueError("Provide 3 points that are not collinear.")
            else:
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
        return 'Plane(' + str(self.p1) + ', normal=' + str(self.normal) + ')'

    def __repr__(self):
        return self.__str__()
    
    def intersect(self, o):
        """ Change this function to return just the new ray. """

        """ Returns a list of intersections with a Ray or Point."""
        if isinstance(o, Point):
            if o in self:
                return o
            else:
                print("POINT")                
                return None
        if isinstance(o, Ray):
            # If ray is entirely in the plane
            if o in self:
                return o
            # If ray is parallel to the plane
            if (o.direction - o.origin).dot(self.normal) == 0:
                print("PARALLEL")
                return None
            # If ray line has a single intersection with the plane
            else:
                p0 = self.p1
                l0 = o.origin
                n = self.normal
                l = o.direction - l0
                d = ((p0 - l0).dot(n))/(l.dot(n))
                if d >= 0:
                    new_o = l*d + l0
                    return Ray(origin=new_o, direction=(new_o + o.direction))
                # If ray is in wrong direction to intersect plane
                else:
                    print("Wrong direction")
                    return None
                
        if isinstance(o, Plane):
            # TODO
            return None
