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
    
    def __contains__(self, other):
        return self.__eq__(other)

    def __str__(self):
        return 'Point(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other):
        return Point(self.y*other.z - self.z*other.y,
                     self.z*other.x - self.x*other.z,
                     self.x*other.y - self.y*other.x)


class Ray:
    """A half line in 3-dimensional Euclidean space from point1 to infinity in 
    the direction of point2."""

    def __init__(self, origin=None, direction=None):
        self.origin = origin
        self.direction = direction

    def __contains__(self, other):
        if isinstance(other, Point):  # Is a point on a half line
            if (other == self.origin) or (other == self.direction):
                return True
            if self.direction.cross(self.origin - other) == Point(0, 0, 0):
                if self.direction.dot(other) >= 0:
                    return True
                else:
                    return False
        
    def __str__(self):
        return 'Ray(' + str(self.origin) + ', ' + str(self.direction) + ')'
        

class Plane:
    """A 2-dimensional plane in 3-dimensional space. """

    def __init__(self, point1=None, point2=None, point3=None):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

    def __contains__(self):
        # TODO
        return True

    def __str__(self):
        return 'Plane(' + str(self.point1) + ', ' + str(self.point2) + ', ' + str(self.point3) + ')'
    
    
    def intersection(self, o):
        """ Returns a list of intersections with a Ray or Point."""

        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []
        if isinstance(o, Ray):
            # TODO
            return []
        if isinstance(o, Plane):
            # TODO
            return []
            
        
    
