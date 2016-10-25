from pmst.geometry import Point, Ray, Plane
import numpy as np


class Detector(Plane):
    """Square plane detector with square pixels.

    Arguments:
    p1 = center point coodinates
    p2 = x edge coordinates
    p3 = y edge coordinates
    xnpix = number of pixels in the x direction
    ynpix = number of pixels in the x direction

    Example:
    Detector(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0),
             xnpix=100, ynpix=100)
    """
    
    def __init__(self, center, x_edge=None, y_edge=None, xnpix=1, ynpix=1):
        Plane.__init__(self, center, x_edge, y_edge)
        self.xwidth = 2*(x_edge - center).length
        self.ywidth = 2*(y_edge - center).length
        self.xnpix = xnpix
        self.ynpix = ynpix
        self.pc = center
        self.px = x_edge
        self.py = y_edge

        # Make this a property?
        self.pixel_values = np.zeros((xnpix, ynpix))

    def __str__(self):
        return ('Detector(' + str(self.pc) + ', ' + str(self.px) + ', ' +
                str(self.py) + ', xnpix=' + str(self.xnpix) +
                ', ynpix=' + str(self.ynpix))

    def intersect(self, o):
        # Make this so that it returns a single ray
        
        # Find intersections using base class method
        o = Plane.intersect(self, o)

        if o is None:
            return None
        else:
            # Increment pixels as needed
            i0 = (o.origin - self.pc)
            x0 = (self.px - self.pc).normalize()
            y0 = (self.py - self.pc).normalize()
            xlen = i0.dot(x0)
            ylen = i0.dot(y0)

            # Which pixel should increment?
            if abs(xlen) > self.xwidth/2 or abs(ylen) > self.ywidth/2:
                pass  # None; ray missed the detector
            else:
                xfrac = xlen/self.xwidth
                yfrac = ylen/self.ywidth
                xind = np.round((self.xnpix - 1)*(0.5 + xfrac))
                yind = np.round((self.ynpix - 1)*(0.5 + yfrac))
                self.pixel_values[int(xind), int(yind)] += 1

            return o

    def propagate(self, o):
        return o
        
