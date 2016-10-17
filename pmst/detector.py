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
    
    def __init__(self, p1, p2=None, p3=None, xnpix=1, ynpix=1):
        Plane.__init__(self, p1, p2, p3)
        self.xwidth = 2*(p2 - p1).length
        self.ywidth = 2*(p3 - p1).length
        self.xnpix = xnpix
        self.ynpix = ynpix
        self.pc = p1
        self.px = p2
        self.py = p3

        # Make this a property?
        self.pixel_values = np.zeros((xnpix, ynpix))

    def intersection(self, o):
        # Find intersections using base class method
        i = Plane.intersection(self, o)

        if len(i) == 1:
            i0 = (i[0] - self.pc)
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
        elif len(i) == 2:
            assert("Warning: Ray intersects the detector twice!")
        else:
            pass
        
        return i
        
