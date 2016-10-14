from pmst.geometry import Point, Ray, Plane
import numpy as np

class Detector(Plane):
    """Plane detector with square pixels"""
    
    def __init__(self, p1, p2, p3, xnpix, ynpix):
        Plane.__init__(self, p1, p2, p3)
        self.xextent = 2*(p2 - p1).length
        self.yextent = 2*(p3 - p1).length
        self.xnpix = xnpix
        self.ynpix = ynpix
        self.pc = p1
        self.px = p2
        self.py = p3

        # Make this a property?
        self.pixel_values = np.zeros((xnpix, ynpix))

    def intersection(self, o):
        i = Plane.intersection(self, o)

        # TODO: Find which pixel to increment
        if i != []:
            i0 = (i[0] - self.pc)
            x0 = (self.px - self.pc).normalize()
            y0 = (self.py - self.pc).normalize()
            xlen = i0.dot(x0)
            ylen = i0.dot(y0)

            # TODO: Clarify xextent (half or full width)
            if abs(xlen) > self.xextent/2 or abs(ylen) > self.yextent/2:
                pass
            else:
                xfrac = xlen/self.xextent
                yfrac = ylen/self.yextent
                xind = np.round((self.xnpix - 1)*xfrac + (self.xnpix/2))
                yind = np.round((self.ynpix - 1)*yfrac + (self.ynpix/2))
                self.pixel_values[int(xind), int(yind)] += 1
            
        return i
        
