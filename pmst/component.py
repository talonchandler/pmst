import numpy as np


class Component:
    """A microcope component (everything other than the source)"""
    def __init__(self, origin, n, absorb_frac, geometry=None):
        self.origin = origin
        self.n = n
        self.absorb_frac = absorb_frac
        self.count = 0
        self.geometry = geometry

    def __str__(self):
        return 'Origin:\t\t' + str(self.origin) + \
            '\nn:\t\t' + str(self.n) + \
            '\nAbsorb frac:\t' + str(self.absorb_frac) + \
            '\nCount:\t\t'+str(self.count)+'\n'


class Pixel(Component):
    """A single detector element"""
    def __init__(self, origin, normal, dimensions):
        Component.__init__(self, origin, 0, 1, geometry='plane')  # Completely absorptive
        self.normal = normal
        self.dimensions = dimensions

    def __str__(self):
        return Component.__str__(self)+'\nNormal:\t\t' + str(self.normal) + \
            '\nDimensions:\t' + str(self.dimensions[0]) + ' x ' + str(self.dimensions[1])


    

