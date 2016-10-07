import numpy as np
from random import uniform

class Ray:
    """A polarized light ray."""
    def __init__(self, origin, direction, polarization=None):
        self.origin = origin
        self.direction = direction
        self.polarization = polarization

    def propagate(self, object):
        # Find intersection between ray and object
        # Update ray origin, direction, polarization
        return 1

    def __str__(self):
        return 'O:\t'+str(self.origin)+\
            '\nD:\t'+str(self.direction)+\
            '\nP:\t'+str(self.polarization)+'\n'


class Source:
    """A light source consisting of a list of rays."""
    def __init__(self):
        self.rays = []

    def add_ray(self, ray):
        self.rays.append(ray)

    def __str__(self):
        return 'Rays:\t'+str(len(self.rays))+\
            '\n1st ray:\n'+str(self.rays[0])+'\n'

        
class IsotropicPointSource(Source):
    """An isotropic point source with n rays propagating from the origin."""
    def __init__(self, origin, n_rays):
        Source.__init__(self)
        self.origin = origin
        self.n_rays = n_rays

        for n in range(int(n_rays)):
            # Generate random directions on a sphere
            # See: http://mathworld.wolfram.com/SpherePointPicking.html
            u = uniform(0, 1)
            v = uniform(0, 1)
            theta = 2*np.pi*u
            phi = np.arccos(2*v - 1)
            self.add_ray(Ray(origin, np.array((theta, phi))))
    
    def __str__(self):
        return 'O:\t'+str(self.origin)+'\n'+Source.__str__(self)

