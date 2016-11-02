import numpy as np
from random import uniform
from pmst.geometry import Point, Ray, Plane


class Source:
    """A light source consisting of a list of rays."""
    def __init__(self):
        self.rays = []

    def add_ray(self, ray):
        self.rays.append(ray)

    def __str__(self):
        return 'Rays:\t\t' + str(len(self.rays)) + \
            '\n1st ray:\n' + str(self.rays[0]) + '\n'


class DirectedPointSource(Source):
    """A directed point source with n rays propagating from a cone specified 
    by an half-angle 'theta' around a vector 'direction'.

    n_rays = number of uniformly distributed rays
    direction = central direction of ray propagation.
    theta = half-angle of the cone with uniformly distributed rays
      /
    /\  theta
    -------- direction
    \ 
      \
    """

    # TODO: Handle psi > pi
    def __init__(self, origin, n_rays, direction, psi):
        Source.__init__(self)
        self.origin = origin
        self.n_rays = n_rays
        self.direction = direction
        self.psi = psi

    def generate_rays(self):
        for n in range(int(self.n_rays)):
            # Generate random directions on a portion of a sphere
            # See: http://mathworld.wolfram.com/SpherePointPicking.html
            u = uniform(0, 1)
            v = uniform(0, 1)
            theta = 2*np.pi*u
            phi_prime = np.arccos(2*v - 1)  # Uniform over all phi
            phi = phi_prime*(self.psi/np.pi)
            self.add_ray(Ray(self.origin,
                             Point(self.origin.x + np.cos(theta)*np.sin(phi),
                                   self.origin.y + np.sin(theta)*np.sin(phi),
                                   self.origin.y + np.cos(phi))))

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+Source.__str__(self)

    
class IsotropicPointSource(DirectedPointSource):
    """An isotropic point source with n rays propagating from the origin."""
    def __init__(self, origin, n_rays):
        DirectedPointSource.__init__(self, origin, n_rays,
                                     origin + Point(0, 0, 1), np.pi)

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+Source.__str__(self)
