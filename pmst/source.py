import numpy as np
from random import uniform
from pmst.geometry import Point, Ray, Plane


# Revisit this class later. Inherit from Ray3D.
# class PolRay3D:
#     """A polarized light ray."""
#     def __init__(self, origin, direction, polarization=None):
#         self.origin = origin
#         self.direction = direction
#         self.polarization = polarization

#     def propagate(self, component_list):
#         #for component in component_list:
#             # Naively go through components one at a time. Throw out ray if
#             # it doesn't intersect the first item in the component list.
#             #
#             # TO DO: Find intersection distance to all components, propagate to
#             # the closest one, then repeat until the photon is absorbed or the
#             # photon goes to inf.
                
#         return 1

#     def __str__(self):
#         return 'Origin:\t\t' + str(self.origin) + \
#             '\nDirection:\t' + str(self.direction) + \
#             '\nPolarization:\t'+str(self.polarization)+'\n'


class Source:
    """A light source consisting of a list of rays."""
    def __init__(self):
        self.rays = []

    def add_ray(self, ray):
        self.rays.append(ray)

    def __str__(self):
        return 'Rays:\t\t' + str(len(self.rays)) + \
            '\n1st ray:\n' + str(self.rays[0]) + '\n'

        
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
            self.add_ray(Ray(self.origin, Point(np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi))))

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+Source.__str__(self)

