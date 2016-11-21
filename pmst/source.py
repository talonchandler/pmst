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

class DirectedPointSourceGPU(Source):
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
        self.n_rays = int(n_rays)        
        self.origin = origin
        self.direction = direction
        self.psi = psi

    def generate_rays(self):
        # Load gpu
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import pycuda.cumath as cumath
        from pycuda.curandom import rand as curand
        from pycuda.elementwise import ElementwiseKernel

        # Generate rays
        n = self.n_rays

        ## Origin
        x0 = gpuarray.empty(n, np.float32)
        y0 = gpuarray.empty(n, np.float32)
        z0 = gpuarray.empty(n, np.float32)
        x0.fill(self.origin.x)
        y0.fill(self.origin.y)
        z0.fill(self.origin.z)
        
        ## Directions
        x1 = gpuarray.zeros(n, np.float32)
        y1 = gpuarray.zeros(n, np.float32)
        z1 = gpuarray.zeros(n, np.float32)
        u1 = curand((n,))
        u2 = curand((n,))
        out = gpuarray.zeros(n, np.float32)
        calc_dir = ElementwiseKernel(
            '''
            float *x0, float *y0, float *z0, 
            float *x1, float *y1, float *z1, 
            float *u1, float *u2, 
            float psi, float pi
            ''',
            '''
            float theta;
            float phi_prime;
            float phi;
            theta = 2*pi*u1[i];
            phi_prime = acos(2*u2[i] -1);
            phi = phi_prime*psi/pi;
            x1[i] = x0[i] + cos(theta)*sin(phi);
            y1[i] = y0[i] + sin(theta)*sin(phi);
            z1[i] = z0[i] + cos(phi);
            ''',
            "calc_dir")
        
        calc_dir(x0, y0, z0, x1, y1, z1, u1, u2, 3.0, np.pi)

        self.ray_list = (x0, y0, z0, x1, y1, z1)

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+Source.__str__(self)
    
    
class IsotropicPointSource(DirectedPointSource):
    """An isotropic point source with n rays propagating from the origin."""
    def __init__(self, origin, n_rays):
        DirectedPointSource.__init__(self, origin, n_rays,
                                     origin + Point(0, 0, 1), np.pi)

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+Source.__str__(self)
