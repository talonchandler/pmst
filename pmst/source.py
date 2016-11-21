import numpy as np
from random import uniform
from pmst.geometry import Point, Ray, Plane

# Load gpu
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.curandom import rand as curand
from pycuda.elementwise import ElementwiseKernel


class DirectedPointSource:
    """A directed point source with n rays propagating from a cone specified 
    by an half-angle 'theta' around a vector 'direction'.

    origin = Point specifying the starting point of the rays
    n_rays = number of uniformly distributed rays
    direction = Point specifying the central direction of ray propagation.
    psi = half-angle of the cone with uniformly distributed rays
            /
           /\  psi
    origin -------- direction
           \ 
            \
    """

    def __init__(self, origin, n_rays, direction, psi):
        self.n_rays = n_rays
        self.origin = origin
        self.direction = direction
        self.psi = psi

    def generate_rays(self):
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
        # TODO: Handle psi > pi
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
        
        calc_dir(x0, y0, z0, x1, y1, z1, u1, u2, self.psi, np.pi)

        self.ray_list = (x0, y0, z0, x1, y1, z1)

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+'Rays:\t\t' + str(len(self.ray_list[0]))
    
    
class IsotropicPointSource(DirectedPointSource):
    """An isotropic point source with n rays propagating from the origin."""
    def __init__(self, origin, n_rays):
        DirectedPointSource.__init__(self, origin, n_rays,
                                     origin + Point(0, 0, 1), np.pi)

    def __str__(self):
        return DirectedPointSource.__str__(self)

class RayListSource:
    """A source specified by a list of rays."""
    def __init__(self, input_ray_list):
        self.input_ray_list = input_ray_list
        self.n_rays = len(input_ray_list)

    def generate_rays(self):
        n = self.n_rays

        # Converting input_ray_list into gpu ray_list
        x0 = []
        y0 = []
        z0 = []
        x1 = []
        y1 = []
        z1 = []
        
        for ray in self.input_ray_list:
            x0.append(ray.origin.x)
            y0.append(ray.origin.y)
            z0.append(ray.origin.z)
            x1.append(ray.direction.x)
            y1.append(ray.direction.y)
            z1.append(ray.direction.z)

        x0 = gpuarray.to_gpu(np.array(x0, np.float32))
        y0 = gpuarray.to_gpu(np.array(y0, np.float32))
        z0 = gpuarray.to_gpu(np.array(z0, np.float32))       
        x1 = gpuarray.to_gpu(np.array(x1, np.float32))
        y1 = gpuarray.to_gpu(np.array(y1, np.float32))
        z1 = gpuarray.to_gpu(np.array(z1, np.float32))        
            
        self.ray_list = (x0, y0, z0, x1, y1, z1)
