import time; start = time.time();
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import uniform
from pmst.geometry import Point, Ray, Plane, util

# Load gpu
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

class RayList:
    """A wrapper class for the gpuarray that represents rays. Provides printing 
    and debugging options."""
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.ray_list = (x0, y0, z0, x1, y1, z1)
        
    def __str__(self):
        s = "Ray List:\n"
        for i in range(len(self.ray_list[0])):
            s = s + str(self.get_ray(i)) + '\n'
            
        return s
    
    def get_ray(self, i):
        r = self.ray_list
        z = []
        for j in range(len(r)):
            z.append(float(r[j][i].get())) # Convert gpuarray to float

        return Ray(Point(z[0], z[1], z[2]),
                   Point(z[3], z[4], z[5]))
                   
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

    def __init__(self,
                 origin=Point(0, 0, 0),
                 n_rays=1,
                 direction=Point(0, 0, 1),
                 psi=np.pi/2):
        
        self.n_rays = int(n_rays)
        self.origin = origin
        self.direction = direction
        self.psi = np.float64(psi)

    def generate_rays(self):
        # Generate rays
        n = self.n_rays

        ## Origin
        x0 = gpuarray.zeros(n, np.float64)
        y0 = gpuarray.zeros(n, np.float64)
        z0 = gpuarray.zeros(n, np.float64)
        x0.fill(self.origin.x)
        y0.fill(self.origin.y)
        z0.fill(self.origin.z)

        ## Directions
        x1 = gpuarray.zeros(n, np.float64)
        y1 = gpuarray.zeros(n, np.float64)
        z1 = gpuarray.zeros(n, np.float64)

        u1 = curandom.rand((n,), dtype=np.float64, stream=None)
        u2 = curandom.rand((n,), dtype=np.float64, stream=None)

        # TODO: Handle non-zero origin
        # See http://mathworld.wolfram.com/SpherePointPicking.html
        # See http://math.stackexchange.com/a/205589/357869
        mod = SourceModule(util + """
        __global__ void propagate(
            double *x0, double *y0, double *z0,
            double *x1, double *y1, double *z1, 
            double *u1, double *u2, double psi,
            double dx, double dy, double dz
        )
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;

            Point r0 = {x0[i], y0[i], z0[i]};
            Point r1 = {x1[i], y1[i], z1[i]};
            Point d = {dx, dy, dz};

            // Calculate the output theta value U(0, 2*PI)
            double theta = 2*PI*u1[i];

            // Calculate the output z value U(cos(psi), 1)
            double zi = z0[i] + (1-cos(psi))*u2[i] + cos(psi);

            // Calculate the output x and y values
            double xi = x0[i] + sqrt(1 - pow(zi, 2))*cos(theta);
            double yi = y0[i] + sqrt(1 - pow(zi, 2))*sin(theta);

            //// Rotate the output values to the correct direction
            // Normalize direction
            double rd = len(d);
            dx = dx/rd;
            dy = dy/rd;
            dz = dz/rd;

            // Calculate the rotation axis (cross product of direction and (0, 0, 1))
            double rr = sqrt(pow(dx, 2) + pow(dy, 2));
            double xr = dy/rr;
            double yr = -dx/rr;
            double zr = 0;            
            if(rr == 0){
                xr = 0;
                yr = 0;
            }

            // Calculate the rotation angle (arccos of dot product of direction and (0, 0, 1))
            double a = acos(dz);

            // Generate the rotation matrix about the axis using a matrix multiplication
            double r11 = cos(a) + pow(xr, 2)*(1 - cos(a));
            double r12 = xr*yr*(1 - cos(a)) - zr*sin(a);
            double r13 = xr*zr*(1 - cos(a)) + yr*sin(a);
            double r21 = yr*xr*(1 - cos(a)) + zr*sin(a);
            double r22 = cos(a) + pow(yr, 2)*(1 - cos(a));
            double r23 = yr*zr*(1 - cos(a)) - xr*sin(a);
            double r31 = zr*xr*(1 - cos(a)) - yr*sin(a);
            double r32 = zr*yr*(1 - cos(a)) + xr*sin(a);
            double r33 = cos(a) + pow(zr, 2)*(1 - cos(a));

            // Apply the rotation matrix to the points
            x1[i] = r11*xi + r12*yi + r13*zi;
            y1[i] = r21*xi + r22*yi + r23*zi;
            z1[i] = r31*xi + r32*yi + r33*zi;
        }
        """)


        prop = mod.get_function("propagate")

        prop(x0, y0, z0, x1, y1, z1, u1, u2, self.psi,
             self.direction.x, self.direction.y, self.direction.z,
             block=(512, 1, 1), grid=(int(n/512), 1))

        self.ray_list = RayList(x0, y0, z0, x1, y1, z1)

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+'Rays:\t\t' + str(len(self.ray_list[0]))

    def schematic(self, ax):
        ax.add_artist(plt.Circle((self.origin.x, self.origin.z),
                                  0.1, color='k'))
        ax.text(x=self.origin.x + 7, y=self.origin.z,
                 s='$\mathrm{Source}$', ha='left', va='center', size=8)
    
class IsotropicPointSource(DirectedPointSource):
    """An isotropic point source with n rays propagating from the origin."""
    def __init__(self, origin, n_rays):
        DirectedPointSource.__init__(self, origin, n_rays,
                                     origin + Point(0, 0, 1), np.pi)

    def __str__(self):
        return DirectedPointSource.__str__(self)

class PlanarSource:
    """A planar source with n rays
    """

    def __init__(self,
                 origin=Point(0, 0, 0),
                 n_rays=1,
                 direction=Point(0, 0, 1),
                 xedge=Point(1, 0, 1),
                 yedge=Point(0, 1, 0)):
        
        self.n_rays = n_rays
        self.origin = origin
        self.direction = direction
        self.xedge = xedge
        self.yedge = yedge

    def generate_rays(self):
        # Generate rays
        n = self.n_rays

        ## Origin
        x0 = gpuarray.zeros(n, np.float64)
        y0 = gpuarray.zeros(n, np.float64)
        z0 = gpuarray.zeros(n, np.float64)

        ## Directions
        x1 = gpuarray.zeros(n, np.float64)
        y1 = gpuarray.zeros(n, np.float64)
        z1 = gpuarray.zeros(n, np.float64)

        u1 = curandom.rand((n,), dtype=np.float64, stream=None)
        u2 = curandom.rand((n,), dtype=np.float64, stream=None)

        calc_dir = ElementwiseKernel(
            '''
            double *x0, double *y0, double *z0, 
            double *x1, double *y1, double *z1, 
            double *u1, double *u2, 
            double xd, double yd, double zd
            ''',
            '''
            // Calculate the output theta value U(0, 2*PI)
            double x = 2*u1[i] - 1;
            double y = 2*u2[i] - 1;

            x0[i] = x;
            y0[i] = y;
            z0[i] = 0;
            x1[i] = x;
            y1[i] = y;
            z1[i] = 1;

            ''',
            "calc_dir")
        calc_dir(x0, y0, z0, x1, y1, z1, u1, u2, 
                 self.direction.x, self.direction.y, self.direction.z)

        self.ray_list = RayList(x0, y0, z0, x1, y1, z1)

    def __str__(self):
        return 'Source O:\t'+str(self.origin)+'\n'+'Rays:\t\t' + str(len(self.ray_list[0]))

    def schematic(self, ax):
        line = plt.Line2D((-1, 1),
                          (0, 0), 
                          color='k',        
                          ms=0)             
        ax.add_artist(line)                        
        #ax.add_artist(plt.Circle((self.origin.x, self.origin.z),
        #                          0.1, color='k'))
        ax.text(x=self.origin.x + 7, y=self.origin.z,
                 s='$\mathrm{Coll.\ Planar\ Source}$', ha='left', va='center', size=8)

    
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

        x0 = gpuarray.to_gpu(np.array(x0, np.float64))
        y0 = gpuarray.to_gpu(np.array(y0, np.float64))
        z0 = gpuarray.to_gpu(np.array(z0, np.float64))       
        x1 = gpuarray.to_gpu(np.array(x1, np.float64))
        y1 = gpuarray.to_gpu(np.array(y1, np.float64))
        z1 = gpuarray.to_gpu(np.array(z1, np.float64))        
            
        self.ray_list = RayList(x0, y0, z0, x1, y1, z1)

        
