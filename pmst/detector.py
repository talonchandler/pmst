from pmst.geometry import Point, Ray, Plane
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pmst.geometry import util

from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

class Detector:
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

    def propagate(self, ray_list):

        mod = SourceModule(util + """
        __global__ void propagate(
            double *x0, double *y0, double *z0,
            double *x1, double *y1, double *z1, 
            double cx, double cy, double cz,
            double nx, double ny, double nz
        )
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;

            Point r0 = {x0[i], y0[i], z0[i]};
            Point r1 = {x1[i], y1[i], z1[i]};
            Point c0 = {cx, cy, cz};
            Point n = {nx, ny, nz};
            Point l = subtract(r1, r0);
            double d = dot(subtract(c0, r0), n)/dot(l, n);
            Point i0 = add(scale(l, d), r0);  // Intersection point

            // New ray origin is at the intersection point
            x0[i] = i0.x;
            y0[i] = i0.y;
            z0[i] = i0.z;

            // Calculate new ray direction
            x1[i] = x0[i] + x1[i];
            y1[i] = y0[i] + y1[i];
            z1[i] = z0[i] + z1[i];
        }
        """
        )
        prop = mod.get_function("propagate")
        rays = ray_list.ray_list
        
        prop(rays[0], rays[1], rays[2],
             rays[3], rays[4], rays[5],
             self.pc.x, self.pc.y, self.pc.z,
             self.normal.x, self.normal.y, self.normal.z, 
             block=(512, 1, 1), grid=(int(len(rays[0])/512), 1))
        
        # Compute histogram results (todo for non-z-aligned planes)
        xedges = np.linspace(self.pc.x - self.px.x, self.pc.x + self.px.x, self.xnpix + 1)
        yedges = np.linspace(self.pc.y - self.py.y, self.pc.y + self.py.y, self.ynpix + 1)
        (hist, xedges, yedges) = np.histogram2d(rays[0].get(), rays[1].get(), bins=(xedges, yedges))
        
        return ray_list, hist

    def schematic(self, ax):
         # Plot data                         
         line = plt.Line2D((-self.px.x, self.px.x),
                           (self.px.z, self.px.z), 
                           color='k',        
                           ms=0)             

         ax.add_artist(line)                
         ax.text(x=self.px.x + 5,              
                  y=self.px.z,                  
                  s=r'$\mathrm{Detector}$',  
                  ha='left',                 
                  va='center',               
                  size=8)
         
