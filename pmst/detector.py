from pmst.geometry import Point, Ray, Plane
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pycuda.elementwise import ElementwiseKernel

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

        prop = ElementwiseKernel(
            '''
            float *x0, float *y0, float *z0,
            float *x1, float *y1, float *z1, 
            float cx, float cy, float cz,
            float nx, float ny, float nz
            ''',
            '''
            float p0minusl0x = cx - x0[i];
            float p0minusl0y = cy - y0[i];
            float p0minusl0z = cz - z0[i];
            float num = p0minusl0x*nx + p0minusl0y*ny + p0minusl0z*nz;

            float lx = x1[i] - x0[i];
            float ly = y1[i] - y0[i];
            float lz = z1[i] - z0[i];
            float den = lx*nx + ly*ny + lz*nz;

            float d = num/den;

            x0[i] = lx*d + x0[i];
            y0[i] = ly*d + y0[i];
            z0[i] = lz*d + z0[i];

            x1[i] = x0[i] + x1[i];
            y1[i] = y0[i] + y1[i];
            z1[i] = z0[i] + z1[i];
            ''',
            "prop")

        prop(ray_list[0], ray_list[1], ray_list[2],
             ray_list[3], ray_list[4], ray_list[5],
             self.pc.x, self.pc.y, self.pc.z,
             self.normal.x, self.normal.y, self.normal.z)
        
        # Compute histogram results (todo for non-z-aligned planes)
        xedges = np.linspace(self.pc.x - self.px.x, self.pc.x + self.px.x, self.xnpix + 1)
        yedges = np.linspace(self.pc.y - self.py.y, self.pc.y + self.py.y, self.ynpix + 1)
        (hist, xedges, yedges) = np.histogram2d(ray_list[0].get(), ray_list[1].get(), bins=(xedges, yedges))

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
         
