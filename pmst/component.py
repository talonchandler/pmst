import numpy as np
from pycuda.elementwise import ElementwiseKernel

class Lens:
    """ A lens"""
    def __init__(self, origin, n, normal, f=1, radius=1, label=True):
        self.origin = origin
        self.n = n
        self.normal = normal
        self.f = f
        self.radius = radius
        self.label = label
        
    def propagate(self, ray_list):
        prop = ElementwiseKernel(
            '''
            float *x0, float *y0, float *z0,
            float *x1, float *y1, float *z1, 
            float cx, float cy, float cz,
            float nx, float ny, float nz, 
            float radius, float focal
            ''',
            '''
            // Calculate intersection point
            float p0minusl0x = cx - x0[i];
            float p0minusl0y = cy - y0[i];
            float p0minusl0z = cz - z0[i];
            float num = p0minusl0x*nx + p0minusl0y*ny + p0minusl0z*nz;

            float lx = x1[i] - x0[i];
            float ly = y1[i] - y0[i];
            float lz = z1[i] - z0[i];
            float den = lx*nx + ly*ny + lz*nz;

            float d = num/den;
            
            // Intersection point
            float ix = lx*d + x0[i];
            float iy = ly*d + y0[i];
            float iz = lz*d + z0[i];

            // Check if the ray intersects with the lens
            float r = sqrt((ix - cx)*(ix - cx) + (iy - cy)*(iy - cy));

            if(r < radius) {
                x0[i] = lx*d + x0[i];
                y0[i] = ly*d + y0[i];
                z0[i] = lz*d + z0[i];

                x1[i] = x0[i];
                y1[i] = y0[i];
                z1[i] = z0[i] + 1;
            }
            ''',
            "prop")

        rays = ray_list.ray_list
        
        prop(rays[0], rays[1], rays[2],
             rays[3], rays[4], rays[5],
             self.origin.x, self.origin.y, self.origin.z,
             self.normal.x, self.normal.y, self.normal.z,
             self.radius, self.f)
        
        return ray_list, None

    def schematic(self, ax):
        from matplotlib.path import Path
        import matplotlib.patches as patches

        verts = [
            (self.origin.x - self.radius, self.origin.z),
            (self.origin.x - self.radius/2, self.origin.z + self.radius/6),
            (self.origin.x + self.radius/2, self.origin.z + self.radius/6),
            (self.origin.x + self.radius, self.origin.z),
            (self.origin.x + self.radius/2, self.origin.z - self.radius/6),
            (self.origin.x - self.radius/2, self.origin.z - self.radius/6),
            (self.origin.x - self.radius, self.origin.z)
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,            
        ]
        
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1)
        ax.add_patch(patch)

        if label:
            ax.text(x=self.origin.x + 7,
                    y=self.origin.z,
                    s=r'$\mathrm{Lens}$',
                    ha='left',
                    va='center',
                    size=8)


