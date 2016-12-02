import numpy as np
from pmst.geometry import util
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

class Lens:
    """ A lens"""
    def __init__(self, origin, n, normal, f=1, radius=1, label=True):
        self.origin = origin
        self.n = n
        self.normal = normal
        self.f = np.float64(f)
        self.radius = np.float64(radius)
        self.label = label

    def propagate(self, ray_list):


        mod = SourceModule(util + """
        __global__ void propagate(
            double *x0, double *y0, double *z0,
            double *x1, double *y1, double *z1, 
            double cx, double cy, double cz,
            double nx, double ny, double nz, 
            double radius, double f
        )
        {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        // Calculate intersection point
        Point r0 = {x0[i], y0[i], z0[i]};
        Point r1 = {x1[i], y1[i], z1[i]};
        Point c0 = {cx, cy, cz};
        Point n = {nx, ny, nz};
        Point l = subtract(r1, r0);
        double d = dot(subtract(c0, r0), n)/dot(l, n);
        Point i0 = add(scale(l, d), r0);  // Intersection point

        // Check if the ray intersects with the lens
        double r = len(subtract(i0, c0));
        if(r < radius) {

            // Calculate incoming ray angles
            Point a = subtract(n, c0);
            double phi_i = acos(dot(a, l)/(len(a)*len(l)));
            double theta_i = atan2(y1[i], x1[i]);

            // Transform angles
            double phi_o = -r/f + phi_i;
            double theta_o = theta_i;

            // New ray origin is at the intersection point
            x0[i] = i0.x;
            y0[i] = i0.y;
            z0[i] = i0.z;

            // Calculate new ray direction
            x1[i] = x0[i] + cos(theta_o)*sin(phi_o);
            y1[i] = y0[i] + sin(theta_o)*sin(phi_o);
            z1[i] = z0[i] + cos(phi_o);
        }
        }
        """
        )

        prop = mod.get_function("propagate")

        rays = ray_list.ray_list

        prop(rays[0], rays[1], rays[2],
             rays[3], rays[4], rays[5],
             self.origin.x, self.origin.y, self.origin.z,
             self.normal.x, self.normal.y, self.normal.z,
             self.radius, self.f,
             block=(512, 1, 1), grid=(int(len(rays[0])/512), 1))
        
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

        if self.label:
            ax.text(x=self.origin.x + 7,
                    y=self.origin.z,
                    s=r'$\mathrm{Lens}$',
                    ha='left',
                    va='center',
                    size=8)


