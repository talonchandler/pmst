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
        Point c = {cx, cy, cz};
        Point n = {nx, ny, nz};

        Ray ri = {r0, r1};
        Plane p = {n, c};
        Point ro = intersect(ri, p);

        // Check if the ray intersects with the lens
        double r = len(subtract(ro, c));
        if(r < radius) {

            // Calculate incoming ray angles
            Point a = subtract(n, c);
            Point l = subtract(r1, r0);
            double phi_i = acos(dot(a, l)/(len(a)*len(l)));
            double theta0_i = atan2(y1[i], x1[i]);

            double theta1_i = atan2(y1[i], x1[i]);

            // Transform angles
            double phi_o = -r*cos(theta1_i - theta0_i)/f + phi_i;
            double theta0_o = theta0_i;

            // New ray origin is at the intersection point
            x0[i] = ro.x;
            y0[i] = ro.y;
            z0[i] = ro.z;

            // Calculate new ray direction
            x1[i] = x0[i] + cos(theta0_o)*sin(phi_o);
            y1[i] = y0[i] + sin(theta0_o)*sin(phi_o);
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


