from pmst.detector import Detector
from functools import reduce
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Microscope:
    """A microscope""" 
    def __init__(self, source):
        self.component_list = []

    def add_component(self, component):
        self.component_list.append(component)

    def simulate(self):
        # Generate rays
        self.source.generate_rays()
        
        # Populate function list (return bound methods)
        # Alternate intersection and propagation functions
        func_list = []
        for component in self.component_list:
            func_list.append(component.intersect)
            func_list.append(component.propagate)

        # Run each function on each ray
        [reduce(lambda v, f: f(v), func_list, ray) for ray in self.source.rays]

    def simulate_gpu(self):
        # Load gpu
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import pycuda.cumath as cumath
        from pycuda.curandom import rand as curand
        from pycuda.elementwise import ElementwiseKernel

        # Generate rays
        n = self.source.n_rays
        n = int(1e7)
        ## Origin
        x0 = gpuarray.empty(n, np.float32)
        y0 = gpuarray.empty(n, np.float32)
        z0 = gpuarray.empty(n, np.float32)
        x0.fill(self.source.origin.x)
        y0.fill(self.source.origin.y)
        z0.fill(self.source.origin.z)
        
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

        # Populate function list (return bound methods)

        # Detector
        cx = gpuarray.empty(n, np.float32)
        cy = gpuarray.empty(n, np.float32)
        cz = gpuarray.empty(n, np.float32)
        cx.fill(self.component_list[0].p1.x)
        cy.fill(self.component_list[0].p1.y)
        cz.fill(self.component_list[0].p1.z)

        nx = gpuarray.empty(n, np.float32)
        ny = gpuarray.empty(n, np.float32)
        nz = gpuarray.empty(n, np.float32)
        nx.fill(self.component_list[0].normal.x)
        ny.fill(self.component_list[0].normal.y)
        nz.fill(self.component_list[0].normal.z)
        
        intersect = ElementwiseKernel(
            '''
            float *x0, float *y0, float *z0,
            float *x1, float *y1, float *z1,
            float *cx, float *cy, float *cz,
            float *nx, float *ny, float *nz
            ''',
            '''
            float lx = x1[i] - x0[i];
            float ly = y1[i] - y0[i];
            float lz = z1[i] - z0[i];
            float p0minl0x = cx[0] - x0[i];
            float p0minl0y = cy[0] - y0[i];
            float p0minl0z = cz[0] - z0[i];
            float num = p0minl0x*nx[i] + p0minl0y*ny[i] + p0minl0z*nz[i];
            float den = lx*nx[i] + ly*ny[i] + lz*nz[i];
            float d = num/den;
            x0[i] += lx*d;
            y0[i] += ly*d;
            z0[i] += lz*d;
            x1[i] += x0[i];
            y1[i] += y0[i];
            z1[i] += z0[i];
            ''',
            "intersect")

        intersect(x0, y0, z0, x1, y1, z1, cx, cy, cz, nx, ny, nz)
        
        # Run each function on each ray

        # Histogram results (todo for non-z-aligned planes)
        bins = (self.component_list[0].xnpix, self.component_list[0].ynpix)
        bins = (5, 5)
        pxrange = [[-100, 100],[-100, 100]]
        (hist, xedges, yedges) = np.histogram2d(x0.get(), y0.get(), bins=bins, range=pxrange)
        return hist

    def simulate_gpu2(self):
        # Load gpu
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import pycuda.cumath as cumath
        from pycuda.curandom import rand as curand
        from pycuda.elementwise import ElementwiseKernel

        # Generate rays
        self.source.generate_rays()
        
        # Populate function list (return bound methods)
        # Alternate intersection and propagation functions
        func_list = []
        for component in self.component_list:
            func_list.append(component.propagate)

        # Run each function on the ray_list
        detector_values = []
        ray_list = self.source.ray_list
        f = lambda f, v: f(v)
        for func in func_list:
            ray_list, pixel_values = f(func, ray_list)
            detector_values.append(pixel_values)


        # reduce(lambda v, f: f(v), func_list, self.source.ray_list)
        
        print(ray_list)
        print(detector_values)        
        
        
    
    def plot_results(self, filename, src='', dpi=300):
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(11, 8))

        # Plot source
        ax2.add_artist(plt.Circle((self.source.origin.x, self.source.origin.z),
                                  0.1, color='k'))
        ax2.text(x=self.source.origin.x + 7, y=self.source.origin.z,
                 s='Source', ha='left', va='center', size=8)

        # Plot each component
        for c in self.component_list:
            c = self.component_list[1]  # temporary choose first
            print(c)
            print(np.max(c.pixel_values))
            if isinstance(c, Detector):
                # Plot data
                px = c.pixel_values
                ax1.imshow(c.pixel_values, interpolation='none')
                
                x = range(len(px[:, int(px.shape[0]/2)-1]))
                y = px[:, int(px.shape[0]/2)-1]
                y = y/self.source.n_rays
                ax3.step(x, y, where='mid', markersize=0, color='k')

                # Calculate true
                def sa(a, b, d):
                    alpha = a/(2*d)
                    beta = b/(2*d)
                    omega = 4*np.arccos(np.sqrt((1+alpha**2+beta**2)/((1+alpha**2)*(1+beta**2))))
                    return omega

                def off_frac(A, B, a, b, d):
                    t1 = sa(2*(A+a), 2*(B+b), d)
                    t2 = sa(2*A, 2*(B+b), d)
                    t3 = sa(2*(A+a), 2*B, d)
                    t4 = sa(2*A, 2*B, d)
                    sa_off = (t1 - t2 - t3 + t4)/4
                    return np.abs(sa_off/(4*np.pi))

                pixel_width = c.xwidth/c.xnpix
                yfit = [8*off_frac((i-50)*pixel_width, 0, pixel_width, pixel_width, 2) for i in x]
                #print(off_frac(10*pixel_width, 0, pixel_width, pixel_width, 2))
                #print("C_FRAC:", off_frac(0, 0, 1, 1, .0001))
                #xfit = np.arange(0, 100, 0.1)
                #yfit = 4/(4+0.05*(xfit-50)**2)
                ax3.step(x, yfit, where='mid', markersize=0, color='r')

                line = plt.Line2D((-c.px.x, c.px.x),
                                  (c.px.z, c.px.z),
                                  color='k',
                                  ms=0)

                ax2.add_artist(line)
                ax2.text(x=c.px.x + 2,
                         y=c.px.z,
                         s='Detector',
                         ha='left',
                         va='center',
                         size=8)
            else:
                pass
        
        ax0.set_xlim([0, 10])
        ax0.set_ylim([-10, 0])

        src = ''.join(src)
        src = src.replace('\n', '\\\\')
        src = src.replace('_', '\_')
        src = '\\texttt{\\noindent \\\\' + src + '}'

        ax0.text(x=5, y=-5, s=src, ha='center', va='center', size=6)
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-10, 10])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax2.set_aspect(aspect=1, adjustable='box')

        #ax3.set_aspect(aspect=100, adjustable='box')
        f.savefig(filename, dpi=dpi)
        
    def __str__(self):
        return 'Components:\t'+str(len(self.component_list))+'\n'


