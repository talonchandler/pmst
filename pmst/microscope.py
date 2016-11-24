from pmst.detector import Detector
from functools import reduce
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=False)

class Microscope:
    """A microscope""" 
    def __init__(self, source):
        self.component_list = []
        self.source = source

    def add_component(self, component):
        self.component_list.append(component)

    def simulate(self):
        # Generate rays
        self.source.generate_rays()
        
        # Populate function list (return bound methods)
        func_list = []
        for component in self.component_list:
            func_list.append(component.propagate)

        # Run each function on the ray_list
        ray_list = self.source.ray_list
        f = lambda f, v: f(v)

        for i, func in enumerate(func_list):
            ray_list, pixel_values = f(func, ray_list)
            if pixel_values is not None:
                self.component_list[i].pixel_values = pixel_values
    
    def plot_results(self, filename, src='', dpi=300):
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(11, 8))

        
        # Plot source
        ax2.add_artist(plt.Circle((self.source.origin.x, self.source.origin.z),
                                  0.1, color='k'))
        ax2.text(x=self.source.origin.x + 7, y=self.source.origin.z,
                 s='$\mathrm{Source}$', ha='left', va='center', size=8)

        # Plot each component
        for c in self.component_list:
            if isinstance(c, Detector):
                # Plot data
                px = c.pixel_values
                ax1.set_xlim(0, c.xnpix)
                ax1.set_ylim(0, c.ynpix)                
                ax1.imshow(c.pixel_values, interpolation='none', cmap='Greys_r')
                ax1.set_aspect(aspect=1./ax1.get_data_ratio(), adjustable='box-forced')
                
                x = range(len(px[:, int(px.shape[0]/2)-1]))
                y = px[:, int(px.shape[0]/2)-1]
                y = y/self.source.n_rays
                ax3.set_xlim(0, c.xnpix)
                ax3.step(x, y, where='mid', markersize=0, color='k')
                ax3.set_aspect(aspect=1./ax3.get_data_ratio(), adjustable='box-forced')
                
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
                # ax3.step(x, yfit, where='mid', markersize=0, color='r')

                line = plt.Line2D((-c.px.x, c.px.x),
                                  (c.px.z, c.px.z),
                                  color='k',
                                  ms=0)

                ax2.add_artist(line)
                ax2.text(x=c.px.x + 5,
                         y=c.px.z,
                         s=r'$\mathrm{Detector}$',
                         ha='left',
                         va='center',
                         size=8)
            else:
                c.schematic(ax2)
        
        ax0.set_xlim([0, 10])
        ax0.set_ylim([-10, 0])

        src = ''.join(src)
        from matplotlib.font_manager import FontProperties
        font = FontProperties()
        font.set_family('monospace')
        font.set_size('xx-small')        
        ax0.text(x=-2, y=-5, s=src, ha='left', va='center', linespacing=1.2, fontproperties=font)
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)

        xlab = ax1.get_xticks().tolist()
        xlab = ['$\mathrm{' + str(int(lab)) + '}$' for lab in xlab]
        ax1.set_xticklabels(xlab)
        ylab = ax1.get_yticks().tolist()
        ylab = ['$\mathrm{' + str(int(lab)) + '}$' for lab in ylab]
        ax1.set_yticklabels(ylab)
        
        
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-10, 10])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        def latex_float(f):
            float_str = "{0:.0e}".format(f)
            if f == 0:
                return "0"
            elif "e" in float_str:
                base, exponent = float_str.split("e")
                return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
            else:
                return float_str

        
        xlab = ax3.get_xticks().tolist()
        xlab = ['$\mathrm{' + str(int(lab)) + '}$' for lab in xlab]
        ax3.set_xticklabels(xlab)
        ylab = ax3.get_yticks().tolist()
        ylab = ['$\mathrm{' + latex_float(lab) + '}$' for lab in ylab]
        ax3.set_yticklabels(ylab)

        
        f.savefig(filename, dpi=dpi)
        
    def __str__(self):
        return 'Components:\t'+str(len(self.component_list))+'\n'


