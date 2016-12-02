import time;
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
        start = time.time();
        self.source.generate_rays()
        print("Allocate:\t", np.round(time.time() - start, 1), 's'); start = time.time()
        
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

        print("Propagate:\t", np.round(time.time() - start, 1), 's');
        
    def plot_results(self, filename, src='', fit=None, dpi=300):
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(11, 8))

        # Plot source code
        src = ''.join(src)
        from matplotlib.font_manager import FontProperties
        font = FontProperties()
        font.set_family('monospace')
        font.set_size('xx-small')        
        ax0.text(x=-10, y=0, s=src, ha='left', va='center',
                 linespacing=1.2, fontproperties=font)

        # Plot schematic
        self.source.schematic(ax2)
        for c in self.component_list:
            c.schematic(ax2)            

        for ax in [ax0, ax2]:
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Plot detector image
        ax1.set_xlim(0, c.xnpix)
        ax1.set_ylim(0, c.ynpix)                
        ax1.imshow(c.pixel_values, interpolation='none', cmap='Greys_r')
        ax1.set_aspect(aspect=1./ax1.get_data_ratio(), adjustable='box-forced')

        xlab = ax1.get_xticks().tolist()
        xlab = ['$\mathrm{' + str(int(lab)) + '}$' for lab in xlab]
        ax1.set_xticklabels(xlab)
        ylab = ax1.get_yticks().tolist()
        ylab = ['$\mathrm{' + str(int(lab)) + '}$' for lab in ylab]
        ax1.set_yticklabels(ylab)
        
        # Plot detector profile and fit
        for c in self.component_list:
            if isinstance(c, Detector):
                d = c
                
        px = d.pixel_values
        x = range(len(px[:, int(px.shape[0]/2)-1]))
        y = px[:, int(px.shape[0]/2)-1]
        y = y/self.source.n_rays
        
        ax3.set_xlim(0, c.xnpix)
        ax3.step(x, y, where='mid', markersize=0, color='k')
        ax3.set_aspect(aspect=1./ax3.get_data_ratio(), adjustable='box-forced')
        if fit is not None:
            ax3.step(x, fit, where='mid', markersize=0, color='r')
        ax3.set_aspect(aspect=1./ax3.get_data_ratio(), adjustable='box-forced')
                
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


