from pmst.source import Source
from pmst.detector import Detector
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Microscope:
    """A microscope""" 
    def __init__(self, source, detector):
        if isinstance(source, Source):
            self.source = source
        else:
            raise ValueError('Source is not the correct type.')
        if isinstance(detector, Detector):
            self.component_list = [detector]
        else:
            raise ValueError('Detector is not the correct type.')

    def add_component(self, component):
        self.component_list.append(component)

    def simulate(self):
        intersections = []
        for ray in self.source.rays:
            for component in self.component_list:
                intersections.append(component.intersection(ray))

        return intersections

    def plot_results(self, filename, src='', dpi=300):
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(11, 8))

        # Plot source
        ax2.add_artist(plt.Circle((self.source.origin.x, self.source.origin.z),
                                  0.1, color='k'))
        ax2.text(x=self.source.origin.x + 7, y=self.source.origin.z,
                 s='Source', ha='left', va='center', size=8)

        # Plot each component
        for c in self.component_list:
            if isinstance(c, Detector):
                # Plot data
                px = c.pixel_values
                ax1.imshow(c.pixel_values, interpolation='none')
                
                x = range(len(px[:, int(px.shape[0]/2)-1]))
                y = px[:, int(px.shape[0]/2)-1]
                y = y/np.max(y)
                ax3.step(x, y, where='mid', markersize=0, color='k')

               #xfit = np.arange(0, 100, 0.1)
               #yfit = 4/(4+0.05*(xfit-50)**2)
               #ax3.plot(xfit, yfit, '-r')

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

        ax3.set_aspect(aspect=100, adjustable='box')
        f.savefig(filename, dpi=dpi)
        
    def __str__(self):
        return 'Components:\t'+str(len(self.component_list))+'\n'


