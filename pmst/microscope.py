from pmst.source import Source
import numpy as np
import matplotlib.pyplot as plt

class Microscope:
    """A microscope""" 
    def __init__(self, source):
        if isinstance(source, Source):
            self.source = source
        else:
            raise ValueError('Argument must be a Source')
        self.component_list = []

    def add_component(self, component):
        self.component_list.append(component)

    def simulate(self):
        intersections = []
        for ray in self.source.rays:
            for component in self.component_list:
                intersections.append(component.intersection(ray))

        return intersections

    def plot_results(self, filename, src):
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(11, 8))
        # TODO Generalize this
        px = self.component_list[0].pixel_values
        
        ax0.set_xlim([0, 10])
        ax0.set_ylim([-10, 0])

        src = src[0][1:]
        src = [x.lstrip() for x in src]
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
        
        ax1.imshow(px, interpolation='none')

        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-10, 10])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax2.add_artist(plt.Circle((0,0), 0.1, color='k'))
        ax2.add_artist(plt.Line2D((-5,5), (-2, -2), color='k', markersize=0))
        ax2.text(x=7, y=0, s='Source', ha='left', va='center', size=8)
        ax2.text(x=7, y=-2, s='Detector', ha='left', va='center', size=8)        
        ax2.set_aspect(aspect=1, adjustable='box')
        
        x = range(len(px[:, int(px.shape[0]/2)-1]))
        y = px[:, int(px.shape[0]/2)-1]
        y = y/np.max(y)
        
        ax3.step(x, y, where='mid', markersize=0, color='k')
        x = np.arange(0, 100, 0.1)
        y = 4/(4+0.05*(x-50)**2)
        #ax3.plot(x, y, '-r')
        ax3.set_aspect(aspect=100, adjustable='box')
        f.savefig(filename, dpi=100)
        
    def __str__(self):
        return 'Components:\t'+str(len(self.component_list))+'\n'


