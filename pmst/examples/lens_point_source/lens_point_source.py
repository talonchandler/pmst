import sys
sys.path.append("../../../")

from pmst.source import DirectedPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point
from pmst.component import Lens
import numpy as np
import time; start = time.time(); print('Running...')

t = time.time()
s = DirectedPointSource(Point(0, 0, 0),
                        n_rays=int(1e7),
                        direction=Point(0, 0, 1),
                        psi=np.pi/2)

m = Microscope(source=s)

l = Lens(Point(0, 0, 1.0),
         n=1.5,
         normal=Point(0, 0, 1.0),
         f=1,
         radius=0.5,
         label=True)

m.add_component(l)

npx = 100
d = Detector(center=Point(0, 0, 2),
             x_edge=Point(4, 0, 2),
             y_edge=Point(0, 4, 2),
             xnpix=npx, ynpix=npx)

m.add_component(d)

m.simulate()

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

m.plot_results('lens_point_source_off.pdf', src=src)

print('Total:\t\t', np.round(time.time() - start, 1), 's')
