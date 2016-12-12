import sys
sys.path.append("../../../")

from pmst.source import PlanarSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point
from pmst.component import Lens
import numpy as np
import time; start = time.time(); print('Running...')

t = time.time()
s = PlanarSource(origin=Point(0, 0, 0),
                 n_rays=int(1e7),
                 direction=Point(0, 0, 1),
                 xedge=Point(1, 0, 1),
                 yedge=Point(0, 1, 1))

m = Microscope(source=s)

l = Lens(Point(0, 0, 1),
         n=1.5,
         normal=Point(0, 0, 1),
         f=1,
         radius=0.5,
         label=True)

m.add_component(l)

npx = 100
d = Detector(Point(0, 0, 2),
             x_edge=Point(2, 0, 2),
             y_edge=Point(0, 2, 2),
             xnpix=npx, ynpix=npx)

m.add_component(d)

m.simulate()
print("GPU:\t", np.round(time.time() - t, 2), 's')

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

m.plot_results('lens_planar_source.pdf', src=src)

print('Total:\t', np.round(time.time() - start, 2), 's')
