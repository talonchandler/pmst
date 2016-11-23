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
s = DirectedPointSource(Point(0, 0, 0), n_rays=int(5e7), direction=Point(0, 0, 1), psi=np.pi/2)
m = Microscope(source=s)

n_lens = 5
width = 5
diameter = width/n_lens
pos = np.arange(0, width, diameter) - np.floor(n_lens/2)
for x in pos:
    for y in pos:
        if x == 0 and y == 0:
            print("HERE")
            mylabel = 'Lens'
        else:
            mylabel = ''
        l = Lens(Point(x, y, 4.5), 1.5, Point(0, 0, 4.5), 1, diameter/2, mylabel)
        m.add_component(l)        

d = Detector(Point(0, 0, 5), x_edge=Point(2, 0, 5), y_edge=Point(0, 2, 5), xnpix=100, ynpix=100)
m.add_component(d)

hist = m.simulate()
print("GPU:\t", np.round(time.time() - t, 2), 's')

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

m.plot_results('lens_array_point_source.png', src=src)

print('Total:\t', np.round(time.time() - start, 2), 's')
