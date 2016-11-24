import sys
sys.path.append("../../../")

from pmst.source import DirectedPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point
import numpy as np
import time; start = time.time(); print('Running...')

t = time.time()
s = DirectedPointSource(Point(0, 0, 0), n_rays=int(1e6), direction=Point(0, 0, 1), psi=np.pi/2)
m = Microscope(source=s)

npx = 100
d = Detector(Point(0, 0, 1), x_edge=Point(2, 0, 1), y_edge=Point(0, 2, 1), xnpix=npx, ynpix=npx)
m.add_component(d)

hist = m.simulate()
print("GPU: ", np.round(time.time() - t, 2), 's')

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

m.plot_results('isotropic_point_source.pdf', src=src)

print('Run time:', np.round(time.time() - start, 2), 's')
