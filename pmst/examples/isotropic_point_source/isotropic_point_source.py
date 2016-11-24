import sys; sys.path.append("../../../")
from pmst.source import DirectedPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point
import numpy as np
import time; start = time.time(); print('Running...')

s = DirectedPointSource(origin=Point(0, 0, 0),
                        n_rays=int(1e6),
                        direction=Point(0, 0, 1),
                        psi=np.pi/2)

m = Microscope(source=s)

d = Detector(center=Point(0, 0, 1),
             x_edge=Point(2, 0, 1),
             y_edge=Point(0, 2, 1),
             xnpix=100, ynpix=100)

m.add_component(d)

m.simulate()

print("GPU: ", np.round(time.time() - start, 2), 's')

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

m.plot_results('isotropic_point_source.pdf', src=src)

print('Total:', np.round(time.time() - start, 2), 's')
