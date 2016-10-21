from pmst.source import DirectedPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point
import numpy as np
import time; start = time.time(); print('Running...')

s = DirectedPointSource(Point(0, 0, 0), n_rays=1e6, direction=Point(0, 0, 1), psi=np.pi/2)

center = Point(0, 0, 2)
x_edge = Point(5, 0, 2)
y_edge = Point(0, 5, 2)
n_pixels = 100
d = Detector(center, x_edge, y_edge, n_pixels, n_pixels)

m = Microscope(source=s, detector=d)
m.add_component(d)
m.simulate()

with open(__file__, 'r') as myfile:
    src = myfile.readlines()
m.plot_results('isotropic_point_source.png', src=src)

print('Run time:', np.round(time.time() - start, 2), 's')
