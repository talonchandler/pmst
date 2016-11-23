import sys
sys.path.append("../../../")

from pmst.source import DirectedPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector, Doubler, DetectorGPU
from pmst.geometry import Point
import numpy as np
import time; start = time.time(); print('Running...')

# GPU
t2 = time.time()
db = Doubler()
sgpu = DirectedPointSource(Point(0, 0, 0), n_rays=int(1e4), direction=Point(0, 0, 1), psi=np.pi/2)
m2 = Microscope(source=sgpu)

npx = 100
d = DetectorGPU(Point(0, 0, 1), x_edge=Point(2, 0, 1), y_edge=Point(0, 2, 1), xnpix=npx, ynpix=npx)
m2.add_component(d)

hist = m2.simulate_gpu2()
print("GPU: ", time.time() - t2)

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

m2.plot_results('isotropic_point_source.png', src=src)

print('Run time:', np.round(time.time() - start, 2), 's')
