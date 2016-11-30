import time; total_time = time.time(); print('Running...'); init_time = total_time
import sys; sys.path.append("../../../")
from pmst.source import DirectedPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point
import numpy as np

print("Initialize:\t", np.round(time.time() - init_time, 1), 's');


s = DirectedPointSource(origin=Point(0, 0, 0),
                        n_rays=int(1e5),
                        direction=Point(0, 0, 2),
                        psi=np.pi/2)

m = Microscope(source=s)

d = Detector(center=Point(0, 0, 1),
             x_edge=Point(2, 0, 1),
             y_edge=Point(0, 2, 1),
             xnpix=100, ynpix=100)

m.add_component(d)

m.simulate()

plot_time = time.time()

with open(__file__, 'r') as myfile:
    src = myfile.readlines()

# Calculate true
def sa(a, b, d):
    alpha = a/(2*d)
    beta = b/(2*d)
    omega = 4*np.arccos(np.sqrt((1+alpha**2+beta**2)/((1+alpha**2)*(1+beta**2))))
    return omega

def off_frac(A, B, a, b, d):
    t1 = sa(2*(A+a), 2*(B+b), d)
    t2 = sa(2*A, 2*(B+b), d)
    t3 = sa(2*(A+a), 2*B, d)
    t4 = sa(2*A, 2*B, d)
    sa_off = (t1 - t2 - t3 + t4)/4
    return np.abs(sa_off/(2*np.pi))

pixel_width = d.xwidth/d.xnpix
px = d.pixel_values
x = range(len(px[:, int(px.shape[0]/2)-1]))
iso_fit = [off_frac((i-50)*pixel_width, 0, pixel_width, pixel_width, 1) for i in x]
    
m.plot_results('isotropic_point_source.pdf', src=src[0:25], fit=iso_fit)

print('Plotting:\t', np.round(time.time() - plot_time, 1), 's')
print('Total:\t\t', np.round(time.time() - total_time, 1), 's')
