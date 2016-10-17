from pmst.source import IsotropicPointSource
from pmst.microscope import Microscope
from pmst.detector import Detector
from pmst.geometry import Point

s = IsotropicPointSource(Point(0, 0, 0), n_rays=1e5)

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

