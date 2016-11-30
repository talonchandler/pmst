import sys
sys.path.append("../../")

from unittest import TestCase
import pmst.component
import numpy as np


class TestPixel(TestCase):

    def setUp(self):
        origin = np.array((0, 0, 0))
        normal = np.array((0, 0))
        dimensions = np.array((.1, .2))
