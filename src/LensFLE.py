

from Lens import Lens
from Util.Globals import INFINITY

class LensFLE(Lens):
    def __init__(self):

        super().__init__()

        """Object distance keys"""
        self.keyObjectDistance = []

        """Surface thicknesses corresponding to the keys"""
        self.keyedSurfaceD = {}

        self.firstSurfaceIndex = 2



    def PresetKeys(self):

        self.keyObjectDistance = [INFINITY, 1588, 740]
        self.keyedSurfaceD = {2: [36.814, 26.599, 16.553],
                             5: [6.883, 17.097, 27.144]}

