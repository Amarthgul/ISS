



from Util.Misc import Normalized
from Util.Backend import backend as bd 
from Util.PlotTest import DrawDisk


from .VirtualSurface import VirtualSurface, SymmetryType


class Pupil(VirtualSurface):
    def __init__(self):
        
        self.symmetryType = SymmetryType.Axial
        # By default the pupil is axial symmetric 

        self.clearSemiDiameter = None 

        self._height = []
        self._zDepth = []


    def AddSamplePoint(self, point):
        self._zDepth.append(point[2])
        self._height.append(Normalized(bd.array([point[0], point[1]])))


    def DrawSurface(self):
        DrawDisk(self.clearSemiDiameter, self._zDepth[0])

