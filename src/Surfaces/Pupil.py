



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

        if(len(self._zDepth) == 1):
            # When there is only one data point,
            # Assume it is the center point on axis and use it as the overall depth 
            DrawDisk(self.clearSemiDiameter, self._zDepth[0])

        else:
            # When there are many different points for the pupil plane 
            pass 



