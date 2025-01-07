

from enum import Enum

from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO
from Util.Misc import Normalized

from .Surface import Surface


class SymmetryType(Enum):
    Axial = 0         # Symmetric around a line
    Relfection = 1    # Symmetric around 1 axis 
    Rectangular = 2   # Symmetric around 2 axis 
    Asymmetric = 3    # Not symmetric at all 



class VirtualSurface():
    """
    A virtual surface is something that is non-physical. It is typically used to represent some aspect of the lens, such as the pupil, the principal plane, etc.
    """
    def __init__(self):
        self.vertex = []
        self.symmetryType = None

    def DrawSurface(self,):
        pass 


    
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
            pass 

