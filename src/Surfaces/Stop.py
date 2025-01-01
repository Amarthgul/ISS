

from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO
from Util.Backend import backend as bd

from .Surface import Surface 
from Material import Material

class Stop(Surface):
    def __init__(self, t):
        self.thickness = t
        self.bladeShape = None
        self.bladeCount = 5
        self.material = Material("AIR")


    def SetFNumber(self, fNum):
        pass 

    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the vertices. This is for when the surface share the same optical axis with the lens. 
        """

        # The local optical axis remains the same as OBJ FACING 
        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT]) 

        # The front vertex and the radius center are the same for a stop. 

