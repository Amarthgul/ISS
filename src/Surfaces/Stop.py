

from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO
from Util.Backend import backend as bd

from .Surface import Surface 
from Material import Material


class Stop(Surface):
    """
    Stop of the system.
    Note that, although the stop in proactive may not be the aperture diaphragm, for practical reason we treat it as such.
    """
    def __init__(self, t):
        self.thickness = t 
        self.bladeShape = None
        self.bladeCount = 5


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


    def EnforceSemiDiameter(self, sd):
        self.clearSemiDiameter = sd 


    def DrawSurface(self):
        """
        Stop does not have a surface to draw.
        """
        return


