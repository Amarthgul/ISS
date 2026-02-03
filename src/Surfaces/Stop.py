

from Util.Backend import backend as bd
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR, DEFAULT_MAT_NAME
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, DrawEllipse, DrawClearBoundary, DrawSphericalInner
from .Surface import Surface
from Material import Material

class Stop(Surface):
    """
    Stop of the system.
    Note that, although the stop in proactive may not be the aperture diaphragm, for practical reason we treat it as such.
    """
    def __init__(self, t):
        super().__init__(INFINITY, t, INFINITY, DEFAULT_MAT_NAME)
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


    def DrawSurface(self, DrawBoundary=True):

        # If semi diameter is not enforced, skip drawing
        if (self.clearSemiDiameter == INFINITY): return

        if self.minAperture is None:
            DrawSpherical(
                self.radius,
                self.clearSemiDiameter,
                self.cumulativeThickness,
                surfaceColor=SURFACE_COLOR,
            )
        else:
            DrawSphericalInner(
                self.radius,
                self.clearSemiDiameter,
                self.minAperture,
                self.cumulativeThickness,
                surfaceColor=SURFACE_COLOR,
            )

        if (DrawBoundary and self.clearBoundaryL is not None):
            self.clearBoundaryL.DrawSurface()

        if (DrawBoundary and self.clearBoundaryT is not None):
            self.clearBoundaryT.DrawSurface()


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):

        pass



