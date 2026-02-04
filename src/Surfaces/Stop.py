import matplotlib.pyplot as plt
from Util.Backend import backend as bd
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR, DEFAULT_MAT_NAME
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, DrawEllipse, DrawClearBoundary, DrawSphericalInner
from .Surface import Surface
from Material import Material
from Raytracing.RayBatch import RayBatch
from Raytracing.Raypath import RayPath
from Raytracing.Refraction import Refract
from Raytracing.Reflection import Reflect



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



    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):

        if (self.material == "MIRROR"):
            return self.TraceMirror(incidentRaybatch, previousRI, inverted, reflection)

        # First find the intersections
        intersections, _temp, boolVig = self.Intersection(incidentRaybatch)

        DrawPoints(intersections)


        if (self.stopOnly):
            incidentRaybatch = incidentRaybatch.Mask(~boolVig)
            TIR = bd.zeros_like(incidentRaybatch.Wavelength()).astype(bool)
            return incidentRaybatch, TIR, boolVig, None

        # DrawPoints(intersections) # ======= Draw call

        # The normal should be pointing at the oppoiste z direction as the indicent raybatch
        desiredDirection = -bd.sign(incidentRaybatch.Direction()[:, 2])[~boolVig]
        # Apply desired direction to the normals
        normals = self.Normal(intersections)
        normals[desiredDirection != bd.sign(normals[:, 2])] *= -1

        # Truncate the rays that are vignetted
        directions = incidentRaybatch.Direction()[~boolVig]

        # Accquire the index of refractions (resp. wavelength)
        n1 = self.material.RI(incidentRaybatch.Wavelength()[~boolVig])
        n2 = previousRI[~boolVig]

        # If the ray hits from the behind, RI needs to be swapped
        if (inverted):
            n1, n2 = n2, n1

            # Only the non vignetted rays goes into refraction
        refracted, TIR, _temp = Refract(directions, normals, n2, n1)

        # DrawDirection(intersections, reflected, lineColor="b") # ======= Draw call

        mainRB = RayBatch(bd.copy(incidentRaybatch.value[~boolVig][~TIR]))
        mainRB.SetPosition(intersections[~TIR])
        mainRB.SetDirection(refracted)

        strayRB = RayBatch(bd.copy(incidentRaybatch.value[~boolVig][~TIR]))

        return mainRB, TIR, boolVig, strayRB

