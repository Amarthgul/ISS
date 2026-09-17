



from Util.Backend import backend as bd
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR, DEFAULT_MAT_NAME, MIRROR
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
        # White/True is open. The image spans +/- apertureShapeSemiDiameter
        # about frontVertex in physical lens units; image rows run toward -y.
        self.apertureShape = None
        self.apertureShapeSemiDiameter = None
        # Full-open radius used to map entrance-pupil changes onto this plane.
        self.apertureReferenceSemiDiameter = None

    def SetApertureShape(self, shape, semiDiameter):
        """Set a grayscale/RGB image or boolean mask and its physical half-size."""
        if not bd.isfinite(semiDiameter) or semiDiameter <= 0:
            raise ValueError("Aperture image semi-diameter must be finite and positive")
        shape = bd.asarray(shape)
        if shape.ndim not in (2, 3) or min(shape.shape[:2]) == 0:
            raise ValueError("Aperture shape must be a nonempty image or 2D mask")
        if shape.ndim == 3 and shape.shape[2] not in (3, 4):
            raise ValueError("Color aperture images must have RGB or RGBA channels")
        self.apertureShape = shape.copy()
        self.apertureShapeSemiDiameter = semiDiameter

    def _ApertureMask(self, intersections):
        local = intersections - self.frontVertex
        opened = super()._ApertureMask(local)
        if self.apertureShape is None:
            return opened
        gray = self.apertureShape
        if gray.ndim == 3:
            gray = gray[..., :3].mean(axis=-1)
        gray = gray.astype(float)
        peak = gray.max()
        gray = gray / bd.where(peak > 0, peak, 1)
        height, width = gray.shape
        x = local[:, 0] / self.apertureShapeSemiDiameter
        y = local[:, 1] / self.apertureShapeSemiDiameter
        inside = bd.isfinite(x) & bd.isfinite(y) & (bd.abs(x) <= 1) & (bd.abs(y) <= 1)
        # Match Pupil's pixel-center convention, but reject outside the image.
        x = bd.where(inside, x, 0)
        y = bd.where(inside, y, 0)
        u = bd.clip(bd.round(x * width / 2 + (width - 1) / 2).astype(int), 0, width - 1)
        v = bd.clip(bd.round(-y * height / 2 + (height - 1) / 2).astype(int), 0, height - 1)
        return opened & inside & (gray[v, u] > 0.5)


    def SetFNumber(self, fNum):
        pass 


    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the vertices. This is for when the surface share the same optical axis with the lens. 
        """

        # The local optical axis remains the same as OBJ FACING 
        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT + self.radius])
        self._radiusDirection = self.frontVertex - self.radiusCenter

        if (self.radius == INFINITY):
            # When r=inf, the cumulative thickness at the edge is the same as the cumulative thickness of the vertex.
            self.sdCumulative = cumulativeT
        else:
            self.sdCumulative = cumulativeT + self.radius + bd.sqrt(
                self.radius ** TWO - self.clearSemiDiameter ** TWO) * bd.sign(-self.radius)


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


    def NaiveTrace(self, incidentRaybatch, previousRI, inverted=False):
        RB, _TIR, _vig, _stray = self.Trace(incidentRaybatch, previousRI, inverted)
        return RB, _TIR, _vig


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):

        if self.material.name == MIRROR:
            return self.TraceMirror(incidentRaybatch, previousRI, inverted, reflection)

        if incidentRaybatch.value is None:
            empty = bd.zeros(0, dtype=bd.bool_)
            return RayBatch(None), empty, empty, RayBatch(None)

        # A stop absorbs blocked rays and leaves transmitted direction,
        # wavelength and polarization unchanged, from either side.
        position = incidentRaybatch.Position()
        direction = incidentRaybatch.Direction()
        denominator = direction @ self._axis
        parallel = bd.isclose(denominator, 0)
        distance = ((self.frontVertex - position) @ self._axis) / bd.where(parallel, 1, denominator)
        intersections = position + distance[:, None] * direction
        valid = (~parallel) & (distance >= -1e-10) & self._ApertureMask(intersections)
        mainRB = RayBatch(bd.copy(incidentRaybatch.value[valid]))
        mainRB.SetPosition(intersections[valid])
        tir = bd.zeros(mainRB.value.shape[0], dtype=bd.bool_)
        stray = RayBatch(bd.copy(incidentRaybatch.value[:0]))
        return mainRB, tir, ~valid, stray
