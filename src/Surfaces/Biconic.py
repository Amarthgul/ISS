

from .Surface import *
from Raytracing.RayBatch import RayBatch
from Util.Backend import backend as bd
from Util.Backend import constant, backend_name
from Util.PltPlot import DrawBiconicSurface
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR
from Util.Misc import ArrayNormalized

class BiconicSurface(Surface):
    def __init__(self,  r, t, sd, m="AIR", K=0, ry = INFINITY, ky = 0):
        """
        Biconic surface. Mostly for modeling cylindrical element.
        """

        # Safe to assume there will never be ASPH associated with this type of surface.

        super().__init__(r, t, sd, m)

        # r radius is regarded as the X direction (horizontal) radius


        """X direction conic factor."""
        self.xConic = constant(K)

        """Y direction radius. IInfinity (flat) by default."""
        self.yRadius = constant(ry)

        """Y direction conic factor. 0 by default."""
        self.yConic = constant(ky)

        # Biconics preserve the previous rectangular default unless callers explicitly switch to a circular field stop.
        self.fieldStop = FieldStopType.Rectangular

        """Y direction semi diameter, or more precisely, just the (half) size of the surface on Y direction. Usable when fieldStop is rectangular."""
        self.ySemi = constant(sd)





    def DrawSurface(self, DrawBoundary=True):

        # First draw the biconic surface itself
        DrawBiconicSurface(
            radiusX=self.radius,
            kX=self.xConic,
            radiusY=self.yRadius,
            kY=self.yConic,
            clearSemiDiameter=self.clearSemiDiameter,
            cumulativeThickness=self.cumulativeThickness,
            ySemi=self.ySemi,
            fieldStop=self.fieldStop,
            surfaceColor=SURFACE_COLOR,
        )

        if DrawBoundary:
            # Use elliptical boundary for circular apertures, flat boundaries for rectangular apertures.
            if self.fieldStop == FieldStopType.Rectangular:
                for boundary in self._RectangularClearBoundaryList():
                    boundary.DrawSurface()
            else:
                if self.clearBoundaryL is not None:
                    self.clearBoundaryL.DrawSurface()

                if self.clearBoundaryT is not None:
                    self.clearBoundaryT.DrawSurface()


    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the surface vertex and a
        conservative edge z value for the rectangular biconic aperture.
        """
        cumulativeT = bd.array(cumulativeT)

        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT + self.radius])
        self._radiusDirection = self.frontVertex - self.radiusCenter

        if self.fieldStop == FieldStopType.Rectangular:
            zMin, zMax = self.RectangularApertureZRange()
            if self._HasNegativeCurvature() and not self._HasPositiveCurvature():
                self.sdCumulative = zMin
            else:
                self.sdCumulative = zMax
        else:
            self.sdCumulative = cumulativeT + self._SagBiconicXY(self.clearSemiDiameter, ZERO)


    def Intersection(self, incidentRaybatch):

        # Use different intersection methods depending on the surface type for faster calculation

        if self._IsPlane():
            return self._PlaneIntersection(incidentRaybatch)

        if self._IsCylindrical():
            return self._CylindricalIntersection(incidentRaybatch)

        return self._SolveSagIntersection(incidentRaybatch)


    def Normal(self, intersections):
        """
        Given the intersections, calculate the normal direction on these intersection points.
        The intersections are treated as already on the surface.

        :param intersections: points on the surface.

        :return: Normalized normals of the intersection points on this surface.
        """
        if intersections.shape[0] == 0:
            return bd.zeros((0, 3))

        x = intersections[:, Axis.X.value]
        y = intersections[:, Axis.Y.value]

        dzdx, dzdy = self._SagGradientXY(x, y)

        normals = bd.stack((-dzdx, -dzdy, bd.ones_like(dzdx)), axis=1)

        return ArrayNormalized(normals)


    def RectangularApertureZRange(self):
        """
        Return the minimum and maximum Z extents over the rectangular aperture.
        This preserves both sides of a saddle / mixed-sign biconic.
        """
        sagMin, sagMax = self._RectangularApertureSagRange()

        return self.cumulativeThickness + sagMin, self.cumulativeThickness + sagMax


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):

        if self.fieldStop != FieldStopType.Rectangular:
            return super().Trace(incidentRaybatch, previousRI, inverted, reflection, useClearBoundary)

        mainRB, TIR, boolVig, strayRB = super().Trace(
            incidentRaybatch,
            previousRI,
            inverted,
            reflection,
            useClearBoundary=False
        )

        if (reflection and useClearBoundary and
                self._ScalarBool(bd.any(boolVig)) and
                (not inverted)):
            vigRB = RayBatch(bd.copy(incidentRaybatch.value[boolVig]))
            vigRI = previousRI[boolVig]
            boundaryStray = self._TraceRectangularClearBoundaries(vigRB, vigRI)
            strayRB = self._MergeRayBatches(strayRB, boundaryStray)

        return mainRB, TIR, boolVig, strayRB


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _SolveSagIntersection(self, incidentRaybatch):
        """
        Intersect rays with z = vertex_z + biconic_sag(x, y).

        The solve is fixed-iteration Newton with a plane-intersection seed,
        which keeps the method vectorized and compatible with the rest of the
        surface tracing contract.
        """
        position = incidentRaybatch.Position()
        direction = incidentRaybatch.Direction()
        rayCount = position.shape[0]

        denom = direction[:, Axis.Z.value]
        valid = ~bd.isclose(denom, ZERO)

        t = bd.full(rayCount, bd.nan)
        t_plane = (self.cumulativeThickness - position[:, Axis.Z.value]) / denom
        t = bd.where(valid, t_plane, t)

        # A plane seed is usually already close in sequential tracing. Keep
        # the iteration count modest because this runs for every ray.
        eps = constant(1e-10)
        for _ in range(10):
            x = position[:, Axis.X.value] + t * direction[:, Axis.X.value]
            y = position[:, Axis.Y.value] + t * direction[:, Axis.Y.value]
            z = position[:, Axis.Z.value] + t * direction[:, Axis.Z.value]

            sag = self._SagBiconicXY(x, y)
            dzdx, dzdy = self._SagGradientXY(x, y)

            f = z - (self.cumulativeThickness + sag)
            fp = direction[:, Axis.Z.value] - \
                dzdx * direction[:, Axis.X.value] - \
                dzdy * direction[:, Axis.Y.value]

            goodStep = valid & bd.isfinite(f) & bd.isfinite(fp) & (bd.abs(fp) > eps)
            step = bd.where(goodStep, f / fp, ZERO)
            t_next = t - step
            t = bd.where(goodStep & bd.isfinite(t_next), t_next, t)

        intersections_all = position + t[:, bd.newaxis] * direction

        final_x = intersections_all[:, Axis.X.value]
        final_y = intersections_all[:, Axis.Y.value]
        final_z = intersections_all[:, Axis.Z.value]
        residual = final_z - (self.cumulativeThickness + self._SagBiconicXY(final_x, final_y))

        apertureMask = self._ApertureMask(intersections_all)
        interMask = valid & bd.isfinite(t) & (t >= ZERO) & bd.isfinite(residual) & \
            (bd.abs(residual) < constant(1e-6)) & apertureMask

        intersections = intersections_all[interMask]

        return intersections, \
            bd.zeros(intersections.shape[0], dtype=bd.bool_), \
            ~interMask


    def _CylindricalIntersection(self, incidentRaybatch):
        """
        Fast path for a zero-conic biconic with exactly one finite radius.

        A finite X radius is a circular arc swept along Y:
            x^2 + (z - center_z)^2 = radius^2
        A finite Y radius is the same arc swept along X.
        """
        position = incidentRaybatch.Position()
        direction = incidentRaybatch.Direction()

        xCylinder = not self._ScalarBool(bd.isinf(self.radius))
        radius = self.radius if xCylinder else self.yRadius
        curvedAxis = Axis.X.value if xCylinder else Axis.Y.value

        u = position[:, curvedAxis]
        du = direction[:, curvedAxis]
        z = position[:, Axis.Z.value] - (self.cumulativeThickness + radius)
        dz = direction[:, Axis.Z.value]

        a = du**TWO + dz**TWO
        b = TWO * (u * du + z * dz)
        c = u**TWO + z**TWO - radius**TWO

        eps = constant(1e-12)
        discriminant = b**TWO - constant(4) * a * c
        canSolve = (a > eps) & (discriminant >= ZERO)

        sqrtD = bd.sqrt(bd.maximum(discriminant, ZERO))
        denom = TWO * bd.where(a > eps, a, ONE)
        t1 = (-b - sqrtD) / denom
        t2 = (-b + sqrtD) / denom

        useSecondRoot = bd.sign(radius) != bd.sign(direction[:, Axis.Z.value])
        t = bd.where(useSecondRoot, t2, t1)

        intersections_all = position + t[:, bd.newaxis] * direction
        valid = canSolve & bd.isfinite(t) & (t >= ZERO) & self._ApertureMask(intersections_all)
        intersections = intersections_all[valid]

        return intersections, \
            bd.zeros(intersections.shape[0], dtype=bd.bool_), \
            ~valid


    def _TraceRectangularClearBoundaries(self, incidentRaybatch, previousRI):
        boundaries = self._RectangularClearBoundaryList()

        if len(boundaries) == 0 or incidentRaybatch.value.shape[0] == 0:
            return RayBatch(bd.copy(incidentRaybatch.value[:0]))

        remainingRB = RayBatch(bd.copy(incidentRaybatch.value))
        remainingRI = bd.copy(previousRI)
        reflectedRB = RayBatch(None)

        for boundary in boundaries:
            if remainingRB.value is None or remainingRB.value.shape[0] == 0:
                break

            boundaryRB, hitMask = boundary.Trace(remainingRB, remainingRI)
            reflectedRB = self._MergeRayBatches(reflectedRB, boundaryRB)

            if not self._ScalarBool(bd.any(~hitMask)):
                break

            remainingRB = RayBatch(bd.copy(remainingRB.value[~hitMask]))
            remainingRI = remainingRI[~hitMask]

        return reflectedRB


    def _RectangularClearBoundaryList(self):
        if len(self.clearBoundaries) > 0:
            return [boundary for boundary in self.clearBoundaries if boundary is not None]

        return [
            boundary for boundary in (self.clearBoundaryL, self.clearBoundaryT)
            if boundary is not None
        ]


    def _MergeRayBatches(self, baseRB, addRB):
        if addRB is None or addRB.value is None or addRB.value.shape[0] == 0:
            return baseRB

        if baseRB is None or baseRB.value is None or baseRB.value.shape[0] == 0:
            return addRB

        return baseRB.Merge(addRB)


    def _ApertureMask(self, intersections):
        """
        Biconics in anamorphic prescriptions commonly use rectangular apertures.
        When fieldStop is circular, fall back to the circular parent aperture.
        """
        if self.fieldStop != FieldStopType.Rectangular:
            return super()._ApertureMask(intersections)

        result = (bd.abs(intersections[:, Axis.X.value]) < self.clearSemiDiameter) & \
            (bd.abs(intersections[:, Axis.Y.value]) < self.ySemi)

        if self.minAperture is not None:
            result &= bd.sqrt(intersections[:, 0]**TWO + intersections[:, 1]**TWO) > self.minAperture

        return result


    def _SagBiconicXY(self, x, y):
        """
        Standard biconic sag:
            z = (cx*x^2 + cy*y^2) /
                (1 + sqrt(1 - (1+kx)cx^2*x^2 - (1+ky)cy^2*y^2))
        where an infinite radius contributes zero curvature.
        """
        cx = self._Curvature(self.radius)
        cy = self._Curvature(self.yRadius)

        numerator = cx * x**TWO + cy * y**TWO
        radicand = ONE - (ONE + self.xConic) * cx**TWO * x**TWO - \
            (ONE + self.yConic) * cy**TWO * y**TWO

        sqrtTerm = bd.sqrt(bd.maximum(radicand, ZERO))
        denom = ONE + sqrtTerm

        return numerator / denom


    def _RectangularApertureSagRange(self):
        x = self.clearSemiDiameter
        y = self.ySemi
        sampleX = bd.array([ZERO, x, -x, ZERO, ZERO, x, x, -x, -x])
        sampleY = bd.array([ZERO, ZERO, ZERO, y, -y, y, -y, y, -y])
        sag = self._SagBiconicXY(sampleX, sampleY)

        return bd.min(sag), bd.max(sag)


    def _HasPositiveCurvature(self):
        xPositive = (not self._ScalarBool(bd.isinf(self.radius))) and \
            self._ScalarBool(self.radius > ZERO)
        yPositive = (not self._ScalarBool(bd.isinf(self.yRadius))) and \
            self._ScalarBool(self.yRadius > ZERO)

        return xPositive or yPositive


    def _HasNegativeCurvature(self):
        xNegative = (not self._ScalarBool(bd.isinf(self.radius))) and \
            self._ScalarBool(self.radius < ZERO)
        yNegative = (not self._ScalarBool(bd.isinf(self.yRadius))) and \
            self._ScalarBool(self.yRadius < ZERO)

        return xNegative or yNegative


    def _SagGradientXY(self, x, y):
        """
        First derivatives of the biconic sag with respect to x and y.
        """
        cx = self._Curvature(self.radius)
        cy = self._Curvature(self.yRadius)

        numerator = cx * x**TWO + cy * y**TWO
        ax = (ONE + self.xConic) * cx**TWO
        ay = (ONE + self.yConic) * cy**TWO
        radicand = ONE - ax * x**TWO - ay * y**TWO

        eps = constant(1e-12)
        sqrtTerm = bd.sqrt(bd.maximum(radicand, eps))
        denom = ONE + sqrtTerm
        denom2 = denom**TWO

        dsqrt_dx = -ax * x / sqrtTerm
        dsqrt_dy = -ay * y / sqrtTerm

        dnum_dx = TWO * cx * x
        dnum_dy = TWO * cy * y

        dzdx = (dnum_dx * denom - numerator * dsqrt_dx) / denom2
        dzdy = (dnum_dy * denom - numerator * dsqrt_dy) / denom2

        return dzdx, dzdy


    def _Curvature(self, radius):
        if self._ScalarBool(bd.isinf(radius)):
            return ZERO

        return ONE / radius


    def _IsPlane(self):
        return self._ScalarBool(bd.isinf(self.radius)) and self._ScalarBool(bd.isinf(self.yRadius))


    def _IsCylindrical(self):
        xFinite = not self._ScalarBool(bd.isinf(self.radius))
        yFinite = not self._ScalarBool(bd.isinf(self.yRadius))
        noConic = self._ScalarBool(self.xConic == ZERO) and \
            self._ScalarBool(self.yConic == ZERO)

        return noConic and (xFinite != yFinite)


    def _IsConicSweep(self):
        xConic = (not self._ScalarBool(bd.isinf(self.radius))) and \
            self._ScalarBool(bd.isinf(self.yRadius))
        yConic = self._ScalarBool(bd.isinf(self.radius)) and \
            (not self._ScalarBool(bd.isinf(self.yRadius)))

        return xConic or yConic


    def _ScalarBool(self, value):
        if hasattr(value, "get"):
            return bool(value.get())

        return bool(value)
