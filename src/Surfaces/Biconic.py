

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


        """When flagged, surface is treated to be a sweep, the semi-diameter would be regarded as the x direction (half) size. """
        self.isSweep = True

        """Y direction semi diameter, or more precisely, just the (half) size of the surface on Y direction. Usable when isSweep is flagged."""
        self.ySemi = constant(sd)


        """List of four clear boundaries along the four directions"""
        self.clearBoundaries = []


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
            isSweep=self.isSweep,
            surfaceColor=SURFACE_COLOR,
        )

        if DrawBoundary:
            # Use elliptical boundary if isSweep is not flagged, use the flat clearBoundaries if isSweep is flagged
            if self.isSweep:
                for boundary in self._SweepClearBoundaryList():
                    boundary.DrawSurface()
            else:
                if self.clearBoundaryL is not None:
                    self.clearBoundaryL.DrawSurface()

                if self.clearBoundaryT is not None:
                    self.clearBoundaryT.DrawSurface()


    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the surface vertex and a
        conservative edge z value for the swept biconic aperture.
        """
        cumulativeT = bd.array(cumulativeT)

        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT + self.radius])
        self._radiusDirection = self.frontVertex - self.radiusCenter

        if self.isSweep:
            x = bd.array([ZERO, self.clearSemiDiameter, -self.clearSemiDiameter, ZERO, ZERO])
            y = bd.array([ZERO, ZERO, ZERO, self.ySemi, -self.ySemi])
            self.sdCumulative = cumulativeT + bd.max(self._SagBiconicXY(x, y))
        else:
            self.sdCumulative = cumulativeT + self._SagBiconicXY(self.clearSemiDiameter, ZERO)


    def Intersection(self, incidentRaybatch):

        # Use different intersection methods depending on the surface type for faster calculation

        if self._IsPlane():
            return self._PlaneIntersection(incidentRaybatch)

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


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):

        if not self.isSweep:
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
            boundaryStray = self._TraceSweepClearBoundaries(vigRB, vigRI)
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


    def _TraceSweepClearBoundaries(self, incidentRaybatch, previousRI):
        boundaries = self._SweepClearBoundaryList()

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


    def _SweepClearBoundaryList(self):
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
        Biconics in anamorphic prescriptions are commonly swept rectangularly.
        When isSweep is disabled, fall back to the circular parent aperture.
        """
        if not self.isSweep:
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
        xSphere = (not self._ScalarBool(bd.isinf(self.radius))) and \
            self._ScalarBool(self.xConic == ZERO) and \
            self._ScalarBool(bd.isinf(self.yRadius))
        ySphere = self._ScalarBool(bd.isinf(self.radius)) and \
            (not self._ScalarBool(bd.isinf(self.yRadius))) and \
            self._ScalarBool(self.yConic == ZERO)

        return xSphere or ySphere


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


