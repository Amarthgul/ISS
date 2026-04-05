"""
A surface that does not bend the rays, only cull them.
"""
from .Surface import Surface
from Util.Backend import backend as bd
from Util.Backend import constant, backend_name
from Util.PltPlot import DrawAspherical, DrawAsphericalProfile, DrawSphericalProfile, DrawPlane
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR, DEFAULT_MAT_NAME
from Raytracing.RayBatch import RayBatch


class ManualAperture(Surface):
    def __init__(self, r=INFINITY, t=-20, sd=INFINITY, m=DEFAULT_MAT_NAME):
        super().__init__(r, t, sd, m)

        self.isCircular = True

        """When is Circular is set to False, this manual aperture will be regarded as being a box. The width and height here are used to define its size"""
        self.gateWidth = 50
        self.gateHeight = 40

        self._plotColor = 'r'


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):
        """
        This surface only performs geometric clipping.
        Rays are intersected with the underlying plane / sphere exactly like a normal
        surface, but their directions are preserved. Rays that miss the surface or fall
        outside the selected aperture shape are vignetted.
        """

        position = incidentRaybatch.Position()
        direction = incidentRaybatch.Direction()

        if self.radius == INFINITY:
            # ---------------- Plane intersection ----------------
            denom = bd.dot(direction, self._axis)
            parallel_mask = bd.isclose(denom, ZERO)
            t = bd.dot((self.frontVertex - position), self._axis) / denom
            forward_mask = (~parallel_mask) & (t >= ZERO)

            intersections_all = position + t[:, bd.newaxis] * direction
        else:
            # -------------- Spherical intersection --------------
            oc = position - self.radiusCenter

            a = bd.sum(direction**TWO, axis=1)
            b = constant(2.0) * bd.sum(oc * direction, axis=1)
            c = bd.sum(oc**TWO, axis=1) - self.radius**TWO

            discriminant = b**TWO - constant(4.0) * a * c
            sphere_hit = discriminant > ZERO

            # Avoid invalid sqrt warnings on missed rays.
            safe_disc = bd.where(sphere_hit, discriminant, ZERO)
            sqrt_disc = bd.sqrt(safe_disc)

            t1 = (-b - sqrt_disc) / (TWO * a)
            t2 = (-b + sqrt_disc) / (TWO * a)

            t = bd.copy(t1)
            mask = bd.sign(self.radius) != bd.sign(direction[:, Axis.Z.value])
            t[mask] = t2[mask]

            intersections_all = position + t[:, bd.newaxis] * direction
            forward_mask = sphere_hit & ~((t1 < ZERO) & (t2 < ZERO))

        if self.isCircular:
            aperture_mask = self.CircularAperture(intersections_all)
        else:
            aperture_mask = self.RectangularAperture(intersections_all)

        valid = forward_mask & aperture_mask
        boolVig = ~valid

        if not bd.any(valid):
            emptyRB = RayBatch(None)
            TIR = bd.zeros(0, dtype=bd.bool_)
            return emptyRB, TIR, boolVig, None

        outRB = RayBatch(bd.copy(incidentRaybatch.value[valid]))
        outRB.SetPosition(intersections_all[valid])
        outRB.SetDirection(direction[valid])

        TIR = bd.zeros(outRB.Wavelength().shape[0], dtype=bd.bool_)

        return outRB, TIR, boolVig, None


    def DrawSurface(self, DrawBoundary=False):
        """
        Override the default drawing behavior.

        Circular manual apertures keep the parent's spherical / planar visualization.
        Rectangular manual apertures are drawn as a simple gate plane centered on the
        optical axis at the surface vertex depth, which is typically the most useful
        visualization for hoods, matte boxes, and rectangular stops.
        """

        if self.isCircular:
            super().DrawSurface(DrawBoundary=DrawBoundary)
            return

        halfW = self.gateWidth / TWO
        halfH = self.gateHeight / TWO
        z = self.frontVertex[Axis.Z.value]

        points = bd.array([
            [-halfW, -halfH, z],
            [ halfW, -halfH, z],
            [ halfW,  halfH, z],
            [-halfW,  halfH, z],
        ])

        DrawPlane(points, color=SURFACE_COLOR)


    def RectangularAperture(self, intersections):

        # Try to use the _ApertureMask from the parent class for easier future maintenance
        halfW = self.gateWidth / TWO
        halfH = self.gateHeight / TWO

        result = (bd.abs(intersections[:, Axis.X.value]) < halfW) & \
                 (bd.abs(intersections[:, Axis.Y.value]) < halfH)

        if self.minAperture is not None:
            result &= ~self._ApertureMask(intersections)

        return result


    def CircularAperture(self, intersections):

        return self._ApertureMask(intersections)



