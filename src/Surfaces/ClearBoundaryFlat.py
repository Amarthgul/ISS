
from Surfaces.ClearBoundary import ClearBoundary
from Raytracing.RayBatch import RayBatch
from Raytracing.Reflection import Reflect, LambertianReflect
from Raytracing.Refraction import Refract
from Raytracing.Polarization import SenkrechtUndParallel, ResidueRB, FresnelReflectance, QuantitativePolarize
from Util.Backend import backend as bd
from Util.Misc import ArrayNormalized
from Util.PltPlot import DrawPlane
from Util.Globals import PBR




class ClearBoundaryFlat(ClearBoundary):
    def __init__(self, coords, mat=PBR.GLASS):
        """
        A flat surface as the clear boundary of a refractive lens surface.

        :param coords: 4 points of the flat surface, assumed to be on the same plane.
        """
        super().__init__(None, None, mat)

        coords = bd.asarray(coords)
        if coords.shape != (4, 3):
            raise ValueError("ClearBoundaryFlat expects coords with shape (4, 3)")

        self.center = bd.mean(coords, axis=0)
        self._normal = self._BuildNormal(coords)
        self._u, self._v = self._BuildPlaneBasis(coords)

        order = self._CoordOrder(coords)
        self.coords = coords[order]
        self._coords2D = self._ProjectToPlane2D(self.coords)



    def DrawSurface(self):
        DrawPlane(self.coords)


    def Intersection(self, incidentRaybatch):
        """
        Calculate ray intersections against the finite quadrilateral plane.

        :return: intersection points (m, 3) and a boolean mask (n,) indicating
                 which input rays intersected the flat boundary.
        """
        origin = incidentRaybatch.Position()
        direction = incidentRaybatch.Direction()

        denom = bd.sum(direction * self._normal, axis=1)
        parallel = bd.isclose(denom, 0.0)

        safeDenom = bd.where(parallel, 1.0, denom)
        t = bd.sum((self.coords[0] - origin) * self._normal, axis=1) / safeDenom

        intersections = origin + t[:, bd.newaxis] * direction
        inside = self._InsideQuad(intersections)
        valid = (~parallel) & (t >= 0.0) & inside

        return intersections[valid], valid


    def Normal(self, intersections):
        """
        Return the plane normal for each intersection.
        """
        return bd.tile(self._normal, (intersections.shape[0], 1))


    def Trace(self, incidentRaybatch, previousRI, inverted=False):
        """
        A flat clear boundary reflects rays like ClearBoundary, but its geometry
        is just one finite plane.

        :return: reflected raybatch and a boolean mask indicating intersected rays.
        """
        intersections, mask = self.Intersection(incidentRaybatch)

        if intersections.shape[0] == 0:
            return RayBatch(bd.copy(incidentRaybatch.value[:0])), mask

        reflectedRB = RayBatch(bd.copy(incidentRaybatch.value[mask]))
        reflectedRB.SetPosition(intersections)

        directions = reflectedRB.Direction()
        normals = self.Normal(intersections)
        normals[bd.sum(directions * normals, axis=1) > 0.0] *= -1

        mirrorReflected = Reflect(directions, normals)
        lambertReflected, lambertIntensity = LambertianReflect(normals, outputPer=1)

        specularReflection = bd.clip(self.specularReflection, 0.0, 1.0)
        reflected = ArrayNormalized(
            mirrorReflected * specularReflection +
            lambertReflected * (1 - specularReflection)
        )
        reflectedRB.SetDirection(reflected)

        lambertCos = bd.sum(reflected * normals, axis=1)
        lambertCos = bd.clip(lambertCos, 0.0, 1.0)
        reflectionIntensity = (
            specularReflection +
            (1 - specularReflection) * lambertIntensity * lambertCos
        )
        reflectedRB.SetRadianceTerms(
            reflectedRB.RadianceTerms() * reflectionIntensity[:, None]
        )

        n1 = previousRI[mask]
        n2 = self.exteriorCoating.RI(reflectedRB.Wavelength())
        if inverted:
            n1, n2 = n2, n1

        refracted, TIR, _temp = Refract(directions, normals, n1, n2)
        nonTIRMask = ~TIR

        if self._Any(nonTIRMask):
            R_s, R_p = FresnelReflectance(
                normals[nonTIRMask],
                directions[nonTIRMask],
                refracted,
                n1[nonTIRMask],
                n2[nonTIRMask]
            )
            senkrecht, parallel = SenkrechtUndParallel(
                directions[nonTIRMask],
                normals[nonTIRMask]
            )

            nonTIRRB = RayBatch(reflectedRB.value[nonTIRMask])
            senkrecht, parallel = QuantitativePolarize(
                nonTIRRB.PolarizationMat(),
                senkrecht[:, :2],
                parallel[:, :2],
                R_s,
                R_p
            )
            nonTIRRB = ResidueRB(nonTIRRB, senkrecht, parallel)

            reflectedRB = nonTIRRB.Merge(RayBatch(reflectedRB.value[TIR]))
        else:
            reflectedRB = RayBatch(reflectedRB.value[TIR])

        absorption = min(max(self.absorption, 0.0), 1.0)
        reflectedRB.RandomDrop(absorption)

        return reflectedRB, mask


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _BuildNormal(self, coords):
        normal = bd.cross(coords[1] - coords[0], coords[2] - coords[0])
        norm = bd.linalg.norm(normal)

        if self._ScalarBool(norm < 1e-12):
            raise ValueError("ClearBoundaryFlat coords must contain at least three non-collinear points")

        return normal / norm


    def _BuildPlaneBasis(self, coords):
        u = coords[0] - self.center
        u = u - bd.sum(u * self._normal) * self._normal
        uNorm = bd.linalg.norm(u)

        if self._ScalarBool(uNorm < 1e-12):
            u = coords[1] - coords[0]
            u = u - bd.sum(u * self._normal) * self._normal
            uNorm = bd.linalg.norm(u)

        if self._ScalarBool(uNorm < 1e-12):
            raise ValueError("ClearBoundaryFlat could not build a plane basis from coords")

        u = u / uNorm
        v = bd.cross(self._normal, u)
        v = v / bd.linalg.norm(v)

        return u, v


    def _CoordOrder(self, coords):
        uv = self._ProjectToPlane2D(coords)
        angles = bd.arctan2(uv[:, 1], uv[:, 0])

        return bd.argsort(angles)


    def _ProjectToPlane2D(self, points):
        centered = points - self.center

        return bd.stack((
            bd.sum(centered * self._u, axis=1),
            bd.sum(centered * self._v, axis=1)
        ), axis=1)


    def _InsideQuad(self, points, eps=1e-10):
        uv = self._ProjectToPlane2D(points)
        poly = self._coords2D
        edgeStart = poly
        edgeEnd = bd.roll(poly, -1, axis=0)
        edges = edgeEnd - edgeStart
        rel = uv[:, bd.newaxis, :] - edgeStart[bd.newaxis, :, :]

        cross = edges[bd.newaxis, :, 0] * rel[:, :, 1] - \
            edges[bd.newaxis, :, 1] * rel[:, :, 0]

        insidePositive = bd.all(cross >= -eps, axis=1)
        insideNegative = bd.all(cross <= eps, axis=1)

        return insidePositive | insideNegative


    def _Any(self, value):
        value = bd.any(value)
        if hasattr(value, "get"):
            return bool(value.get())

        return bool(value)


    def _ScalarBool(self, value):
        if hasattr(value, "get"):
            return bool(value.get())

        return bool(value)
