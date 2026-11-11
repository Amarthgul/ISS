"""Third-, fifth- and seventh-order Taylor aberrations of spherical lenses.

Contract: centered sequential spherical/plane refracting surfaces and stops,
positive indices, a nonsingular paraxial pupil and a finite Gaussian image
away from the last vertex. No mirrors, aspheres, clipping or real stop aiming.
The prescription is held fixed; rendering state and lens updates are unused.

Surface contributions are transverse ray errors at the common Gaussian image,
not classical per-surface wavefront sums. Each nonlinear surface increment is
transferred through the remaining first-order system. Upstream interactions
are included in the increment of the surface receiving the aberrated ray.
Combined wavefront coefficients through each surface are computed separately
from accumulated optical path, using each prefix's Gaussian sphere/plane.
All prefixes retain the original input pupil, field and object. Finite prefix
images must not coincide with their evaluation vertex.
"""

from dataclasses import dataclass
import math

from Raytracing.RayReference import RayReference, GaussianReference, Taylor, WavefrontCoefficients
from Util.Globals import LambdaLines
from .Paraxial import TraceLens, MatMul, RefractionMatrix, TranslationMatrix


@dataclass(frozen=True)
class RayAberrations:
    """Cartesian image errors in mm, polynomial in normalized pupil x,y,h.

    Both pupil coordinates are required for off-axis terms. The positive
    field and tangential pupil directions are +y. ForOrder isolates total
    pupil+field degree 3, 5 or 7; Evaluate sums the retained degrees.
    """
    x: Taylor
    y: Taylor

    def AsDict(self):
        return {"x": self.x.AsDict(), "y": self.y.AsDict()}

    def ForOrder(self, order):
        return RayAberrations(self.x.Select((order,)), self.y.Select((order,)))

    def Evaluate(self, pupilX, pupilY, h=1.0):
        return (self.x.Evaluate(pupilX, pupilY, h),
                self.y.Evaluate(pupilX, pupilY, h))


@dataclass(frozen=True)
class SurfaceHigherOrder:
    surfaceIndex: int
    coefficients: RayAberrations
    wavefront: WavefrontCoefficients
    wavefrontReference: GaussianReference


@dataclass(frozen=True)
class HigherOrderResult:
    surfaces: tuple[SurfaceHigherOrder, ...]
    totals: RayAberrations
    wavefront: WavefrontCoefficients
    reference: RayReference
    maxOrder: int
    units: str = "mm"
    convention: str = "Nonlinear surface ray increments at the Gaussian image"

    @property
    def paraxial(self):
        return self.reference.paraxial

    def SumSurfaces(self, first=0, last=None):
        """Sum an inclusive range, retaining full-system ray histories."""
        if last is None:
            last = len(self.surfaces) - 1
        zero = Taylor.Constant(0.0, self.maxOrder + 1)
        x, y = zero, zero
        for row in self.surfaces[first:last + 1]:
            x = x + row.coefficients.x
            y = y + row.coefficients.y
        return RayAberrations(x, y)

    def CumulativeCoefficients(self):
        """Running sums in the same final-image coordinates for every row."""
        zero = Taylor.Constant(0.0, self.maxOrder + 1)
        x, y = zero, zero
        rows = []
        for row in self.surfaces:
            x = x + row.coefficients.x
            y = y + row.coefficients.y
            rows.append(RayAberrations(x, y))
        return tuple(rows)

    def CumulativeWavefrontCoefficients(self):
        """Combined optical-path W through each surface, in the original pupil.

        Each entry uses that prefix's Gaussian reference through its vertex.
        These are accumulated wavefronts, not sums of the ray contributions.
        Their last entry is the complete-system wavefront. Differences between
        entries include propagation and reference changes as well as refraction.
        """
        return tuple(row.wavefront for row in self.surfaces)


def _RefractSphere(x, y, u, v, curvature, incidentRI, outgoingRI):
    """Map a ray between the two sides of a surface's vertex plane.

    Signed propagation to/from the actual sag makes translation between
    vertex planes exactly linear in slope. The sphere equation is
    z = c*(x_hit^2 + y_hit^2 + z^2)/2. Fixed-point substitution gains two
    degrees per iteration about the axial ray, without a numerical solver.
    """
    sag = Taylor.Constant(0.0, x.degree)
    for _ in range(x.degree // 2):
        hitX, hitY = x + u * sag, y + v * sag
        sag = (hitX * hitX + hitY * hitY + sag * sag).Scale(curvature / 2.0)
    hitX, hitY = x + u * sag, y + v * sag
    nx, ny, nz = (hitX.Scale(-curvature), hitY.Scale(-curvature),
                  sag.Scale(-curvature).AddConstant(1.0))
    norm = (u * u + v * v).AddConstant(1.0).Power(0.5)
    invNorm = norm.Power(-1.0)
    dx, dy, dz = u * invNorm, v * invNorm, invNorm
    cosine = dx * nx + dy * ny + dz * nz
    ratio = incidentRI / outgoingRI
    transmitted = (cosine * cosine).Scale(ratio * ratio).AddConstant(1.0 - ratio * ratio)
    bend = transmitted.Power(0.5) - cosine.Scale(ratio)
    tx, ty, tz = dx.Scale(ratio) + bend * nx, dy.Scale(ratio) + bend * ny, dz.Scale(ratio) + bend * nz
    invZ = tz.Power(-1.0)
    outU, outV = tx * invZ, ty * invZ
    outX, outY = hitX - outU * sag, hitY - outV * sag
    outNorm = (outU * outU + outV * outV).AddConstant(1.0).Power(0.5)
    path = sag * (norm.Scale(incidentRI) - outNorm.Scale(outgoingRI))
    return outX, outY, outU, outV, path


def ComputeHigherOrder(lens, maxOrder=7, referenceField=1.0,
                       wavelength=LambdaLines["d"], objectDistance=math.inf,
                       pupilSemiDiameter=None, stopSemiDiameter=None,
                       referenceFieldUnits=None):
    """Calculate through maxOrder (3, 5 or 7), without modifying the lens.

    referenceField is nonzero: slope expressed in degrees at infinity,
    object height in mm at finite conjugates by default. Explicit "degrees"
    retains the legacy paraxial slope convention at finite conjugates too.
    Normalized field h scales that slope/object height linearly. Physical
    angle is atan(slope); exact Snell directions are normalized internally.

    Pupil coordinates label the paraxial entrance pupil, matching TraceLens.
    stopSemiDiameter fixes its first-order scale, not the exact footprint
    on the physical stop. The Gaussian image excludes sensor defocus.
    Wavelength is nm; ray errors and W coefficients are mm. This is a local
    expansion: fast apertures and wide fields can exceed its useful range.
    Each surface also retains its combined optical-path wavefront and its
    partial-system Gaussian reference, without re-aiming or renormalizing rays.
    """
    fieldUnits = referenceFieldUnits
    if fieldUnits is None:
        fieldUnits = "degrees" if math.isinf(objectDistance) else "mm"
    if fieldUnits == "degrees":
        fieldAngle, objectHeight = referenceField, None
    else:
        fieldAngle, objectHeight = 0.0, referenceField
    paraxial = TraceLens(lens, fieldAngle, wavelength, objectDistance, objectHeight,
                         pupilSemiDiameter, stopSemiDiameter)
    reference = RayReference(paraxial, referenceField, fieldUnits)
    degree = maxOrder + 1
    x, y, u, v, path = reference.Launch(degree)

    # Remaining linear maps: outgoing vertex plane of each surface -> image.
    remaining = [None] * len(lens.surfaces)
    matrix = TranslationMatrix(reference.imageDistance)
    for index in range(len(lens.surfaces) - 1, -1, -1):
        remaining[index] = matrix
        trace = paraxial.surfaces[index]
        refraction = RefractionMatrix(
            (trace.outgoingRI - trace.incidentRI) * trace.curvature,
            trace.incidentRI, trace.outgoingRI)
        matrix = MatMul(matrix, refraction)
        if index > 0:
            matrix = MatMul(matrix, TranslationMatrix(lens.surfaces[index - 1].thickness))

    rows = []
    zero = Taylor.Constant(0.0, degree)
    totalX, totalY = zero, zero
    orders = tuple(range(3, maxOrder + 1, 2))
    for index, surface in enumerate(lens.surfaces):
        trace = paraxial.surfaces[index]
        n, nprime = trace.incidentRI, trace.outgoingRI
        if surface.stopOnly or n == nprime:
            contribution = RayAberrations(zero, zero)
        else:
            curvature = 1.0 / float(surface.radius)
            outX, outY, outU, outV, increment = _RefractSphere(x, y, u, v, curvature, n, nprime)
            linearU = u.Scale(n / nprime) - x.Scale((nprime - n) * curvature / nprime)
            linearV = v.Scale(n / nprime) - y.Scale((nprime - n) * curvature / nprime)
            a, b = remaining[index][0]
            errorX = ((outX - x).Scale(a) + (outU - linearU).Scale(b)).Select(orders)
            errorY = ((outY - y).Scale(a) + (outV - linearV).Scale(b)).Select(orders)
            contribution = RayAberrations(errorX, errorY)
            x, y, u, v = outX, outY, outU, outV
            path = path + increment
        wavefrontReference = (reference.imageReference if index == len(lens.surfaces) - 1
                              else GaussianReference.FromSurface(trace))
        wavefront = wavefrontReference.Wavefront(x, y, u, v, path)
        rows.append(SurfaceHigherOrder(index, contribution, wavefront, wavefrontReference))
        totalX, totalY = totalX + contribution.x, totalY + contribution.y
        if index < len(lens.surfaces) - 1:
            distance = float(surface.thickness)
            path = path + (u * u + v * v).AddConstant(1.0).Power(0.5).AddConstant(-1.0).Scale(nprime * distance)
            x, y = x + u.Scale(distance), y + v.Scale(distance)

    return HigherOrderResult(tuple(rows), RayAberrations(totalX, totalY),
                             wavefront, reference, maxOrder)
