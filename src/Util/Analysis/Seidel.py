"""Five monochromatic Seidel sums from full-system paraxial rays.

The Welford convention is used: S1 spherical, S2 coma, S3 astigmatism,
S4 Petzval field curvature, S5 distortion. Values are unconverted coefficients
in mm, including the chosen pupil/field scale, rather than transverse errors,
distortion percentages, or waves. Chromatic terms and plotting are separate.
"""

from dataclasses import dataclass
import math

from Util.Globals import LambdaLines
from .Paraxial import TraceLens, ParaxialResult


@dataclass(frozen=True)
class SeidelCoefficients:
    S1: float = 0.0
    S2: float = 0.0
    S3: float = 0.0
    S4: float = 0.0
    S5: float = 0.0

    def AsTuple(self):
        """Return values in S1, S2, S3, S4, S5 order."""
        return self.S1, self.S2, self.S3, self.S4, self.S5

    def AsDict(self):
        return {"spherical": self.S1, "coma": self.S2,
                "astigmatism": self.S3, "fieldCurvature": self.S4,
                "distortion": self.S5}


@dataclass(frozen=True)
class SurfaceSeidel:
    surfaceIndex: int
    sphericalBase: SeidelCoefficients
    asphericDeparture: SeidelCoefficients
    coefficients: SeidelCoefficients


@dataclass(frozen=True)
class SeidelResult:
    surfaces: tuple[SurfaceSeidel, ...]
    totals: SeidelCoefficients
    paraxial: ParaxialResult
    units: str = "mm"
    convention: str = "Welford unconverted Seidel sums; S4 is Petzval"

    def SumSurfaces(self, first=0, last=None):
        """Sum an inclusive zero-based range, retaining the full-system rays."""
        if last is None:
            last = len(self.surfaces) - 1
        return _Sum(row.coefficients for row in self.surfaces[first:last + 1])


def _Sum(coefficients):
    rows = tuple(row.AsTuple() for row in coefficients)
    return SeidelCoefficients(*(math.fsum(row[index] for row in rows)
                                for index in range(5)))


def CalculateSeidel(paraxial):
    """Calculate contributions from a TraceLens result, without retracing.

    A = n(u + cy), Abar = n(ubar + cybar), H = n(y*ubar - ybar*u).
    The distortion expression is expanded to avoid division by A; likewise
    aspheric corrections use height products rather than ybar/y. Normal
    marginal incidence and zero marginal height therefore remain regular.
    """
    rows = []
    invariant = paraxial.lagrangeInvariant
    for trace in paraxial.surfaces:
        y, u = trace.marginalIncident.height, trace.marginalIncident.slope
        ybar, ubar = trace.chiefIncident.height, trace.chiefIncident.slope
        n, nprime = trace.incidentRI, trace.outgoingRI
        curvature = trace.curvature
        a = n * (u + curvature * y)
        abar = n * (ubar + curvature * ybar)
        delta = trace.marginalOutgoing.slope / nprime - u / n
        petzval = curvature * (1.0 / nprime - 1.0 / n)
        base = SeidelCoefficients(
            -a * a * y * delta,
            -a * abar * y * delta,
            -abar * abar * y * delta,
            -invariant * invariant * petzval,
            -abar * (abar * abar * y * (1.0 / nprime**2 - 1.0 / n**2) -
                     (2.0 * y * abar - ybar * a) * ybar * petzval))

        # Subtract the sphere having the actual vertex curvature. This also
        # handles A2: its curvature change alters both the ray trace and the
        # reference sphere's quartic sag, not merely an added A4 coefficient.
        departure = trace.quarticSag - curvature**3 / 8.0
        scale = 8.0 * (nprime - n) * departure
        aspheric = SeidelCoefficients(scale * y**4, scale * y**3 * ybar,
                                      scale * y**2 * ybar**2, 0.0,
                                      scale * y * ybar**3)
        rows.append(SurfaceSeidel(trace.surfaceIndex, base, aspheric,
                                  _Sum((base, aspheric))))
    rows = tuple(rows)
    return SeidelResult(rows, _Sum(row.coefficients for row in rows), paraxial)


def ComputeSeidel(lens, fieldAngle=1.0, wavelength=LambdaLines["d"],
                  objectDistance=math.inf, objectHeight=0.0,
                  pupilSemiDiameter=None, stopSemiDiameter=None):
    """Return per-surface coefficients, totals, settings, and ray histories.

    See TraceLens for aperture/conjugate definitions. Wavelength is numeric
    nanometers; fieldAngle is degrees (default 1). Use objectHeight in mm for
    finite conjugates. No lens update, real-ray clipping, or imager is required.
    """
    return CalculateSeidel(TraceLens(
        lens, fieldAngle, wavelength, objectDistance, objectHeight,
        pupilSemiDiameter, stopSemiDiameter))
