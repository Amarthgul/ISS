"""Monochromatic Seidel sums and reference wavefront coefficients.

The Welford convention is used: S1 spherical, S2 coma, S3 astigmatism,
S4 Petzval field curvature, S5 distortion. Values are unconverted coefficients
in mm, including the chosen pupil/field scale, rather than transverse errors,
distortion percentages, or waves. ComputeSeidelCoefficients additionally
returns per-surface coefficients of the combined field/pupil polynomial at
a fixed reference normalization. Chromatic terms and plotting are separate.
"""

from dataclasses import dataclass, replace
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

    def ToWavefront(self):
        """Convert to pupil-polynomial amplitudes in mm, at this field."""
        return WavefrontCoefficients(self.S1 / 8.0, self.S2 / 2.0,
                                     self.S3 / 2.0, (self.S3 + self.S4) / 4.0,
                                     self.S5 / 2.0)


@dataclass(frozen=True)
class WavefrontCoefficients:
    """Coefficients of rho^4, h*rho^3*cos(phi), h^2*rho^2*cos(phi)^2,
    h^2*rho^2, and h^3*rho*cos(phi), with normalized field h.

    All values are mm, not waves. W220 is the polynomial field-curvature
    coefficient (S3 + S4)/4; the separate Petzval coefficient is S4/4.
    """
    W040: float = 0.0
    W131: float = 0.0
    W222: float = 0.0
    W220: float = 0.0
    W311: float = 0.0

    def AsTuple(self):
        return self.W040, self.W131, self.W222, self.W220, self.W311

    def AsDict(self):
        return {"W040": self.W040, "W131": self.W131, "W222": self.W222,
                "W220": self.W220, "W311": self.W311}


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

    def CumulativeSums(self):
        """Return the selected-field sum through each surface, inclusively."""
        return tuple(self.SumSurfaces(0, index) for index in range(len(self.surfaces)))


@dataclass(frozen=True)
class SurfaceWavefront:
    surfaceIndex: int
    sphericalBase: WavefrontCoefficients
    asphericDeparture: WavefrontCoefficients
    coefficients: WavefrontCoefficients


@dataclass(frozen=True)
class SeidelReferenceResult:
    """Field-independent coefficients for h = field / referenceField.

    Aperture, wavelength, conjugate and nonzero reference field are fixed.
    ``seidel`` retains the corresponding reference-field Seidel sums and rays.
    """
    surfaces: tuple[SurfaceWavefront, ...]
    totals: WavefrontCoefficients
    seidel: SeidelResult
    referenceField: float
    fieldUnits: str
    units: str = "mm"

    def SumSurfaces(self, first=0, last=None):
        """Sum field-independent coefficients over an inclusive surface range."""
        return self.seidel.SumSurfaces(first, last).ToWavefront()

    def CumulativeCoefficients(self):
        """Return coefficient sums through each surface for correction plots."""
        return tuple(value.ToWavefront() for value in self.seidel.CumulativeSums())

    def EvaluateField(self, field):
        """Return selected-field Seidel sums and matching histories, without
        retracing the lens. Field uses fieldUnits; zero and negative fields
        are valid. The reference coefficients remain unchanged.
        """
        field = float(field)
        factor = field / self.referenceField
        reference = self.seidel.paraxial
        histories = tuple(replace(
            row,
            chiefIncident=replace(row.chiefIncident,
                                  height=row.chiefIncident.height * factor,
                                  slope=row.chiefIncident.slope * factor),
            chiefOutgoing=replace(row.chiefOutgoing,
                                  height=row.chiefOutgoing.height * factor,
                                  slope=row.chiefOutgoing.slope * factor))
            for row in reference.surfaces)
        paraxial = replace(reference, surfaces=histories,
                           lagrangeInvariant=reference.lagrangeInvariant * factor)
        if math.isinf(reference.objectDistance):
            paraxial = replace(paraxial, fieldAngle=field)
        else:
            paraxial = replace(paraxial, objectHeight=field)
        return CalculateSeidel(paraxial)


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


def ComputeSeidelCoefficients(lens, referenceField=1.0,
                              wavelength=LambdaLines["d"], objectDistance=math.inf,
                              pupilSemiDiameter=None, stopSemiDiameter=None):
    """Return field-independent wavefront coefficients and reference sums.

    referenceField must be nonzero: degrees for an infinite object, signed
    object height in mm for a finite object. Its default is 1 degree or 1 mm,
    respectively. Field-independent means coefficients of the combined
    field/pupil polynomial for h = field/referenceField, at the fixed aperture,
    wavelength and conjugate. The result can evaluate any selected field,
    including zero, through EvaluateField(), without retracing the lens.
    """
    referenceField = float(referenceField)
    objectDistance = float(objectDistance)
    if math.isinf(objectDistance):
        fieldAngle, objectHeight, fieldUnits = referenceField, 0.0, "degrees"
    else:
        fieldAngle, objectHeight, fieldUnits = 0.0, referenceField, "mm"
    seidel = ComputeSeidel(lens, fieldAngle, wavelength, objectDistance,
                           objectHeight, pupilSemiDiameter, stopSemiDiameter)
    rows = tuple(SurfaceWavefront(row.surfaceIndex, row.sphericalBase.ToWavefront(),
                                 row.asphericDeparture.ToWavefront(),
                                 row.coefficients.ToWavefront())
                 for row in seidel.surfaces)
    return SeidelReferenceResult(rows, seidel.totals.ToWavefront(), seidel,
                                 referenceField, fieldUnits)
