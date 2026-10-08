

"""Finite-conjugate Seidel analysis for centered refractive lenses.

The default conjugates give an inverted 1:1 image: both planes are 2*EFL
from their respective principal planes. Lengths/coefficients are mm and
wavelengths nm. A focal length alone cannot determine aberrations; the full
prescription, stop, aperture and field are required. No lens update or
prescription changes are performed.
"""

from dataclasses import dataclass
import math

from Util.Globals import LambdaLines
from .Paraxial import SystemMatrix
from .Seidel import ComputeSeidelCoefficients, SeidelReferenceResult


@dataclass(frozen=True)
class EqualConjugateResult:
    reference: SeidelReferenceResult
    effectiveFocalLength: float
    frontPrincipalPlaneZ: float
    rearPrincipalPlaneZ: float
    objectDistance: float  # Positive distance back from the first vertex.
    imageDistance: float  # Requested distance forward from the last vertex.
    gaussianImageDistance: float
    magnification: float  # At the Gaussian image, not a defocused sensor.

    @property
    def seidel(self):
        return self.reference.seidel

    @property
    def totals(self):
        """Wavefront coefficients W040, W131, W222, W220, W311 in mm."""
        return self.reference.totals

    @property
    def objectZ(self):
        return -self.objectDistance

    @property
    def imageZ(self):
        return self.seidel.paraxial.surfaces[-1].vertexZ + self.imageDistance

    @property
    def imageDefocus(self):
        """Requested image plane minus Gaussian image plane, in mm.

        This mismatch is reported separately; no W020 is added to Seidel.
        """
        return self.imageDistance - self.gaussianImageDistance

    @property
    def isEqualConjugate(self):
        return (math.isclose(self.magnification, -1.0, abs_tol=1e-9)
                and math.isclose(self.imageDefocus, 0.0, abs_tol=1e-9))

    def Report(self):
        """Return conjugates, aperture and both S/W conventions as text."""
        trace = self.seidel.paraxial
        objectPrincipal = self.frontPrincipalPlaneZ - self.objectZ
        imagePrincipal = self.imageZ - self.rearPrincipalPlaneZ
        lines = [
            "1:1 conjugate Seidel analysis" if self.isEqualConjugate
            else "Finite-conjugate Seidel analysis (placement is not focused 1:1)",
            f"EFL: {self.effectiveFocalLength:.8g} mm; magnification: {self.magnification:.8g}",
            f"Object / image distances from principal planes: {objectPrincipal:.8g} / {imagePrincipal:.8g} mm",
            f"Object / image distances from outer vertices: {self.objectDistance:.8g} / {self.imageDistance:.8g} mm",
            f"Gaussian image distance from last vertex: {self.gaussianImageDistance:.8g} mm; image defocus: {self.imageDefocus:.8g} mm",
            f"Reference object height: {self.reference.referenceField:g} mm; wavelength: {trace.wavelength:g} nm",
            f"Entrance-pupil / stop semi-diameters: {trace.pupilSemiDiameter:.8g} / {trace.stopSemiDiameter:.8g} mm",
            "Welford Seidel sums (mm):",
        ]
        names = ("spherical", "coma", "astigmatism", "Petzval", "distortion")
        lines.extend(f"  S{i}: {value:.8g} ({name})" for i, (name, value)
                     in enumerate(zip(names, self.seidel.totals.AsTuple()), 1))
        lines.append("Wavefront coefficients (mm; normalized pupil and object field):")
        lines.extend(f"  {key}: {value:.8g}" for key, value in self.totals.AsDict().items())
        lines.append("W220 = (S3 + S4)/4; axial defocus and chromatic aberrations are excluded.")
        return "\n".join(lines)


def _Positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def ComputeEqualConjugateSeidel(lens, referenceField=1.0,
                                wavelength=LambdaLines["d"],
                                objectDistance=None, imageDistance=None,
                                pupilSemiDiameter=None, stopSemiDiameter=None,
                                distanceReference="principal"):
    """Calculate 1:1 Seidel sums and reference wavefront coefficients.

    With omitted distances, solve exact unit magnification from the ABCD
    matrix, independent of lens.focalLength and real-ray best focus. Explicit
    distances use principal planes by default (100/100 for EFL=50 mm); choose
    distanceReference="vertex" for distances from the first/last surfaces.
    An omitted imageDistance follows the object's Gaussian image. Explicit
    placements are allowed and report their actual magnification and defocus,
    rather than assuming that equal vertex distances imply a focused 1:1 image.

    referenceField is a nonzero signed object height in mm, unlike the degrees
    used by _test.EFL. The physical stop radius is used by default, preserving
    the aperture at finite conjugates; pupilSemiDiameter can override it.
    Returned totals are W coefficients; result.seidel.totals contains S sums.
    Supports positive-power centered refractive systems with equal exterior
    indices, a valid stop, and real object/image planes outside the lens.
    """
    if not lens.surfaces:
        raise ValueError("A nonempty lens prescription is required.")
    if lens.stopIndex is None or not 0 <= lens.stopIndex < len(lens.surfaces):
        raise ValueError("A valid lens.stopIndex is required.")
    if distanceReference not in ("principal", "vertex"):
        raise ValueError('distanceReference must be "principal" or "vertex".')
    wavelength = _Positive(wavelength, "wavelength")
    referenceField = float(referenceField)
    if not math.isfinite(referenceField) or referenceField == 0.0:
        raise ValueError("referenceField must be finite and nonzero (object height in mm).")
    if pupilSemiDiameter is not None and stopSemiDiameter is not None:
        raise ValueError("Supply either pupilSemiDiameter or stopSemiDiameter, not both.")
    if any(not getattr(surface, "IsOnAxis", True) or
           float(surface.thickness) < 0.0 or
           getattr(getattr(surface, "material", None), "name", "") == "MIRROR"
           for surface in lens.surfaces):
        raise ValueError("Only centered forward refractive prescriptions are supported.")

    initialRI = float(lens.env.RI(wavelength))
    matrix = SystemMatrix(lens.surfaces, wavelength, initialRI)
    (a, b), (c, d) = matrix
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("The system matrix must be finite.")
    if not math.isclose(a*d - b*c, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Equal-conjugate analysis requires equal object/image refractive indices.")
    if c >= 0.0:
        raise ValueError("A positive-power, non-afocal lens is required.")
    focalLength = -1.0 / c
    lastVertex = math.fsum(float(s.thickness) for s in lens.surfaces[:-1])
    frontPrincipal = (d - 1.0) / c
    rearPrincipal = lastVertex + (1.0 - a) / c

    if objectDistance is None:
        objectVertex = (-1.0 - d) / c
    else:
        objectVertex = _Positive(objectDistance, "objectDistance")
        if distanceReference == "principal":
            objectVertex -= frontPrincipal
    objectVertex = _Positive(objectVertex, "Object distance from first vertex")
    denominator = c*objectVertex + d
    if math.isclose(denominator, 0.0, abs_tol=1e-14):
        raise ValueError("This object conjugate has an image at infinity.")
    gaussianImage = -(a*objectVertex + b) / denominator
    gaussianImage = _Positive(gaussianImage, "Gaussian image distance from last vertex")
    if imageDistance is None:
        imageVertex = gaussianImage
    else:
        imageVertex = _Positive(imageDistance, "imageDistance")
        if distanceReference == "principal":
            imageVertex += rearPrincipal - lastVertex
    imageVertex = _Positive(imageVertex, "Image distance from last vertex")

    if pupilSemiDiameter is not None:
        pupilSemiDiameter = _Positive(pupilSemiDiameter, "pupilSemiDiameter")
    else:
        stopSemiDiameter = _Positive(
            lens.surfaces[lens.stopIndex].clearSemiDiameter
            if stopSemiDiameter is None else stopSemiDiameter, "stopSemiDiameter")
    stopMatrix = SystemMatrix(lens.surfaces[:lens.stopIndex + 1], wavelength, initialRI)
    stopA, stopB = stopMatrix[0]
    if (math.isclose(stopA, 0.0, abs_tol=1e-14) or
            math.isclose(stopA*objectVertex + stopB, 0.0, abs_tol=1e-14)):
        raise ValueError("The entrance-pupil/stop aiming is singular at this conjugate.")

    reference = ComputeSeidelCoefficients(
        lens, referenceField=referenceField, wavelength=wavelength,
        objectDistance=objectVertex, pupilSemiDiameter=pupilSemiDiameter,
        stopSemiDiameter=stopSemiDiameter, referenceFieldUnits="mm")
    return EqualConjugateResult(reference, focalLength, frontPrincipal,
                               rearPrincipal, objectVertex, imageVertex,
                               gaussianImage, 1.0 / denominator)


def PlotEqualConjugateSeidel(lens, *, result=None, show=True, **settings):
    """Plot finite ray geometry and five cumulative W tracks; return fig, axes, result.

    settings are ComputeEqualConjugateSeidel arguments. Pass a previously
    computed result to reuse its rays/coefficients, without retracing. The
    upper panel shows the full object/image geometry; the five lower panels
    share an expanded prescription axis so individual corrections are legible.
    Cumulative coefficients are bars starting at each surface vertex and
    spanning its following axial interval; the last uses the final nonzero gap.
    These are paraxial rays and primary aberration coefficients, without
    aperture clipping. Set show=False to save or embed the returned figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if result is not None and settings:
        raise ValueError("Supply a result or calculation settings, not both.")
    if result is None:
        result = ComputeEqualConjugateSeidel(lens, **settings)
    traces = result.seidel.paraxial.surfaces
    fig, axes = plt.subplots(6, 1, figsize=(13, 12), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1, 1]})
    layout = axes[0]
    vertices = [row.vertexZ for row in traces]
    for surface, row in zip(lens.surfaces, traces):
        radius = float(surface.clearSemiDiameter)
        layout.plot([row.vertexZ]*2, [-radius, radius], color="0.55", linewidth=2)
    z = [result.objectZ, *vertices, result.imageZ]
    for ray, color, label in (("marginal", "#A63832", "Marginal ray"),
                              ("chief", "#3B8E53", "Chief ray")):
        states = [getattr(row, ray + "Incident") for row in traces]
        last = getattr(traces[-1], ray + "Outgoing")
        heights = ([0.0 if ray == "marginal" else result.reference.referenceField]
                   + [state.height for state in states]
                   + [last.height + result.imageDistance*last.slope])
        layout.plot(z, heights, color=color, label=label, marker=".")
        if ray == "marginal":
            layout.plot(z, [-value for value in heights], color=color, alpha=.5)
    for plane, label, color in ((result.objectZ, "Object", "0.25"),
                                (result.imageZ, "Image", "0.25"),
                                (result.frontPrincipalPlaneZ, "H1", "#8064AD"),
                                (result.rearPrincipalPlaneZ, "H2", "#2696A1")):
        layout.axvline(plane, color=color, linestyle="--", linewidth=.8, label=label)
    if not math.isclose(result.imageDefocus, 0.0, abs_tol=1e-9):
        layout.axvline(result.seidel.paraxial.objectImageZ, color="orange",
                       linestyle=":", label="Gaussian image")
    layout.axhline(0, color="0.8", linewidth=.7)
    layout.set(xlabel="Z from first surface (mm)", ylabel="Height (mm)")
    layout.legend(loc="upper center", ncol=6, fontsize=8)
    cumulative = np.array([row.AsTuple() for row in result.reference.CumulativeCoefficients()])
    gaps = np.diff(vertices)
    positiveGaps = gaps[gaps > 0.0]
    finalWidth = float(positiveGaps[-1]) if positiveGaps.size else 1.0
    widths = np.append(gaps, finalWidth)
    colors = ("#A63832", "#3B8E53", "#8064AD", "#2696A1", "#D29A2E")
    for index, (key, color, axis) in enumerate(zip(result.totals.AsDict(), colors, axes[1:])):
        if index:
            axis.sharex(axes[1])
        axis.axhline(0, color="0.55", linewidth=.7)
        axis.bar(vertices, cumulative[:, index], width=widths, align="edge",
                 color=color, edgecolor=color, alpha=.75)
        axis.set_ylabel(f"Σ {key} (mm)")
        axis.set_title(f"Total {key} = {result.totals.AsTuple()[index]:.6g} mm", loc="right", fontsize=9)
        axis.grid(alpha=.2)
        # Adjacent cemented surfaces can be only 0.13 mm apart. Keep regular
        # readable Z labels and mark individual vertices with minor ticks.
        axis.set_xticks(vertices, minor=True)
        axis.grid(axis="x", which="minor", alpha=.1)
        axis.tick_params(axis="x", labelbottom=index == 4)
    axes[-1].set_xlabel("Surface vertex Z (mm); cumulative correction through each surface")
    mode = "1:1 conjugate" if result.isEqualConjugate else "Finite conjugate"
    fig.suptitle(f"{mode} Seidel analysis — EFL {result.effectiveFocalLength:.6g} mm; "
                 f"m = {result.magnification:.6g}\n"
                 f"Object height {result.reference.referenceField:g} mm; "
                 f"λ = {result.seidel.paraxial.wavelength:g} nm; "
                 f"stop radius {result.seidel.paraxial.stopSemiDiameter:.6g} mm; "
                 f"image defocus {result.imageDefocus:.6g} mm")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    return fig, axes, result
