"""First-order tracing for centered refractive photographic objectives.

Lengths are millimeters, wavelengths nanometers, and ray slopes radians
(dy/dz in the paraxial approximation). Tracing uses prescription thicknesses
without changing the lens or invoking its real-ray update machinery.
"""

from dataclasses import dataclass
import math

from Util.Globals import LambdaLines


IDENTITY = ((1.0, 0.0), (0.0, 1.0))


def MatMul(left, right):
    """Compose ray-transfer matrices acting on [height, slope]."""
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(2))
                       for j in range(2)) for i in range(2))


def TranslationMatrix(distance):
    return ((1.0, float(distance)), (0.0, 1.0))


def SurfaceMatrix(surface, previousRI, wavelength):
    """Return refraction matrix and outgoing medium; stops preserve the medium."""
    previousRI = float(previousRI)
    if surface.stopOnly:
        return IDENTITY, previousRI
    currentRI = float(surface.RI(wavelength))
    curvature, _quartic = surface.ParaxialGeometry()
    return ((1.0, 0.0),
            ((previousRI - currentRI) * curvature / currentRI,
             previousRI / currentRI)), currentRI


def SystemMatrix(surfaces, wavelength=LambdaLines["d"], initialRI=1.0):
    """Transfer from just before the first to just after the last surface."""
    matrix = IDENTITY
    previousRI = float(initialRI)
    for index, surface in enumerate(surfaces):
        refraction, previousRI = SurfaceMatrix(surface, previousRI, wavelength)
        matrix = MatMul(refraction, matrix)
        if index < len(surfaces) - 1:
            matrix = MatMul(TranslationMatrix(surface.thickness), matrix)
    return matrix


@dataclass(frozen=True)
class RayState:
    height: float
    slope: float


@dataclass(frozen=True)
class SurfaceTrace:
    surfaceIndex: int
    vertexZ: float
    curvature: float
    quarticSag: float
    incidentRI: float
    outgoingRI: float
    marginalIncident: RayState
    marginalOutgoing: RayState
    chiefIncident: RayState
    chiefOutgoing: RayState


@dataclass(frozen=True)
class ParaxialResult:
    surfaces: tuple[SurfaceTrace, ...]
    wavelength: float
    objectDistance: float
    objectHeight: float
    fieldAngle: float
    pupilSemiDiameter: float
    stopSemiDiameter: float
    stopIndex: int
    lagrangeInvariant: float
    systemMatrix: tuple


def TraceLens(lens, fieldAngle=1.0, wavelength=LambdaLines["d"],
              objectDistance=math.inf, objectHeight=0.0,
              pupilSemiDiameter=None, stopSemiDiameter=None):
    """Trace a marginal/chief pair through the complete prescription.

    ``fieldAngle`` is the signed incident slope expressed in degrees for an
    infinite object (default: a 1-degree reference field). At finite conjugates,
    ``objectDistance`` is positive, measured back from the first vertex, and
    ``objectHeight`` defines the field in mm; fieldAngle is then ignored.

    The chief ray is aimed at lens.stopIndex. An explicit stopSemiDiameter
    scales the marginal ray at that stop. Otherwise pupilSemiDiameter, or the
    current lens.entrancePupil.clearSemiDiameter, sets the object-side pupil
    radius. At finite conjugates this radius is defined at the paraxial
    entrance-pupil plane obtained from the stop transfer matrix.

    The contract is a nonempty centered refractive lens with a defined stop,
    nonsingular aiming, spherical/even-aspheric surfaces, and a finite pupil
    radius when no stop radius is supplied. No clipping or pupil masks apply.
    """
    wavelength = float(wavelength)
    fieldAngle = float(fieldAngle)
    objectDistance = float(objectDistance)
    objectHeight = float(objectHeight)
    initialRI = float(lens.env.RI(wavelength))

    # Only the incident height at the stop is needed for aiming. Refraction
    # at a powered stop changes slope, but leaves this height unchanged.
    stopMatrix = SystemMatrix(lens.surfaces[:lens.stopIndex + 1],
                              wavelength, initialRI)
    a, b = stopMatrix[0]
    pupilZ = b / a
    if math.isinf(objectDistance):
        chiefSlope = math.radians(fieldAngle)
        chiefHeight = -pupilZ * chiefSlope
        marginalSlope = 0.0
        objectHeight = 0.0
    else:
        chiefSlope = -objectHeight / (objectDistance + pupilZ)
        chiefHeight = objectHeight + objectDistance * chiefSlope
        fieldAngle = 0.0

    if stopSemiDiameter is None:
        if pupilSemiDiameter is None:
            pupilSemiDiameter = float(lens.entrancePupil.clearSemiDiameter)
        pupilSemiDiameter = float(pupilSemiDiameter)
        if math.isinf(objectDistance):
            marginalHeight = pupilSemiDiameter
        else:
            marginalSlope = pupilSemiDiameter / (objectDistance + pupilZ)
            marginalHeight = objectDistance * marginalSlope
        stopSemiDiameter = a * marginalHeight + b * marginalSlope
    else:
        stopSemiDiameter = float(stopSemiDiameter)
        if math.isinf(objectDistance):
            marginalHeight = stopSemiDiameter / a
        else:
            marginalSlope = stopSemiDiameter / (a * objectDistance + b)
            marginalHeight = objectDistance * marginalSlope
        pupilSemiDiameter = marginalHeight + pupilZ * marginalSlope

    invariant = initialRI * (marginalHeight * chiefSlope -
                             chiefHeight * marginalSlope)
    histories = []
    matrix = IDENTITY
    vertexZ = 0.0
    previousRI = initialRI
    for index, surface in enumerate(lens.surfaces):
        refraction, currentRI = SurfaceMatrix(surface, previousRI, wavelength)
        curvature, quartic = surface.ParaxialGeometry()
        marginalOut = (refraction[1][0] * marginalHeight +
                       refraction[1][1] * marginalSlope)
        chiefOut = (refraction[1][0] * chiefHeight +
                    refraction[1][1] * chiefSlope)
        histories.append(SurfaceTrace(
            index, vertexZ, curvature, quartic, previousRI, currentRI,
            RayState(marginalHeight, marginalSlope),
            RayState(marginalHeight, marginalOut),
            RayState(chiefHeight, chiefSlope), RayState(chiefHeight, chiefOut)))
        matrix = MatMul(refraction, matrix)
        marginalSlope, chiefSlope = marginalOut, chiefOut
        previousRI = currentRI
        if index < len(lens.surfaces) - 1:
            distance = float(surface.thickness)
            marginalHeight += distance * marginalSlope
            chiefHeight += distance * chiefSlope
            vertexZ += distance
            matrix = MatMul(TranslationMatrix(distance), matrix)

    return ParaxialResult(tuple(histories), wavelength, objectDistance,
                          objectHeight, float(fieldAngle), float(pupilSemiDiameter),
                          float(stopSemiDiameter), lens.stopIndex, invariant, matrix)
