
# Paraxial effective focal length


from Surfaces.Surface import Surface
from Util.Backend import constant
from Util.Globals import LambdaLines, INFINITY, NEAR_ZERO

import math


def _Scalar(value):
    """
    Convert backend scalar arrays to a Python float for small ABCD matrices.
    """
    if hasattr(value, "get"):
        value = value.get()
    return float(value)


def _Wavelength(fraunhoferLine):
    """
    Accept either a Fraunhofer line name or a numeric wavelength in nm.
    """
    if isinstance(fraunhoferLine, str):
        if fraunhoferLine not in LambdaLines:
            raise ValueError(
                f"Unknown Fraunhofer line '{fraunhoferLine}'. "
                f"Expected one of {list(LambdaLines.keys())}."
            )
        return constant(LambdaLines[fraunhoferLine])

    return constant(fraunhoferLine)


def _SurfaceMatrix(surface: Surface, previousRI, wavelength):
    """
    Matrix for refraction at a spherical surface, using ray vector [height, angle].
    """
    currentRI = _Scalar(surface.RI(wavelength))
    radius = _Scalar(surface.radius)

    if math.isinf(radius):
        c = 0.0
    elif abs(radius) <= _Scalar(NEAR_ZERO):
        raise ValueError("Surface radius is too close to zero for paraxial calculation.")
    else:
        c = (previousRI - currentRI) / (radius * currentRI)

    return (
        (1.0, 0.0),
        (c, previousRI / currentRI),
    ), currentRI


def _TranslationMatrix(distance):
    """
    Matrix for axial propagation between adjacent vertices.
    """
    return (
        (1.0, _Scalar(distance)),
        (0.0, 1.0),
    )


def _MatMul(left, right):
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def SingleGroup(surfaces: list[Surface], fraunhoferLine="d"):
    """
    Return the paraxial effective focal length of the given lens group.

    The calculation uses ABCD/ray-transfer matrices with the ray vector
    [height, angle]. A lens group is expected to begin and end in air, matching
    Lens._PartitionGroups(). For an afocal group, INFINITY is returned.
    """
    if len(surfaces) == 0:
        return INFINITY

    wavelength = _Wavelength(fraunhoferLine)
    systemMatrix = (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    previousRI = 1.0

    for i, surface in enumerate(surfaces):
        refractionMatrix, previousRI = _SurfaceMatrix(surface, previousRI, wavelength)
        systemMatrix = _MatMul(refractionMatrix, systemMatrix)

        if i < len(surfaces) - 1:
            translationMatrix = _TranslationMatrix(surface.thickness)
            systemMatrix = _MatMul(translationMatrix, systemMatrix)

    c = systemMatrix[1][0]
    if abs(c) <= _Scalar(NEAR_ZERO):
        return INFINITY

    return constant(-1.0 / c)


def LensPartitionFL(lens, fraunhoferLine="d"):

    if not getattr(lens, "groups", None):
        lens.UpdateLens()

    return [
        SingleGroup([lens.surfaces[surfaceIndex] for surfaceIndex in group], fraunhoferLine)
        for group in lens.groups
    ]

