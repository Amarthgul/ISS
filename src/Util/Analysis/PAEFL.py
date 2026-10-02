
# Paraxial effective focal length


from Surfaces.Surface import Surface
from Util.Backend import constant
from Util.Globals import LambdaLines, INFINITY, NEAR_ZERO
from .Paraxial import SystemMatrix


def _Wavelength(fraunhoferLine):
    """
    Accept either a Fraunhofer line name or a numeric wavelength in nm.
    """
    if isinstance(fraunhoferLine, str):
        return constant(LambdaLines[fraunhoferLine])

    return constant(fraunhoferLine)


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
    systemMatrix = SystemMatrix(surfaces, wavelength)

    c = systemMatrix[1][0]
    if abs(c) <= NEAR_ZERO:
        return INFINITY

    return constant(-1.0 / c)


def LensPartitionFL(lens, fraunhoferLine="d"):

    if not lens.groups:
        lens.UpdateLens()

    return [
        SingleGroup([lens.surfaces[surfaceIndex] for surfaceIndex in group], fraunhoferLine)
        for group in lens.groups
    ]

