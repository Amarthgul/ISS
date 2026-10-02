

"""
Utilities for tracing paraxial and marginal ray heights through a lens.
"""

from Raytracing.RayBatch import GenerateBeam
from Raytracing.Raypath import RayPath
from Surfaces.Stop import Stop
from Util.Backend import backend as bd
from Util.Globals import AXIAL_ZERO, Axis, FAR_DISTANCE, LambdaLines


def _Scalar(value):
    """
    Convert a scalar from either supported numerical backend to a Python float.
    """
    if hasattr(value, "get"):
        value = value.get()
    return float(value)


def _MarginalStartHeight(lens):
    """
    Return the positive object-side height for the lens's marginal ray.
    """
    pupil = getattr(lens, "entrancePupil", None)

    if pupil is not None:
        # An alpha-masked pupil can have a non-circular edge, so use its actual
        # highest available sample when one exists.
        samples = getattr(pupil, "_pupilPointSamples", None)
        if samples is not None and len(samples) > 0:
            return _Scalar(bd.max(samples[:, Axis.Y.value]))

        if getattr(pupil, "clearSemiDiameter", None) is not None:
            return abs(_Scalar(pupil.clearSemiDiameter))

    if not lens.surfaces:
        return 0.0

    return abs(_Scalar(lens.surfaces[0].clearSemiDiameter))


def _EnsureSurfacePositions(lens):
    """
    Initialize axial vertices for lenses that have not yet been updated.
    """
    if all(surface.frontVertex is not None for surface in lens.surfaces):
        return

    cumulativeThickness = bd.array(0.0)
    for surfaceIndex, surface in enumerate(lens.surfaces):
        surface.SetCumulative(bd.copy(cumulativeThickness))
        if surfaceIndex < len(lens.surfaces) - 1:
            cumulativeThickness += surface.thickness


def _TraceParallelRay(lens, height, wavelength):
    """
    Trace one forward, on-axis-parallel ray and record each physical surface.
    """
    incidentRay = GenerateBeam(
        bd.array([0.0, height, -_Scalar(FAR_DISTANCE)]),
        bd.array([0.0, 0.0, 1.0]),
        size=1,
        wavelength=wavelength,
    )

    rayPath = RayPath()
    rayPath.Append(incidentRay, None, None)
    reachedSurfaces = {}

    for surfaceIndex, surface in enumerate(lens.surfaces):
        if isinstance(surface, Stop):
            continue

        incidentRay, tir, vignetted = surface.NaiveTrace(
            incidentRay,
            lens._FindPreviousRI(surfaceIndex, incidentRay),
        )

        # A vignetted marginal-ray trial may return RayBatch(None). Do not add
        # it to RayPath; returning an incomplete trace triggers bisection in
        # _MarginalRayTrace.
        if incidentRay is None or incidentRay.IsNoneType():
            break

        rayPath.Append(incidentRay, tir, vignetted)
        reachedSurfaces[surfaceIndex] = _Scalar(
            incidentRay.Position()[0, Axis.Y.value]
        )

    physicalSurfaceCount = sum(
        not isinstance(surface, Stop) for surface in lens.surfaces
    )
    exitedLens = len(reachedSurfaces) == physicalSurfaceCount

    return exitedLens, reachedSurfaces, rayPath


def _MarginalRayTrace(lens):
    """
    Trace the highest exiting marginal ray and return its heights and path.

    :return: A tuple of the physical-surface height dictionary and RayPath.
    """
    if not lens.surfaces:
        return {}, RayPath()

    _EnsureSurfacePositions(lens)

    wavelength = LambdaLines["d"]
    initialHeight = _MarginalStartHeight(lens)
    exitedLens, heights, rayPath = _TraceParallelRay(
        lens, initialHeight, wavelength
    )

    if exitedLens:
        return heights, rayPath

    # The axial ray establishes the lower bound for a valid bisection.
    lowerHeight = 0.0
    lowerExited, lowerHeights, lowerPath = _TraceParallelRay(
        lens, lowerHeight, wavelength
    )
    if not lowerExited:
        raise RuntimeError(
            "The axial ray did not exit the lens, so no marginal ray height "
            "can be determined."
        )

    upperHeight = initialHeight
    heights = lowerHeights
    rayPath = lowerPath

    # The fixed iteration cap is a safeguard; normal lenses stop once the
    # bracket reaches the project's axial tolerance.
    for _ in range(30):
        trialHeight = (lowerHeight + upperHeight) / 2.0
        trialExited, trialHeights, trialPath = _TraceParallelRay(
            lens, trialHeight, wavelength
        )

        if trialExited:
            lowerHeight = trialHeight
            heights = trialHeights
            rayPath = trialPath
        else:
            upperHeight = trialHeight

        if upperHeight - lowerHeight <= _Scalar(AXIAL_ZERO):
            break

    return heights, rayPath


def MarginalRayPath(lens):
    """
    Return the recorded path for the highest exiting parallel marginal ray.
    """
    _heights, rayPath = _MarginalRayTrace(lens)
    return rayPath


def MarginalRayHeight(lens):
    """
    Return the height of a marginal, optical-axis-parallel ray at each surface.

    The attempted ray starts at the upper edge of the entrance pupil. If it is
    vignetted before leaving the lens, bisection finds the highest initial ray
    height that exits. Stop surfaces are represented by ``None`` because they
    do not refract the ray.

    :param lens: Lens instance to trace.
    :return: Dictionary mapping each surface index to its signed ray height,
        with ``None`` for stop surfaces.
    """
    if not lens.surfaces:
        return {}

    heights, _rayPath = _MarginalRayTrace(lens)

    return {
        surfaceIndex: None
        if isinstance(surface, Stop)
        else heights.get(surfaceIndex)
        for surfaceIndex, surface in enumerate(lens.surfaces)
    }
