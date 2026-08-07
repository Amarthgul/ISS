
"""
Aligned lens-layout and prescription-data plotting helpers.
"""

from enum import Enum

import math
import matplotlib.pyplot as plt
import numpy as np

from Surfaces.Stop import Stop
from Util.Backend import constant
from Util.Globals import AXIAL_ZERO, Axis, LambdaLines
from Util.PAEFL import LensPartitionFL
from Util.RayHeight import MarginalRayPath


class SurfaceDataType(Enum):
    RayHeight = 0
    OpticalPower = 1
    RefractiveIndex = 2
    AbbeNumber = 3


displayConfig = [
    SurfaceDataType.RayHeight,
    SurfaceDataType.OpticalPower,
    SurfaceDataType.RefractiveIndex,
    SurfaceDataType.AbbeNumber
]


def PlotSurfaceData(lens, maxPower=None, PlotTrackLength=None):
    """
    Plot an optical layout and the tracks selected by ``displayConfig``.

    The optical layout is always shown. ``SurfaceDataType.RayHeight`` controls
    its marginal-ray overlay; the other display types control the lower tracks.
    ``PlotTrackLength`` frames every aligned axis from the focal point back
    toward object space. When omitted, the lens total axial track length is used.
    """
    _EnsureLensData(lens)
    if not lens.surfaces:
        raise ValueError("PlotSurfaceData requires a lens with at least one surface.")

    fraunhoferLine = "d"
    vertices = _SurfaceVertices(lens)
    selectedTypes = set(displayConfig)
    showRayHeight = SurfaceDataType.RayHeight in selectedTypes
    trackSpecs = []

    if SurfaceDataType.OpticalPower in selectedTypes:
        trackSpecs.append((SurfaceDataType.OpticalPower, 1.0))
    if SurfaceDataType.RefractiveIndex in selectedTypes:
        trackSpecs.append((SurfaceDataType.RefractiveIndex, 1.1))
    if SurfaceDataType.AbbeNumber in selectedTypes:
        trackSpecs.append((SurfaceDataType.AbbeNumber, 1.1))

    materialData = None
    if any(trackType in selectedTypes for trackType in (
            SurfaceDataType.RefractiveIndex,
            SurfaceDataType.AbbeNumber,
    )):
        materialData = _MaterialData(lens, fraunhoferLine, vertices)

    groupData = None
    if SurfaceDataType.OpticalPower in selectedTypes:
        groupData = _GroupData(lens, fraunhoferLine, vertices, maxPower)

    layoutHeight = _LayoutHeight(lens)
    layoutTrackLength = _ResolvedTrackLength(lens, vertices, PlotTrackLength)
    figHeight = 4.8 + sum(height for _trackType, height in trackSpecs) * 1.35
    fig, axes = plt.subplots(
        1 + len(trackSpecs),
        1,
        figsize=(max(10, layoutTrackLength * 0.24), figHeight),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, *(height for _type, height in trackSpecs)]},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    layoutAxis = axes[0]

    focus = _DrawLensLayout(layoutAxis, lens, showRayHeight)
    focus = focus or _LensFocalPoint(lens)
    plotBounds = _PlotBounds(lens, focus, layoutTrackLength)

    for axis, (trackType, _height) in zip(axes[1:], trackSpecs):
        if trackType == SurfaceDataType.OpticalPower:
            _DrawGroupPower(axis, groupData)
        elif trackType == SurfaceDataType.RefractiveIndex:
            _DrawMaterialMetric(axis, materialData, "ri", f"n{fraunhoferLine}", "#3B6FB6", 3, 0.01)
        elif trackType == SurfaceDataType.AbbeNumber:
            _DrawMaterialMetric(axis, materialData, "abbe", "Vd", "#6BBF45", 1, 1.0)

    _ConfigureAlignment(axes, vertices, plotBounds)
    _SetLayoutScale(layoutAxis, layoutHeight)
    _DrawLayoutLengths(layoutAxis, lens, vertices, focus)
    fig.suptitle("Lens Layout and Surface Data")
    _MatchLayoutWidth(layoutAxis, axes[-1])

    if plt.get_backend().lower() != "agg":
        plt.show()

    return fig, axes


# ==================================================================
""" ====================== Private Methods ===================== """
# ==================================================================


def _Scalar(value):
    """Convert a NumPy or CuPy scalar to a Python float."""
    if hasattr(value, "get"):
        value = value.get()
    return float(value)


def _EnsureLensData(lens):
    """Populate groups and axial positions when they have not been calculated."""
    if not lens.surfaces:
        return

    hasRefractiveSurface = any(
        not isinstance(surface, Stop) and not surface.IsAirMaterial()
        for surface in lens.surfaces
    )
    if not lens.groups and hasRefractiveSurface:
        lens.UpdateLens()

    if all(surface.cumulativeThickness is not None for surface in lens.surfaces):
        return

    cumulativeThickness = 0.0
    for surfaceIndex, surface in enumerate(lens.surfaces):
        surface.SetCumulative(cumulativeThickness)
        if surfaceIndex < len(lens.surfaces) - 1:
            cumulativeThickness += _Scalar(surface.thickness)


def _SurfaceVertices(lens):
    """Return the axial vertex position of every surface in millimeters."""
    return [_Scalar(surface.cumulativeThickness) for surface in lens.surfaces]


def _LayoutHeight(lens):
    """Return a finite positive aperture height for the optical layout."""
    finiteSemiDiameters = [
        abs(_Scalar(surface.clearSemiDiameter))
        for surface in lens.surfaces
        if np.isfinite(_Scalar(surface.clearSemiDiameter))
    ]
    return max(finiteSemiDiameters, default=1.0)


def _LensFocalPoint(lens):
    """Return the recorded system focal point as a plotting coordinate."""
    focalPoint = getattr(lens, "focalPoint", None)
    if focalPoint is None:
        return None

    return _Scalar(focalPoint[Axis.Z.value]), 0.0


def _ResolvedTrackLength(lens, vertices, PlotTrackLength):
    """Resolve the focal-point-relative horizontal plot length in millimeters."""
    if PlotTrackLength is None:
        trackLength = getattr(lens, "totalAxialLength", None)
        if trackLength is None:
            trackLength = max(vertices) - min(vertices)
    else:
        trackLength = PlotTrackLength

    trackLength = _Scalar(trackLength)
    if not np.isfinite(trackLength) or trackLength <= _Scalar(AXIAL_ZERO):
        raise ValueError("PlotTrackLength must be a finite, positive length.")

    return trackLength


def _PlotBounds(lens, focus, trackLength):
    """Return object-side-to-focus bounds, preserving a concave front surface."""
    focalZ = focus[0] if focus is not None else trackLength
    objectSideLimit = focalZ - trackLength

    firstSurface = lens.surfaces[0]
    if _Scalar(firstSurface.radius) < 0 and firstSurface.sdCumulative is not None:
        objectSideLimit = min(objectSideLimit, _Scalar(firstSurface.sdCumulative))

    return objectSideLimit, focalZ


def _SetLayoutScale(axis, layoutHeight):
    """Apply a 1:1 physical scale while preserving the shared horizontal bounds."""
    axis.set_ylim(0.0, layoutHeight * 1.12)
    axis.set_aspect("equal", adjustable="box")


def _MatchLayoutWidth(layoutAxis, alignmentAxis):
    """Resize the figure until the equal-scale layout shares its track width."""
    if layoutAxis is alignmentAxis:
        return

    figure = layoutAxis.figure
    xMin, xMax = layoutAxis.get_xlim()
    yMin, yMax = layoutAxis.get_ylim()
    xRange = xMax - xMin
    yRange = yMax - yMin
    if xRange <= 0.0 or yRange <= 0.0:
        return

    # Equal aspect can otherwise letterbox the upper axis while the shared-x
    # data tracks retain their full width. A few constrained-layout passes
    # bring both plotting areas to the same horizontal span.
    for _ in range(3):
        figure.canvas.draw()
        layoutBounds = layoutAxis.get_position()
        alignmentBounds = alignmentAxis.get_position()
        targetWidth = (
            layoutBounds.height
            * figure.get_figheight()
            * xRange
            / (yRange * alignmentBounds.width)
        )

        if abs(targetWidth - figure.get_figwidth()) <= 0.01:
            break
        figure.set_figwidth(targetWidth)


def _LayoutLengthText(lens, vertices, focus):
    """Format total and first-to-last-vertex optical track lengths."""
    opticalLength = max(vertices) - min(vertices)
    if focus is not None:
        totalTrackLength = abs(focus[0] - vertices[0])
    else:
        totalTrackLength = getattr(lens, "totalAxialLength", None)
        if totalTrackLength is None:
            totalTrackLength = opticalLength

    return (
        f"Total track: {_Scalar(totalTrackLength):.2f} mm\n"
        f"Optical length: {opticalLength:.2f} mm"
    )


def _DrawLayoutLengths(axis, lens, vertices, focus):
    """Place the lens-length summary in the upper-right layout corner."""
    axis.text(
        0.985,
        0.975,
        _LayoutLengthText(lens, vertices, focus),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
        zorder=6,
    )


def _SurfaceProfile(surface, sampleCount=80):
    """Return the positive meridional profile of a spherical surface."""
    semiDiameter = abs(_Scalar(surface.clearSemiDiameter))
    vertexZ = _Scalar(surface.cumulativeThickness)
    radius = _Scalar(surface.radius)
    height = np.linspace(0.0, semiDiameter, sampleCount)

    if math.isfinite(radius) and abs(radius) > _Scalar(AXIAL_ZERO):
        sagTerm = np.maximum(radius * radius - height * height, 0.0)
        axial = vertexZ + radius - math.copysign(1.0, radius) * np.sqrt(sagTerm)
    else:
        axial = np.full_like(height, vertexZ)

    return axial, height


def _ElementPolygon(frontSurface, rearSurface):
    """Return a closed positive-half polygon for one glass element."""
    frontZ, frontY = _SurfaceProfile(frontSurface)
    rearZ, rearY = _SurfaceProfile(rearSurface)

    return (
        np.concatenate((frontZ, rearZ[::-1])),
        np.concatenate((frontY, rearY[::-1])),
    )


def _BoundarySegments(lens):
    """Extract positive-half clear-boundary segments from the lens prescription."""
    segments = []
    seen = set()

    for surface in lens.surfaces:
        boundaries = [
            getattr(surface, "clearBoundaryL", None),
            getattr(surface, "clearBoundaryT", None),
            *getattr(surface, "clearBoundaries", []),
        ]

        for boundary in boundaries:
            if boundary is None or id(boundary) in seen:
                continue
            if not hasattr(boundary, "E1") or not hasattr(boundary, "E2"):
                continue

            seen.add(id(boundary))
            try:
                startZ = _Scalar(boundary.E1.ZCoord())
                endZ = _Scalar(boundary.E2.ZCoord())
                startY = _Scalar(boundary.E1.SemiAxisMagnititude())
                endY = _Scalar(boundary.E2.SemiAxisMagnititude())
            except (AttributeError, TypeError):
                continue

            segments.append(((startZ, startY), (endZ, endY)))

    return segments


def _LensElements(lens):
    """Return physical front/rear surface pairs that form glass elements."""
    if lens.lenses:
        return [
            (lens.surfaces[frontIndex], lens.surfaces[rearIndex])
            for frontIndex, rearIndex in lens.lenses
            if rearIndex < len(lens.surfaces)
        ]

    elements = []
    for index, surface in enumerate(lens.surfaces[:-1]):
        if not isinstance(surface, Stop) and not surface.IsAirMaterial():
            elements.append((surface, lens.surfaces[index + 1]))
    return elements


def _MaterialData(lens, fraunhoferLine, vertices):
    """Collect per-surface RI and Abbe data with prescription-aligned spans."""
    wavelength = constant(LambdaLines[fraunhoferLine])
    data = []

    for index, surface in enumerate(lens.surfaces):
        left = vertices[index]
        if index < len(vertices) - 1:
            right = vertices[index + 1]
        else:
            right = left

        ri = None
        abbe = None
        if not isinstance(surface, Stop) and not surface.IsAirMaterial():
            try:
                ri = _Scalar(surface.RI(wavelength))
            except Exception:
                pass

            try:
                abbe = _Scalar(surface.material.V_d())
            except Exception:
                pass

        data.append({
            "index": index,
            "left": left,
            "right": right,
            "ri": ri,
            "abbe": abbe,
        })

    return data


def _GroupData(lens, fraunhoferLine, vertices, maxPower):
    """Collect group focal length, power, color, and horizontal span data."""
    focalLengths = [_Scalar(value) for value in LensPartitionFL(lens, fraunhoferLine)]
    powers = [
        1.0 / focalLength
        if np.isfinite(focalLength) and abs(focalLength) > _Scalar(AXIAL_ZERO)
        else 0.0
        for focalLength in focalLengths
    ]

    if maxPower is None:
        normalizationPower = max((abs(power) for power in powers), default=1.0)
        if normalizationPower <= _Scalar(AXIAL_ZERO):
            normalizationPower = 1.0
    else:
        normalizationPower = _Scalar(maxPower)
        if not np.isfinite(normalizationPower) or normalizationPower <= _Scalar(AXIAL_ZERO):
            raise ValueError("maxPower must be a finite, positive optical power.")

    colors = plt.get_cmap("tab10").colors
    groups = []
    for groupIndex, group in enumerate(lens.groups):
        if not group or groupIndex >= len(focalLengths):
            continue

        groups.append({
            "left": vertices[group[0]],
            "right": vertices[group[-1]],
            "focalLength": focalLengths[groupIndex],
            "normalizedPower": powers[groupIndex] / normalizationPower,
            "color": colors[groupIndex % len(colors)],
        })

    return groups


def _MarginalRayPoints(lens):
    """Return traced marginal-ray points and its forward axial crossing."""
    rayPath = MarginalRayPath(lens)
    physicalIndices = [
        index for index, surface in enumerate(lens.surfaces)
        if not isinstance(surface, Stop)
    ]
    points = []
    finalDirection = None

    for pathIndex, _surfaceIndex in enumerate(physicalIndices, start=1):
        if pathIndex >= len(rayPath.position):
            break

        positions = rayPath.position[pathIndex]
        directions = rayPath.direction[pathIndex]
        if len(positions) == 0:
            break

        point = (
            _Scalar(positions[0, Axis.Z.value]),
            _Scalar(positions[0, Axis.Y.value]),
        )
        points.append(point)
        finalDirection = directions[0]

    focus = None
    if points and finalDirection is not None:
        directionY = _Scalar(finalDirection[Axis.Y.value])
        directionZ = _Scalar(finalDirection[Axis.Z.value])
        if abs(directionY) > _Scalar(AXIAL_ZERO):
            distance = -points[-1][1] / directionY
            if distance > 0:
                focus = (points[-1][0] + distance * directionZ, 0.0)

    return points, focus


def _MetricLimits(values, minimumPadding):
    """Return padded limits matching the existing PlotSurfaceData behavior."""
    if not values:
        return None

    lower = min(values)
    upper = max(values)
    padding = max((upper - lower) * 0.15, minimumPadding)
    return lower - padding, upper + padding, padding


def _StopHeight(lens, stopIndex, layoutHeight):
    """Infer a visible stop height from the nearest physical surfaces."""
    neighboringSemiDiameters = []

    for direction in (-1, 1):
        surfaceIndex = stopIndex + direction
        while 0 <= surfaceIndex < len(lens.surfaces):
            surface = lens.surfaces[surfaceIndex]
            if not isinstance(surface, Stop):
                semiDiameter = abs(_Scalar(surface.clearSemiDiameter))
                if np.isfinite(semiDiameter):
                    neighboringSemiDiameters.append(semiDiameter)
                break
            surfaceIndex += direction

    if neighboringSemiDiameters:
        return min(sum(neighboringSemiDiameters) / len(neighboringSemiDiameters), layoutHeight)

    return layoutHeight


def _DrawLensLayout(axis, lens, showRayHeight):
    """Draw the positive-half lens cross section and optional marginal ray."""
    for frontSurface, rearSurface in _LensElements(lens):
        polygonZ, polygonY = _ElementPolygon(frontSurface, rearSurface)
        axis.fill(
            polygonZ,
            polygonY,
            color="#72C9E8",
            alpha=0.28,
            edgecolor="#6D7780",
            linewidth=1.1,
            zorder=1,
        )

    for surface in lens.surfaces:
        if isinstance(surface, Stop):
            continue
        profileZ, profileY = _SurfaceProfile(surface)
        axis.plot(profileZ, profileY, color="#69737B", linewidth=1.1, zorder=2)

    for start, end in _BoundarySegments(lens):
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#69737B",
            linewidth=1.0,
            zorder=2,
        )

    layoutHeight = _LayoutHeight(lens)
    for surfaceIndex, surface in enumerate(lens.surfaces):
        if not isinstance(surface, Stop):
            continue
        stopHeight = _StopHeight(lens, surfaceIndex, layoutHeight)
        axis.plot(
            [_Scalar(surface.cumulativeThickness)] * 2,
            [0.0, stopHeight],
            color="#424242",
            alpha=0.42,
            linewidth=5.0,
            solid_capstyle="butt",
            zorder=3,
        )

    focus = None
    if showRayHeight:
        rayPoints, focus = _MarginalRayPoints(lens)
        if rayPoints:
            rayZ, rayY = zip(*rayPoints)
            axis.plot(rayZ, rayY, color="#A63832", linewidth=2.0, zorder=4)
            if focus is not None:
                axis.plot(
                    [rayZ[-1], focus[0]],
                    [rayY[-1], focus[1]],
                    color="#A63832",
                    linewidth=2.0,
                    zorder=4,
                )
                axis.scatter([focus[0]], [focus[1]], color="#A63832", s=18, zorder=5)

    axis.axhline(0.0, color="#303030", linewidth=0.9, zorder=3)
    axis.set_ylabel("Height (mm)")

    return focus


def _DrawGroupPower(axis, groups):
    """Draw group-power bars over their physical group spans."""
    axis.axhline(0.0, color="0.25", linewidth=0.8)

    for group in groups:
        width = group["right"] - group["left"]
        if width <= 0:
            continue

        height = group["normalizedPower"]
        axis.bar(
            group["left"],
            height,
            width=width,
            align="edge",
            color=group["color"],
            alpha=0.55,
            edgecolor=group["color"],
        )

        label = (
            "INF"
            if not np.isfinite(group["focalLength"])
            else f"{group['focalLength']:.2f} mm"
        )
        textY = height + 0.06 if height >= 0 else height - 0.06
        axis.text(
            group["right"],
            textY,
            label,
            color=group["color"],
            ha="left",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    axis.set_ylabel("Group power")
    axis.set_ylim(-1.15, 1.15)
    axis.set_yticks([-1, 0, 1])
    axis.set_yticklabels(["-max power", "0", "+max power"])


def _DrawMaterialMetric(axis, data, key, label, color, decimals, minimumPadding):
    """Draw one prescription metric using its physical surface spans."""
    values = [item[key] for item in data if item[key] is not None]
    limits = _MetricLimits(values, minimumPadding)

    for item in data:
        value = item[key]
        width = item["right"] - item["left"]
        if value is None or width <= 0:
            continue

        axis.bar(
            item["left"],
            value,
            width=width,
            align="edge",
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
        )

        if limits is not None:
            axis.text(
                item["left"] + width / 2.0,
                value + limits[2] * 0.25,
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    if limits is None:
        axis.set_ylim((1.4, 2.2) if key == "ri" else (20, 90))
    else:
        axis.set_ylim(limits[0], limits[1])

    axis.set_ylabel(label)


def _ConfigureAlignment(axes, vertices, plotBounds):
    """Apply focal-point-relative x limits and vertex guides to every track."""
    for axis in axes:
        axis.set_xlim(*plotBounds)
        for vertex in vertices:
            axis.axvline(vertex, color="0.82", linewidth=0.8, zorder=0)
        axis.grid(axis="y", color="0.9", linewidth=0.7)

    axes[-1].set_xticks(vertices)
    axes[-1].set_xticklabels([str(index + 1) for index in range(len(vertices))])
    axes[-1].set_xlabel("Surface index at vertex position")

# ==================================================================
""" ====================== Archive Methods ===================== """
# ==================================================================

def PlotSurfaceDataOld(lens, maxPower=None):
    """
    Old version of the surface date plotter without the surface sags.
    """
    fraunhoferLine = "d"
    FLP = 2

    def _Scalar(value):
        if hasattr(value, "get"):
            value = value.get()
        return float(value)

    def _MaterialData(surface):
        if isinstance(surface, Stop) or surface.IsAirMaterial():
            return None, None

        wavelength = constant(LambdaLines[fraunhoferLine])

        try:
            ri = _Scalar(surface.RI(wavelength))
        except Exception:
            ri = None

        try:
            abbe = _Scalar(surface.material.V_d())
        except Exception:
            abbe = None

        return ri, abbe

    if not lens.groups:
        lens.UpdateLens()

    surfaceCount = len(lens.surfaces)
    x = np.arange(surfaceCount)

    riValues = []
    abbeValues = []
    for surface in lens.surfaces:
        ri, abbe = _MaterialData(surface)
        riValues.append(ri)
        abbeValues.append(abbe)

    groupFocalLengths = LensPartitionFL(lens, fraunhoferLine) if lens.groups else []
    groupFocalLengthValues = [_Scalar(fl) for fl in groupFocalLengths]

    groupPowers = [
        1.0 / fl
        if np.isfinite(fl) and abs(fl) > _Scalar(AXIAL_ZERO)
        else 0.0
        for fl in groupFocalLengthValues
    ]
    if maxPower is None:
        normalizationPower = max(
            [abs(power) for power in groupPowers],
            default=1.0
        )
        if normalizationPower <= _Scalar(AXIAL_ZERO):
            normalizationPower = 1.0
    else:
        normalizationPower = _Scalar(maxPower)
        if not np.isfinite(normalizationPower) or normalizationPower <= _Scalar(AXIAL_ZERO):
            raise ValueError("maxPower must be a finite, positive optical power.")

    colors = plt.get_cmap("tab10").colors
    figWidth = max(9, surfaceCount * 0.55)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(figWidth, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    groupAx, riAx, abbeAx = axes

    groupAx.axhline(0, color="0.25", linewidth=0.8)
    for groupIndex, group in enumerate(lens.groups):
        if groupIndex >= len(groupFocalLengthValues) or not group:
            continue

        groupFL = groupFocalLengthValues[groupIndex]
        barHeight = groupPowers[groupIndex] / normalizationPower

        color = colors[groupIndex % len(colors)]
        groupAx.bar(
            group,
            [barHeight] * len(group),
            width=0.82,
            color=color,
            alpha=0.55,
            edgecolor=color,
        )

        textY = barHeight + 0.06 if barHeight >= 0 else barHeight - 0.06
        textVa = "bottom" if barHeight >= 0 else "top"
        label = "INF" if not np.isfinite(groupFL) else f"{groupFL:.{FLP}f} mm"
        groupAx.text(
            group[-1] + 0.42,
            textY,
            label,
            color=color,
            ha="left",
            va=textVa,
            fontsize=9,
            fontweight="bold",
        )

    groupAx.set_ylabel("Group power")
    groupAx.set_ylim(-1.15, 1.15)
    groupAx.set_yticks([-1, 0, 1])
    groupAx.set_yticklabels(["-max power", "0", "+max power"])

    validRI = [(i, v) for i, v in enumerate(riValues) if v is not None]
    if validRI:
        riX, riY = zip(*validRI)
        riAx.bar(riX, riY, width=0.55, color="#3B6FB6", alpha=0.75)
        riMin = min(riY)
        riMax = max(riY)
        riPad = max((riMax - riMin) * 0.15, 0.01)
        riAx.set_ylim(riMin - riPad, riMax + riPad)

        for barX, barY in validRI:
            riAx.text(
                barX,
                barY + riPad * 0.25,
                f"{barY:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#23446F",
            )
    else:
        riAx.set_ylim(1.4, 2.2)

    riAx.set_ylabel(f"n{fraunhoferLine}")

    validAbbe = [(i, v) for i, v in enumerate(abbeValues) if v is not None]
    if validAbbe:
        abbeX, abbeY = zip(*validAbbe)
        abbeAx.bar(abbeX, abbeY, width=0.55, color="#7A9A3A", alpha=0.75)
        abbeMin = min(abbeY)
        abbeMax = max(abbeY)
        abbePad = max((abbeMax - abbeMin) * 0.15, 1.0)
        abbeAx.set_ylim(abbeMin - abbePad, abbeMax + abbePad)

        for barX, barY in validAbbe:
            abbeAx.text(
                barX,
                barY + abbePad * 0.25,
                f"{barY:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#465C23",
            )
    else:
        abbeAx.set_ylim(20, 90)

    abbeAx.set_ylabel("Vd")

    for ax in axes:
        ax.set_xlim(-0.5, surfaceCount - 0.5)
        ax.set_xticks(x)
        ax.set_xticks(np.arange(-0.5, surfaceCount, 1), minor=True)
        ax.grid(axis="x", which="minor", color="0.82", linewidth=0.8)
        ax.grid(axis="y", color="0.9", linewidth=0.7)

    surfaceLabels = [
        "STOP" if isinstance(surface, Stop) else str(i + 1)
        for i, surface in enumerate(lens.surfaces)
    ]
    abbeAx.set_xticklabels(surfaceLabels)
    abbeAx.set_xlabel("Surface")
    fig.suptitle("Surface and Group Data")

    if plt.get_backend().lower() != "agg":
        plt.show()

    return fig, axes
