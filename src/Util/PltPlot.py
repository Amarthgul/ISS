
"""
This module is used to provide preliminary visuals for the project, it is not the best for accuracy nor effect.
In the future the project should consider switching to mayavi or better libaraies for scientific visulization.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import ORIGIN, INFINITY, Axis, THETA_DIV, RENDER_MODE
from Util.ColorWavelength import ColorTuplePLT


# ==================================================================
""" ============================================================ """
# ==================================================================


# Matplotlib z axis is always shortened
zAxisCompensationFactor = 1.25

fig = None
AX = None


def _as_numpy(value):
    if backend_name == "cupy" and hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _as_float(value):
    return float(_as_numpy(value))


def _is_infinite(value):
    return np.isinf(_as_float(value))


def Setup3Dplot():
    if(RENDER_MODE):
        return None

    global fig, AX
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure()

    if AX is None or AX.figure is not fig:
        AX = fig.add_subplot(111, projection='3d')
        AX.set_proj_type('ortho')

    return AX



def Reset2D():
    if(RENDER_MODE):
        return

    global fig, AX
    fig = plt.figure()
    AX = None


def CheckAX(ax=None):
    """
    This function checks if the axis is initialized.
    Such that the user does not have to call Setup3Dplot() manually and could add plot through the entire program.
    """
    if(RENDER_MODE):
        return None

    if ax is not None:
        return ax

    return Setup3Dplot()


# ==================================================================
""" ============================================================ """
# ==================================================================

def RemoveBG(ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    ax.figure.patch.set_alpha(0)
    ax.grid(False)  # Remove grid
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks
    ax.set_zticks([])  # Remove z ticks

    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Remove the lines
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # Optional: Remove background panels
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def AddXYZ(unitLength = 10, lineWidth = 1, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    ax.plot([0, unitLength], [0, 0], [0, 0], label = '3D Line', color = 'r', linewidth = lineWidth)
    ax.plot([0, 0], [0, unitLength], [0, 0], label = '3D Line', color = 'g', linewidth = lineWidth)
    ax.plot([0, 0], [0, 0], [0, unitLength], label = '3D Line', color = 'b', linewidth = lineWidth)


def SetUnifScale(lim = 10, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    offsetScalar = zAxisCompensationFactor * lim
    ax.set_xlim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_ylim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_zlim(lim, 0)


def DrawPoint(point, color='red', ax=None):
    """
    Draw a single point.
    """
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    point = _as_numpy(point)

    ax.scatter3D(point[0], point[1], point[2], color=color)


def DrawPoints(points, ptSize=0.5, color='red', ax=None):
    """
    Draw an array of points.
    """

    if(RENDER_MODE):
        return

    ax = CheckAX(ax)

    points = _as_numpy(points)


    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter3D(x, y, z, color=color, s=ptSize)


def DrawPointsPerColor(points, color, ax=None):
    """
    This is basically the same as DrawPoints(), but the color parameter is treated as an array of same dimension as points.
    """
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)

    points = _as_numpy(points)
    if(backend_name == "cupy"):
        color = ColorTuplePLT(color)



    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter3D(x, y, z, c=color, s=0.5)


def Draw3D(x, y, z, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    ax.plot(x, y, z)


def DrawLine(point1, point2, lineColor = "k", lineWidth = 2, zorder=10, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]],
            label = '3D Line', color = lineColor, linewidth = lineWidth, zorder=zorder)


def DrawLines(pointSetOne, pointSetTwo, lineColor = "k", lineWidth = 0.5, zorder=10, ax=None):
    if(RENDER_MODE):
        return

    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ax = CheckAX(ax)
    start = pointSetOne  # [x_start, y_start, z_start]
    end = pointSetTwo    # [x_end, y_end, z_end]

    start = _as_numpy(start)
    end = _as_numpy(end)

    # Create segments for Line3DCollection
    segments = np.array([[s, e] for s, e in zip(start, end)])

    line_collection = Line3DCollection(segments, linewidths=lineWidth, color=lineColor)
    ax.add_collection3d(line_collection)


def DrawCircle(radius, offset = 0, num_points=100, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    radius = _as_float(radius)
    offset = _as_float(offset)
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta) + offset])
    ax.plot(circle_points[0], circle_points[1], circle_points[2])


def DrawIncidentPlane(posA, posB, posC, posP, d, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    originOffset = np.array([_as_float(ORIGIN[0]), _as_float(ORIGIN[1]), _as_float(posA[2])])
    DrawLine(originOffset, posA, lineWidth = 1, ax=ax)
    DrawLine(originOffset, posC, lineWidth = 1, ax=ax)
    DrawLine(posP, posA, lineWidth = 1, ax=ax)
    DrawLine(posP, posC, lineWidth = 1, ax=ax)
    DrawLine(posA, posB, lineWidth = 1, ax=ax)


def DrawRaybatch(rayBatch, lineColor='blue', lLength = 10, arrowRatio=.1, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    data = _as_numpy(rayBatch.value)

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    u, v, w = data[:, 3], data[:, 4], data[:, 5]

    q = ax.quiver(x, y, z, u, v, w,
              length=lLength,                # Increase arrow length
              normalize=False,           # Maintain relative vector sizes
              arrow_length_ratio=arrowRatio,    # Smaller arrowhead
              pivot='tail',              # Arrows start at [x,y,z]
              linewidths=0.5,            # Thicker arrows
              color=lineColor)


def DrawNormal(intersections, normals, lineColor='green', lineLength=1, lineWidths = 0.5, arrowRatio=0.1, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    intersections = _as_numpy(intersections)
    normals = _as_numpy(normals)

    x, y, z = intersections[:, 0], intersections[:, 1], intersections[:, 2]
    u, v, w = normals[:, 0], normals[:, 1], normals[:, 2]

    q = ax.quiver(x, y, z, u, v, w,
              length=lineLength,                # Increase arrow length
              normalize=False,                  # Maintain relative vector sizes
              arrow_length_ratio=arrowRatio,    # Smaller arrowhead
              pivot='tail',                     # Arrows start at [x,y,z]
              linewidths=lineWidths,
              color=lineColor)


def DrawDirection(position, direction, lineColor='green', lineLength=5, arrowRatio=0.1, ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    position = _as_numpy(position)
    direction = _as_numpy(direction)

    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    u, v, w = direction[:, 0], direction[:, 1], direction[:, 2]

    q = ax.quiver(x, y, z, u, v, w,
              length=lineLength,                # Increase arrow length
              normalize=False,                  # Maintain relative vector sizes
              arrow_length_ratio=arrowRatio,    # Smaller arrowhead
              pivot='tail',                     # Arrows start at [x,y,z]
              linewidths=0.5,                   # Thicker arrows
              color=lineColor)


def DrawSpherical(radius, clearSemiDiameter, cumulativeThickness, numPoints = THETA_DIV, surfaceColor = "k", opacity=0.1,  ax=None):
    """
    Draw a spherical surface along the z axis.

    :param ax: axis to draw on.
    :param radius: radius of the spherical surface.
    :param clearSemiDiameter: clear semi diameter. Surface will be trimed around it.
    :param cumulativeThickness: Cumulative thickness from 1st surface.
    :param numPoints: number of points, controls the subdivision of the surface.
    :surfaceColor: color of the surface.
    """
    if(RENDER_MODE):
        return


    radius = _as_float(radius)
    clearSemiDiameter = _as_float(clearSemiDiameter)
    cumulativeThickness = _as_float(cumulativeThickness)

    if(_is_infinite(radius)):
        return DrawDisk(clearSemiDiameter, cumulativeThickness, numPoints, surfaceColor, ax=ax)

    ax = CheckAX(ax)
    unsignedrRadius = radius * np.sign(radius)

    radianLimit = np.arcsin(clearSemiDiameter/unsignedrRadius)

    theta = np.linspace(0, 2 * np.pi, int(numPoints / 1))  # Azimuthal angle (0 <= theta <= 2*pi)
    phi = np.linspace(0, radianLimit, int(numPoints / 1))  # Polar angle (0 <= phi <= phi_max for a bowl)

    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = - np.sign(radius) * (unsignedrRadius * np.cos(phi) - unsignedrRadius) + cumulativeThickness

    ax.plot_surface(x, y, z, color = surfaceColor, alpha =opacity)


def DrawSphericalInner(radius, maxAperture, minAperture, cumulativeThickness,
                       numPoints=THETA_DIV, surfaceColor="k", opacity=0.1, ax=None):
    """
    Draw a spherical surface along the z axis, but with a hole in the middle.
    The hole is defined by minAperture (inner radius), and the outer edge by maxAperture.

    This is a parallel of DrawSpherical(), except it draws only the annulus region.
    """
    if (RENDER_MODE):
        return

    radius = _as_float(radius)
    maxAperture = _as_float(maxAperture)
    minAperture = _as_float(minAperture)
    cumulativeThickness = _as_float(cumulativeThickness)

    if _is_infinite(radius):
        return DrawDiskInner(maxAperture, minAperture, cumulativeThickness, numPoints, surfaceColor, opacity, ax=ax)

    ax = CheckAX(ax)

    # Safety clamps
    if minAperture < 0:
        minAperture = 0.0
    if maxAperture < minAperture:
        maxAperture = minAperture

    # Finite radius: spherical annulus cap
    unsignedRadius = radius * np.sign(radius)  # == abs(radius)
    # Clamp apertures so arcsin() stays valid
    maxA = np.minimum(maxAperture, unsignedRadius * 0.999999)
    minA = np.minimum(minAperture, unsignedRadius * 0.999999)

    # Convert radial limits into polar-angle limits on the sphere
    phi_min = np.arcsin(minA / unsignedRadius)
    phi_max = np.arcsin(maxA / unsignedRadius)

    theta = np.linspace(0, 2 * np.pi, int(numPoints / 1))
    phi = np.linspace(phi_min, phi_max, int(numPoints / 1))

    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = -np.sign(radius) * (unsignedRadius * np.cos(phi) - unsignedRadius) + cumulativeThickness

    ax.plot_surface(x, y, z, color=surfaceColor, alpha=opacity)


def DrawAspherical(radius, k, A, clearSemiDiameter, cumulativeThickness,
                   numPoints=THETA_DIV, surfaceColor="k", opacity=0.1, ax=None):
    """
    Draw an aspherical surface along the z axis.

    Parameters:
        radius (float): Radius of curvature.
        k (float): Conic constant.
        A (list): Even aspheric coefficients (A4, A6, A8, ...).
        clearSemiDiameter (float): Clear semi diameter.
        cumulativeThickness (float): Axial offset from origin.
        numPoints (int): Number of sampling points per axis.
        surfaceColor (str): Surface color.
        opacity (float): Surface opacity.
        ax: Plot axis.
    """
    if RENDER_MODE:
        return

    ax = CheckAX(ax)

    radius = _as_float(radius)
    k = _as_float(k)
    A = _as_numpy(A)
    clearSemiDiameter = _as_float(clearSemiDiameter)
    cumulativeThickness = _as_float(cumulativeThickness)

    r = np.linspace(0, clearSemiDiameter, numPoints)
    theta = np.linspace(0, 2 * np.pi, numPoints)
    R, Theta = np.meshgrid(r, theta)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    R2 = X**2 + Y**2

    sqrt_term = np.sqrt(1 - (1 + k) * R2 / radius**2)
    base = R2 / (radius * (1 + sqrt_term))
    asphere = np.zeros_like(R2)
    for i, a in enumerate(A):
        # This starts from the 2nd order term
        asphere += a * R2 ** (i + 1)

    Z = base + asphere + cumulativeThickness

    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=opacity)


def DrawBiconicSurface(radiusX, kX, radiusY, kY, clearSemiDiameter, cumulativeThickness,
                       ySemi=None, isSweep=True, numPoints=THETA_DIV,
                       surfaceColor="k", opacity=0.1, ax=None):
    """
    Draw a standard biconic surface along the z axis.

    When isSweep is enabled, clearSemiDiameter and ySemi are treated as the
    half-width and half-height of a rectangular swept aperture. Otherwise the
    surface is clipped to a circular aperture with radius clearSemiDiameter.
    """
    if RENDER_MODE:
        return

    ax = CheckAX(ax)

    radiusX = _as_float(radiusX)
    kX = _as_float(kX)
    radiusY = _as_float(radiusY)
    kY = _as_float(kY)
    clearSemiDiameter = _as_float(clearSemiDiameter)
    cumulativeThickness = _as_float(cumulativeThickness)
    ySemi = clearSemiDiameter if ySemi is None else _as_float(ySemi)

    x = np.linspace(-clearSemiDiameter, clearSemiDiameter, numPoints)
    yLimit = ySemi if isSweep else clearSemiDiameter
    y = np.linspace(-yLimit, yLimit, numPoints)
    X, Y = np.meshgrid(x, y)

    cx = 0.0 if np.isinf(radiusX) else 1.0 / radiusX
    cy = 0.0 if np.isinf(radiusY) else 1.0 / radiusY

    radicand = 1.0 - (1.0 + kX) * cx**2 * X**2 - (1.0 + kY) * cy**2 * Y**2
    valid = radicand >= 0.0

    if not isSweep:
        valid &= (X**2 + Y**2) <= clearSemiDiameter**2

    numerator = cx * X**2 + cy * Y**2
    denominator = 1.0 + np.sqrt(np.maximum(radicand, 0.0))
    Z = cumulativeThickness + numerator / denominator
    Z = np.where(valid, Z, np.nan)

    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=opacity)


def DrawAsphericalProfile(radius, k, A, clearSemiDiameter, cumulativeThickness,
                          axis="x", numPoints=THETA_DIV, lineColor="r", lineWidth=1.0, ax=None):
    """
    Draw the sagittal (cross-section) profile of an aspherical surface in the XZ or YZ plane.

    :param radius: Radius of curvature.
    :param k: Conic constant.
    :param A: Even aspheric coefficients (A4, A6, ...).
    :param clearSemiDiameter: Max radial distance from optical axis.
    :param cumulativeThickness: Z-offset.
    :param axis: "x" or "y" to select profile direction.
    :param numPoints: Number of samples.
    :param lineColor: Line color.
    :param lineWidth: Line thickness.
    :param ax: Plot axis (matplotlib 3D).
    """

    if RENDER_MODE:
        return

    ax = CheckAX(ax)

    radius = _as_float(radius)
    k = _as_float(k)
    A = _as_numpy(A)
    clearSemiDiameter = _as_float(clearSemiDiameter)
    cumulativeThickness = _as_float(cumulativeThickness)

    r = np.linspace(-clearSemiDiameter, clearSemiDiameter, numPoints)
    r2 = r**2

    sqrt_term = np.sqrt(1 - (1 + k) * r2 / radius**2)
    base = r2 / (radius * (1 + sqrt_term))
    asphere = np.zeros_like(r2)
    for i, a in enumerate(A):
        # This starts from 2nd order term
        asphere += a * r2 ** (i + 1)

    z = base + asphere + cumulativeThickness

    if axis.lower() == "x":
        x = r
        y = np.zeros_like(x)
    elif axis.lower() == "y":
        y = r
        x = np.zeros_like(y)
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    ax.plot(x, y, z, color=lineColor, linewidth=lineWidth)


def DrawSphericalProfile(radius, clearSemiDiameter, cumulativeThickness,  axis="x", numPoints=THETA_DIV,
                         lineColor="m",  lineWidth=0.5, ax=None):
    """
    Draw the sagittal (cross-section) profile of a spherical surface in the XZ or YZ plane.

    Parameters:
        radius (float): Signed radius of curvature (following your convention).
        clearSemiDiameter (float): Max radial distance from the optical axis.
        cumulativeThickness (float): Z offset of the vertex (same meaning as elsewhere).
        axis (str): "x" or "y" – which meridian to draw along.
        numPoints (int): Number of sample points along the meridian.
        lineColor (str): Plot line color.
        lineWidth (float): Plot line width.
        ax: Matplotlib 3D axis to plot on.
    """
    if RENDER_MODE:
        return

    ax = CheckAX(ax)
    radius = _as_float(radius)
    clearSemiDiameter = _as_float(clearSemiDiameter)
    cumulativeThickness = _as_float(cumulativeThickness)

    # Handle flat surface directly
    if _is_infinite(radius):
        r = np.linspace(-clearSemiDiameter, clearSemiDiameter, numPoints)
        z = np.full_like(r, cumulativeThickness)
        if axis.lower() == "x":
            x, y = r, np.zeros_like(r)
        elif axis.lower() == "y":
            y, x = r, np.zeros_like(r)
        else:
            raise ValueError("axis must be 'x' or 'y'.")

        ax.plot(x, y, z, color=lineColor, linewidth=lineWidth)
        return

    # Finite radius: clamp the plotting range to avoid sqrt domain issues
    Rabs = np.abs(radius)
    # Ensure we don't sample beyond the sphere's valid cap (r <= |R|)
    # Use a tiny margin factor to keep sqrt(...) strictly non-negative.
    r_max = float(clearSemiDiameter)
    if Rabs < r_max:
        r_max = float(Rabs) * 0.999999

    r = np.linspace(-r_max, r_max, numPoints)
    r2 = r**2

    # Spherical sag: z(r) = cumulativeThickness + (R - sign(R)*sqrt(R^2 - r^2))
    # Use maximum(...) for numerical safety at the ends
    inside = np.maximum(radius**2 - r2, 0.0)
    z = cumulativeThickness + (radius - np.sign(radius) * np.sqrt(inside))

    if axis.lower() == "x":
        x = r
        y = np.zeros_like(r)
    elif axis.lower() == "y":
        y = r
        x = np.zeros_like(r)
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    ax.plot(x, y, z, color=lineColor, linewidth=lineWidth)


def DrawDisk(radius, z_height=2, num_points=THETA_DIV ,surfaceColor="b",  ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    radius = _as_float(radius)
    z_height = _as_float(z_height)

    # Parametric equation for the disk
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = np.linspace(0, radius, num_points)
    R, Theta = np.meshgrid(r, theta)

    # Convert to Cartesian coordinates
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = np.full_like(X, z_height)  # Set Z to constant height

    # Plot the disk
    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=0.2)


def DrawDiskInner(maxAperture, minAperture, cumulativeThickness,
                  numPoints=THETA_DIV, surfaceColor="k", opacity=0.1, ax=None):
    """
    Draw an annular disk (a disk with a hole), centered on optical axis (z),
    located at z = cumulativeThickness.

    maxAperture: outer radius
    minAperture: inner radius (hole)
    """
    if (RENDER_MODE):
        return

    ax = CheckAX(ax)
    maxAperture = _as_float(maxAperture)
    minAperture = _as_float(minAperture)
    cumulativeThickness = _as_float(cumulativeThickness)

    # Safety clamps
    if minAperture < 0:
        minAperture = 0.0
    if maxAperture < minAperture:
        maxAperture = minAperture

    theta = np.linspace(0, 2 * np.pi, int(numPoints / 1))
    r = np.linspace(minAperture, maxAperture, int(numPoints / 1))
    Theta, R = np.meshgrid(theta, r)

    x = R * np.cos(Theta)
    y = R * np.sin(Theta)
    z = np.full_like(x, cumulativeThickness)

    ax.plot_surface(x, y, z, color=surfaceColor, alpha=opacity)


def DrawClearBoundary(E1, E2, surfaceColor="k", opacity=0.1, ax=None):
    """
    Draw the clear boundary of a lens element given its 2 ellipses
    """
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)

    # TODO: this method has not been tested with off axis ellipses

    def _SlerpVectors(v1, v2, t):
        """Spherical linear interpolation between two unit vectors."""
        dot = np.dot(v1, v2)
        # Handle the case where v1 and v2 are nearly identical
        if np.abs(dot) > 0.9999:
            return v1  # Skip interpolation and return v1 directly
        omega = np.arccos(dot)
        return (np.sin((1 - t) * omega) / np.sin(omega)) * v1 + (np.sin(t * omega) / np.sin(omega)) * v2

    def _FrustumBetweenEllipses(C1, u1, v1, a1, b1, C2, u2, v2, a2, b2, theta_res=THETA_DIV, t_res=5):
        # Interpolate centers
        t_values = np.linspace(0, 1, t_res)
        centers = (1 - t_values[:, None]) * C1 + t_values[:, None] * C2

        # Interpolate semi-axes
        a_values = (1 - t_values) * a1 + t_values * a2
        b_values = (1 - t_values) * b1 + t_values * b2

        # Interpolate orientations using slerp
        u_interp = np.array([_SlerpVectors(u1, u2, t) for t in t_values])
        v_interp = np.array([_SlerpVectors(v1, v2, t) for t in t_values])

        # Ensure orthonormality
        for i in range(t_res):
            u = u_interp[i]
            v = v_interp[i] - np.dot(v_interp[i], u) * u
            v_interp[i] = v / np.linalg.norm(v)

        # Generate points
        theta = np.linspace(0, 2*np.pi, theta_res)
        points = np.zeros((t_res, theta_res, 3))
        for i in range(t_res):
            # Reshape centers[i] to (3,) for broadcasting
            center = centers[i]
            # Compute the points for the current ellipse
            points[i] = (
                center[None, :] +  # Shape (1, 3)
                a_values[i] * np.outer(np.cos(theta), u_interp[i]) +  # Shape (theta_res, 3)
                b_values[i] * np.outer(np.sin(theta), v_interp[i])  # Shape (theta_res, 3)
            )

        return points.reshape(t_res, theta_res, 3)

    points = _FrustumBetweenEllipses(_as_numpy(E1.center),
                                    _as_numpy(E1.semiAxisDirU),
                                    _as_numpy(E1.semiAxisDirV),
                                    _as_float(E1.semiAxisMagA),
                                    _as_float(E1.semiAxisMagB),
                                    _as_numpy(E2.center),
                                    _as_numpy(E2.semiAxisDirU),
                                    _as_numpy(E2.semiAxisDirV),
                                    _as_float(E2.semiAxisMagA),
                                    _as_float(E2.semiAxisMagB))


    X = points[:, :, 0]
    Y = points[:, :, 1]
    Z = points[:, :, 2]

    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=opacity)


def DrawEllipse(Q, center, num_points=THETA_DIV, lColor="c", lWidth=.35, ax=None):
    """
    Plots a 3D ellipse along the xy-plane at a given z-depth.

    Parameters:
        Q (numpy.ndarray): 2x2 quadratic form matrix defining the ellipse.
        z (float): The z-coordinate where the ellipse is plotted.
        ax (matplotlib Axes3D): The 3D axis to plot on (optional).
        num_points (int): Number of points to approximate the ellipse.
    """
    if(RENDER_MODE):
        return


    ax = CheckAX(ax)
    Q = _as_numpy(Q)
    center = _as_numpy(center)
    eigenvalues, eigenvectors = np.linalg.eig(Q)

    # Compute semi-axes lengths
    axes_lengths = 1 / np.sqrt(eigenvalues)

    # Generate points for a unit circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])

    # Transform the unit circle to match the ellipse
    ellipse = (eigenvectors @ np.diag(axes_lengths) @ unit_circle).T

    # Extract x, y points and set z as constant
    x, y = ellipse[:, 0]+center[Axis.X.value], ellipse[:, 1]+center[Axis.Y.value]
    z_vals = np.full_like(x, center[Axis.Z.value])

    # Plot the ellipse in 3D space
    ax.plot(x, y, z_vals, color=lColor, linewidth=lWidth)


def DrawPlane(points, color = "b", ax=None):

    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    points = _as_numpy(points)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    verts = [list(zip(x, y, z))]
    plane = Poly3DCollection(verts, alpha=0.2, color=color)
    ax.add_collection3d(plane)


def DrawPupil(radius, axialDepth, num_points = 100 ,surfaceColor = "b",  ax=None):
    if(RENDER_MODE):
        return

    ax = CheckAX(ax)
    radius = _as_float(radius)
    axialDepth = _as_float(axialDepth)

    # Parametric equation for the disk
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = radius
    R, Theta = np.meshgrid(r, theta)

    # Convert to Cartesian coordinates
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = np.tile(axialDepth, (len(theta), 1))  # Repeat z values for each angle

    # Plot the disk
    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=0.2, edgecolor=surfaceColor)






def main():
    SetUnifScale()
    AddXYZ(6)

    DrawAspherical(radius=50.0,
                   k=-1.0,
                   A=[1e-5, -2e-7],
                   clearSemiDiameter=25.0,
                   cumulativeThickness=0,
                   surfaceColor="magenta",
                   opacity=0.2)


    plt.show()


if __name__ == "__main__":
    main()
