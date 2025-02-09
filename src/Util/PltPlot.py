
"""
This module is used to provide preliminary visuals for the project, it is not the best for accuracy nor effect. 
In the future the project should consider switching to mayavi or better libaraies for scientific visulization. 
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.linalg import norm


from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import ORIGIN, INFINITY, DEVELOPER_MODE
from Util.ColorWavelength import ColorTuplePLT


# ==================================================================
""" ============================================================ """
# ==================================================================


# Matplotlib z axis is always shortened 
zAxisCompensationFactor = 1.25

fig = None 
AX = None 

if(DEVELOPER_MODE):
    fig = plt.figure()

def Setup3Dplot():
    global fig 
    ax = fig.add_subplot(111, projection='3d')
    return ax 
    
if(DEVELOPER_MODE):
    AX = Setup3Dplot()



def Reset2D():
    global fig 
    fig = plt.figure()


def CheckAX():
    """
    This function checks if the axis is initialized.
    Such that the user does not have to call Setup3Dplot() manually and could add plot through the entire program. 
    """
    global AX
    if (AX == None):
        AX = Setup3Dplot()


# ==================================================================
""" ============================================================ """
# ==================================================================

def RemoveBG(ax=AX):
    fig.patch.set_alpha(0)
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


def AddXYZ(unitLength = 10, lineWidth = 1, ax=AX):
    CheckAX()
    ax.plot([0, unitLength], [0, 0], [0, 0], label = '3D Line', color = 'r', linewidth = lineWidth)
    ax.plot([0, 0], [0, unitLength], [0, 0], label = '3D Line', color = 'g', linewidth = lineWidth)
    ax.plot([0, 0], [0, 0], [0, unitLength], label = '3D Line', color = 'b', linewidth = lineWidth)
    

def SetUnifScale(lim = 10, ax=AX):
    CheckAX()
    offsetScalar = zAxisCompensationFactor * lim
    ax.set_xlim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_ylim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_zlim(lim, 0)


def DrawPoint(point, color='red', ax=AX):
    CheckAX()
    if(backend_name == "cupy"):
        point = bd.asnumpy(point)

    ax.scatter3D(point[0], point[1], point[2], color=color)


def DrawPoints(points, color='red', ax=AX):
    CheckAX()

    if(backend_name == "cupy"):
        points = bd.asnumpy(points)

    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter3D(x, y, z, color=color)


def DrawPointsPerColor(points, color, ax=AX):
    """
    This is basically the same as DrawPoints(), but the color parameter is treated as an array of same dimension as points. 
    """

    CheckAX()

    if(backend_name == "cupy"):
        points = bd.asnumpy(points)
        color = ColorTuplePLT(color)
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter3D(x, y, z, color=color, s=0.5)


def Draw3D(x, y, z, ax=AX):
    CheckAX()
    ax.plot(x, y, z)


def DrawLine(point1, point2, lineColor = "k", lineWidth = 2, zorder=10, ax=AX):
    CheckAX()
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 
            label = '3D Line', color = lineColor, linewidth = lineWidth, zorder=zorder)


def DrawLines(pointSetOne, pointSetTwo, lineColor = "k", lineWidth = 0.5, zorder=10, ax=AX):
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    CheckAX() 
    start = pointSetOne  # [x_start, y_start, z_start]
    end = pointSetTwo    # [x_end, y_end, z_end]

    if(backend_name == "cupy"):
        start = bd.asnumpy(start)
        end = bd.asnumpy(end)

    # Create segments for Line3DCollection
    segments = bd.array([[s, e] for s, e in zip(start, end)])
    if(backend_name == "cupy"):
        segments = bd.asnumpy(segments)

    line_collection = Line3DCollection(segments, linewidths=lineWidth, color=lineColor)
    ax.add_collection3d(line_collection)


def DrawCircle(radius, offset = 0, num_points=100, ax=AX):
    CheckAX()
    theta = bd.linspace(0, 2 * bd.pi, num_points)
    circle_points = bd.array([radius * bd.cos(theta), radius * bd.sin(theta), bd.zeros_like(theta) + offset])
    ax.plot(circle_points[0], circle_points[1], circle_points[2])
    

def DrawIncidentPlane(posA, posB, posC, posP, d, ax=AX):
    CheckAX()
    originOffset = bd.array([ORIGIN[0], ORIGIN[1], posA[2]])
    DrawLine(ax, originOffset, posA, lineWidth = 1)
    DrawLine(ax, originOffset, posC, lineWidth = 1)
    DrawLine(ax,    posP,   posA, lineWidth = 1)
    DrawLine(ax,    posP,   posC, lineWidth = 1)
    DrawLine(ax,    posA,   posB, lineWidth = 1)
    

def DrawRaybatch(rayBatch, lineColor='blue', length = 10, ax=AX):
    CheckAX()
    if(backend_name == "cupy"):
        data = bd.asnumpy(rayBatch.value)
    else:
        data = rayBatch.value 

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    u, v, w = data[:, 3], data[:, 4], data[:, 5]

    q = ax.quiver(x, y, z, u, v, w,
              length=length,                # Increase arrow length
              normalize=False,           # Maintain relative vector sizes
              arrow_length_ratio=0.1,    # Smaller arrowhead
              pivot='tail',              # Arrows start at [x,y,z]
              linewidths=0.5,            # Thicker arrows
              color=lineColor) 


def DrawNormal(intersections, normals, lineColor='green', lineLength=1, lineWidths = 0.5, arrowRatio=0.1, ax=AX):
    CheckAX()
    if(backend_name == "cupy"):
        intersections = bd.asnumpy(intersections)
        normals = bd.asnumpy(normals)

    x, y, z = intersections[:, 0], intersections[:, 1], intersections[:, 2]
    u, v, w = normals[:, 0], normals[:, 1], normals[:, 2]

    q = ax.quiver(x, y, z, u, v, w,
              length=lineLength,                # Increase arrow length
              normalize=False,                  # Maintain relative vector sizes
              arrow_length_ratio=arrowRatio,    # Smaller arrowhead
              pivot='tail',                     # Arrows start at [x,y,z]
              linewidths=lineWidths,            
              color=lineColor) 


def DrawDirection(position, direction, lineColor='green', lineLength=5, arrowRatio=0.1, ax=AX):
    CheckAX()
    if(backend_name == "cupy"):
        position = bd.asnumpy(position)
        direction = bd.asnumpy(direction)

    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    u, v, w = direction[:, 0], direction[:, 1], direction[:, 2]

    q = ax.quiver(x, y, z, u, v, w,
              length=lineLength,                # Increase arrow length
              normalize=False,                  # Maintain relative vector sizes
              arrow_length_ratio=arrowRatio,    # Smaller arrowhead
              pivot='tail',                     # Arrows start at [x,y,z]
              linewidths=0.5,                   # Thicker arrows
              color=lineColor) 

        
def DrawSpherical(radius, clearSemiDiameter, cumulativeThickness, numPoints = 20, surfaceColor = "k", ax=AX):
    """
    Draw a spherical surface along the z axis. 
    
    :param ax: axis to draw on. 
    :param radius: radius of the spherical surface. 
    :param clearSemiDiameter: clear semi diameter. Surface will be trimed around it. 
    :param cumulativeThickness: Cumulative thickness from 1st surface. 
    :param numPoints: number of points, controls the subdivision of the surface. 
    :surfaceColor: color of the surface. 
    """
    if(radius == INFINITY):
        DrawDisk(clearSemiDiameter, cumulativeThickness, numPoints, surfaceColor)

    CheckAX()
    unsignedrRadius = radius * bd.sign(radius)

    radianLimit = bd.arcsin(clearSemiDiameter/unsignedrRadius) 
    
    theta = bd.linspace(0, 2 * bd.pi, int(numPoints / 1))  # Azimuthal angle (0 <= theta <= 2*pi)
    phi = bd.linspace(0, radianLimit, int(numPoints / 1))  # Polar angle (0 <= phi <= phi_max for a bowl)
    
    theta, phi = bd.meshgrid(theta, phi)
    
    x = radius * bd.sin(phi) * bd.cos(theta)
    y = radius * bd.sin(phi) * bd.sin(theta)
    z = - bd.sign(radius) * (unsignedrRadius * bd.cos(phi) - unsignedrRadius) + cumulativeThickness
        
    if(backend_name == "cupy"):
        x = bd.asnumpy(x)
        y = bd.asnumpy(y)
        z = bd.asnumpy(z)

    ax.plot_surface(x, y, z, color = surfaceColor, alpha = 0.1)


def DrawDisk(radius, z_height = 2, num_points = 100 ,surfaceColor = "b",  ax=AX):
    CheckAX()

    # Parametric equation for the disk
    theta = bd.linspace(0, 2 * bd.pi, num_points)
    r = bd.linspace(0, radius, num_points)
    R, Theta = bd.meshgrid(r, theta)

    # Convert to Cartesian coordinates
    X = R * bd.cos(Theta)
    Y = R * bd.sin(Theta)
    Z = bd.full_like(X, z_height)  # Set Z to constant height

    if(backend_name == "cupy"):
        X = bd.asnumpy(X)
        Y = bd.asnumpy(Y)
        Z = bd.asnumpy(Z)

    # Plot the disk
    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=0.2)


def DrawPlane(points, color = "b", ax=AX):
    CheckAX()
    if(backend_name == "cupy"):
        points = bd.asnumpy(points)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    verts = [list(zip(x, y, z))]
    plane = Poly3DCollection(verts, alpha=0.2, color=color)
    ax.add_collection3d(plane)


def DrawPupil(radius, axialDepth, num_points = 100 ,surfaceColor = "b",  ax=AX):
    CheckAX()

    # Parametric equation for the disk
    theta = bd.linspace(0, 2 * bd.pi, num_points)
    r = radius
    R, Theta = bd.meshgrid(r, theta)

    # Convert to Cartesian coordinates
    X = R * bd.cos(Theta)
    Y = R * bd.sin(Theta)
    Z = bd.tile(axialDepth, (len(theta), 1))  # Repeat z values for each angle

    if(backend_name == "cupy"):
        X = bd.asnumpy(X)
        Y = bd.asnumpy(Y)
        Z = bd.asnumpy(Z)

    # Plot the disk
    ax.plot_surface(X, Y, Z, color=surfaceColor, alpha=0.2, edgecolor=surfaceColor)




def main():
    SetUnifScale()
    AddXYZ(6)
    
    DrawSpherical(INFINITY, 4, 0)
    
    
    plt.show()
    

if __name__ == "__main__":
    main() 




