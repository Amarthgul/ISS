
"""
This module is used to provide visual validation for the process. 
"""

from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import ORIGIN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

# ==================================================================
""" ============================================================ """
# ==================================================================

# Matplotlib z axis is always shortened 
zAxisCompensationFactor = 1.25

AX = None

def Setup3Dplot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax 
    
def CheckAX():
    """
    This function checks if the axis is initialized.
    Such that the user does not have to call Setup3Dplot() manually and could add plot through the entire program. 
    """
    global AX
    if (AX == None):
        AX = Setup3Dplot()

AX = Setup3Dplot()


# ==================================================================
""" ============================================================ """
# ==================================================================



def AddXYZ(unitLength = 1, lineWidth = 2, ax=AX):
    CheckAX()
    ax.plot([0, unitLength], [0, 0], [0, 0], label = '3D Line', color = 'r', linewidth = lineWidth)
    ax.plot([0, 0], [0, unitLength], [0, 0], label = '3D Line', color = 'g', linewidth = lineWidth)
    ax.plot([0, 0], [0, 0], [0, unitLength], label = '3D Line', color = 'b', linewidth = lineWidth)
    

def SetUnifScale(lim = 6, ax=AX):
    CheckAX()
    offsetScalar = zAxisCompensationFactor * lim
    ax.set_xlim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_ylim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_zlim(lim, 0)


def DrawPoint(point, ax=AX):
    CheckAX()
    ax.scatter3D(point[0], point[1], point[2])


def DrawPoints(points, ax=AX):
    CheckAX()

    if(backend_name == "cupy"):
        data = bd.asnumpy(points)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    ax.scatter3D(x, y, z)


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
    x, y, z = pointSetOne[:, 0], pointSetOne[:, 1], pointSetOne[:, 2]
    u, v, w = pointSetTwo[:, 0], pointSetTwo[:, 1], pointSetTwo[:, 2]
    start = pointSetOne  # [x_start, y_start, z_start]
    end = pointSetTwo    # [x_end, y_end, z_end]

    if(backend_name == "cupy"):
        start = bd.asnumpy(start)
        end = bd.asnumpy(end)

    # Create segments for Line3DCollection
    segments = np.array([[s, e] for s, e in zip(start, end)])
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
    

def DrawRaybatch(rayBatch, color='blue', length = 10, ax=AX):
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
              arrow_length_ratio=0,    # Smaller arrowhead
              pivot='tail',              # Arrows start at [x,y,z]
              linewidths=0.5,            # Thicker arrows
              color=color) 


def DrawNormal(intersections, normals, color='green', ax=AX):
    CheckAX()
    if(backend_name == "cupy"):
        intersections = bd.asnumpy(intersections)
        normals = bd.asnumpy(normals)

    x, y, z = intersections[:, 0], intersections[:, 1], intersections[:, 2]
    u, v, w = normals[:, 0], normals[:, 1], normals[:, 2]

    q = ax.quiver(x, y, z, u, v, w,
              length=1,                # Increase arrow length
              normalize=False,           # Maintain relative vector sizes
              arrow_length_ratio=0.1,    # Smaller arrowhead
              pivot='tail',              # Arrows start at [x,y,z]
              linewidths=0.5,            # Thicker arrows
              color=color) 

        
def DrawSpherical(radius, clearSemiDiameter, thickness, numPoints = 20, surfaceColor = "k", ax=AX):
    """
    Draw a spherical surface along the z axis. 
    
    :param ax: axis to draw on. 
    :param radius: radius of the spherical surface. 
    :param clearSemiDiameter: clear semi diameter. Surface will be trimed around it. 
    :param thickness: Cumulative thickness from 1st surface. 
    :param numPoints: number of points, controls the subdivision of the surface. 
    :surfaceColor: color of the surface. 
    """
    CheckAX()
    unsignedrRadius = radius * bd.sign(radius)

    radianLimit = bd.arcsin(clearSemiDiameter/unsignedrRadius) 
    
    theta = bd.linspace(0, 2 * bd.pi, int(numPoints / 1))  # Azimuthal angle (0 <= theta <= 2*pi)
    phi = bd.linspace(0, radianLimit, int(numPoints / 1))  # Polar angle (0 <= phi <= phi_max for a bowl)
    
    theta, phi = bd.meshgrid(theta, phi)
    
    x = radius * bd.sin(phi) * bd.cos(theta)
    y = radius * bd.sin(phi) * bd.sin(theta)
    z = - bd.sign(radius) * (unsignedrRadius * bd.cos(phi) - unsignedrRadius) + thickness
        
    if(backend_name == "cupy"):
        x = bd.asnumpy(x)
        y = bd.asnumpy(y)
        z = bd.asnumpy(z)
    ax.plot_surface(x, y, z, color = surfaceColor, alpha = 0.25)



def main():
    SetUnifScale()
    AddXYZ(6)
    
    DrawSpherical(-5, 4, 0)
    
    DrawSpherical(10, 4, 0.2)
    
    DrawSpherical(10, 4, 1)
    DrawSpherical(-20, 4, 4)
    
    plt.show()
    

if __name__ == "__main__":
    main() 




