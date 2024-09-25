
"""
This module is used to provide visual validation for the process. 
"""

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm


origin = np.array([0, 0, 0])

# Matplotlib z axis is always shortened 
zAxisCompensationFactor = 1.25

def Setup3Dplot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax 
    
def AddXYZ(ax, unitLength = 1, lineWidth = 2):
    ax.plot([0, unitLength], [0, 0], [0, 0], label = '3D Line', color = 'r', linewidth = lineWidth)
    ax.plot([0, 0], [0, unitLength], [0, 0], label = '3D Line', color = 'g', linewidth = lineWidth)
    ax.plot([0, 0], [0, 0], [0, unitLength], label = '3D Line', color = 'b', linewidth = lineWidth)
    
def SetUnifScale(ax, lim = 6):
    offsetScalar = zAxisCompensationFactor * lim
    ax.set_xlim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_ylim(offsetScalar/2.0, -offsetScalar/2.0)
    ax.set_zlim(lim, 0)

def DrawPoint(ax, point):
    ax.scatter3D(point[0], point[1], point[2])

def DrawPoints(ax, points):
    for p in points:
        ax.scatter3D(p[0], p[1], p[2])

def Draw3D(ax, x, y, z):
    ax.plot(x, y, z)

def DrawLine(ax, point1, point2, lineColor = "k", lineWidth = 2):
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 
            label = '3D Line', color = lineColor, linewidth = lineWidth)
    
def DrawCircle(ax, radius, offset = 0, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta) + offset])
    ax.plot(circle_points[0], circle_points[1], circle_points[2])
    

def DrawIncidentPlane(ax, posA, posB, posC, posP, d):
    originOffset = np.array([origin[0], origin[1], posA[2]])
    DrawLine(ax, originOffset, posA, lineWidth = 1)
    DrawLine(ax, originOffset, posC, lineWidth = 1)
    DrawLine(ax,    posP,   posA, lineWidth = 1)
    DrawLine(ax,    posP,   posC, lineWidth = 1)
    DrawLine(ax,    posA,   posB, lineWidth = 1)
    
def DrawEmission(points):
    for p in points:
        print(p)
        

def DrawSpherical(ax, radius, clearSemiDiameter, thickness, numPoints = 20, surfaceColor = "k"):
    """
    Draw a spherical surface along the z axis. 
    
    :param ax: axis to draw on. 
    :param radius: radius of the spherical surface. 
    :param clearSemiDiameter: clear semi diameter. Surface will be trimed around it. 
    :param thickness: Cumulative thickness from 1st surface. 
    :param numPoints: number of points, controls the subdivision of the surface. 
    :surfaceColor: color of the surface. 
    """
    unsignedrRadius = radius * np.sign(radius)

    radianLimit = np.arcsin(clearSemiDiameter/unsignedrRadius) 
    
    theta = np.linspace(0, 2 * np.pi, int(numPoints / 1))  # Azimuthal angle (0 <= theta <= 2*pi)
    phi = np.linspace(0, radianLimit, int(numPoints / 1))  # Polar angle (0 <= phi <= phi_max for a bowl)
    
    theta, phi = np.meshgrid(theta, phi)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = -np.sign(radius) * (unsignedrRadius * np.cos(phi) - unsignedrRadius) + thickness
        
    ax.plot_surface(x, y, z, color = surfaceColor, alpha = 0.25)



def main():
    ax = Setup3Dplot()
    SetUnifScale(ax)
    AddXYZ(ax, 6)
    
    DrawSpherical(ax, -5, 4, 0)
    
    DrawSpherical(ax, 10, 4, 0.2)
    
    DrawSpherical(ax, 10, 4, 1)
    DrawSpherical(ax, -20, 4, 4)
    
    plt.show()
    

if __name__ == "__main__":
    main() 




