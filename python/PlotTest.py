
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
    ax.set_xlim(offsetScalar, -offsetScalar)
    ax.set_ylim(offsetScalar, -offsetScalar)
    ax.set_zlim(lim, -lim)

def DrawPoint(ax, point):
    ax.scatter3D(point[0], point[1], point[2])

def Draw3D(ax, x, y, z):
    ax.plot(x, y, z)

def DrawLine(ax, point1, point2, lineColor = "k", lineWidth = 2):
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 
            label = '3D Line', color = lineColor, linewidth = lineWidth)
    
def DrawCircle(ax, radius, offset = 0, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta) + offset])
    ax.plot(circle_points[0], circle_points[1], circle_points[2])
    

def DrawIncidentPlane(ax, posP, posB, d):
    P_xy_projection = np.array([posP[0], posP[1], 0])
    # Use the projection to derive the plane intersection
    posA = d * ( P_xy_projection / np.linalg.norm(P_xy_projection) )
    posC = d * (-P_xy_projection / np.linalg.norm(P_xy_projection) )
    
    DrawLine(ax, origin, posA, lineWidth = 1)
    DrawLine(ax, origin, posC, lineWidth = 1)
    DrawLine(ax,   posP, posA, lineWidth = 1)
    DrawLine(ax,   posP, posC, lineWidth = 1)
    DrawLine(ax,   posA, posB, lineWidth = 1)
    
def DrawEmission(points):
    for p in points:
        print(p)