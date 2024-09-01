

from locale import normalize
from mailbox import Babyl
from xml.dom.expatbuilder import theDOMImplementation
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

import PlotTest

origin = np.array([0, 0, 0])

def Normalized(inputVec):
    return inputVec / np.linalg.norm(inputVec)

def Minus90(inputRadian):
    return np.pi / 2 - inputRadian

def ORotation(theta, axis, inputVertex):
    """
    Rotate the input vertex by theta radians along y-axis.
    :param theta: Angle in radians.
    :param axis: Axis of rotation.
    :param inputVertex: Input vertex.
    :return: Rotated vertex
    """
    R = np.array([
        [np.cos(theta) + axis[0]**2 * (1 - np.cos(theta)), 
         axis[0] * axis[1] * (1 - np.cos(theta)) - axis[2] * np.sin(theta), 
         axis[0] * axis[2] * (1 - np.cos(theta)) + axis[1]],
        [axis[0] * axis[1] * (1 - np.cos(theta)) + axis[2] * np.sin(theta), 
         np.cos(theta) + axis[1]**2 * (1 - np.cos(theta)),
         axis[1] * axis[2] * (1 - np.cos(theta)) -axis[1]],
        [axis[2] * axis[0] * (1 - np.cos(theta)) - axis[1] * np.sin(theta), 
         axis[2] * axis[1] * (1 - np.cos(theta)) + axis[0] * np.sin(theta), 
         np.cos(theta) + axis[2]**2 * (1 - np.cos(theta))]
    ])
    #pt = np.matmul(R, inputVertex)  
 
    return np.matmul(R, inputVertex) 

def Rotation(theta, pivot, axis, inputVertex):
    #offsetVertex = inputVertex - pivot 
    offsetVertex = np.transpose(np.transpose(inputVertex) - pivot )
    #print(offsetVertex)
    offsetVertex = ORotation(theta, axis, offsetVertex)
    return np.transpose(np.transpose(inputVertex) + pivot )
    

def Translate(inputVertex, translation):
    return np.transpose(np.transpose(inputVertex) + translation)


#  ===========================================================================
"""
==============================================================================
"""

def linePlaneIntersection(plane_normal, point_on_plane, line_direction, point_on_line):
    """
    Calculates the intersection point between a plane and a line in 3D space.
    
    :param plane_normal: A 3D vector normal to the plane (numpy array).
    :param point_on_plane: A point on the plane (numpy array).
    :param line_direction: The direction vector of the line (numpy array).
    :param point_on_line: A point on the line (numpy array).
    :return: The intersection point (numpy array), or None if the line is parallel to the plane.
    """
    # Calculate the dot product of the plane normal and the line direction
    dot_product = np.dot(plane_normal, line_direction)
    
    # If the dot product is 0, the line is parallel to the plane (no intersection or line lies on the plane)
    if np.isclose(dot_product, 0):
        print("The line is parallel to the plane and does not intersect.")
        return None
    
    # Calculate the parameter t for the line equation
    t = np.dot(plane_normal, (point_on_plane - point_on_line)) / dot_product
    
    # Calculate the intersection point using the parametric line equation
    intersection_point = point_on_line + t * line_direction
    
    return intersection_point

def angleBetweenVectors(v1, v2, useDegrees = False):
    """
    Calculates the angle between two vectors in radians.
    
    :param v1: First vector (numpy array).
    :param v2: Second vector (numpy array).
    :return: The angle between the vectors in degrees.
    """
    # Normalize vectors to avoid floating-point issues
    v1_normalized = np.linalg.norm(v1)
    v2_normalized = np.linalg.norm(v2)
    
    # Calculate the dot product of the two vectors
    dot_product = np.dot(v1, v2)
    
    # Calculate the magnitudes (norms) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure cos_theta is in the valid range [-1, 1] to avoid errors due to floating-point precision
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Convert to degrees
    if useDegrees:
        np.degrees(angle_radians)
    else:
        return angle_radians

def CircularDistribution(radius = 1, layer = 5, densityScale = 0.02, powerCoef = 0.8):
    """
    Accquire a distribution based on polar coordinate. 
    :param radius: radius of the circle, it is suggested to keep it at 1. 
    :param layer: number of layers of samples, each layer is in a concentric circle, with different radius.
    :param densityScale: with increase of radius, the delta area is used to calculate the points needed, this parameter is used to divide the delta area and get the number of sample points. The lower it is, the more sample. 
    :param powerCoef: due to the use of delta area depending on radius, linear scale will make outer edges having more samples. This parameter reduces this unevenness. 
    """
    partitionLayer = (np.arange(layer) + 1.0) / layer
    lastArea = 0
    
    points = np.array([[0], [0], [0]])
    
    for current in partitionLayer:
        area = np.pi * current ** 2
        deltaArea = area - lastArea
        lastArea = area 
        
        pointsInLayer = int((deltaArea / densityScale) ** powerCoef)
        layerH = current 
        print(current, "    ", layerH)
        layerTheta = np.arange(pointsInLayer) * ((np.pi * 2) / pointsInLayer) 

        layerPoints = np.array([layerH * np.cos(layerTheta), 
                                    layerH * np.sin(layerTheta), 
                                    np.zeros(pointsInLayer)])
        points = np.hstack((points, layerPoints))
        
    return points * radius

    
#  ===========================================================================
"""
==============================================================================
"""

def FindB(posA, posC, posP, d):
    """
    Find the position of point B. 
    """
    vecPA = posA - posP   
    vecPC = posC - posP  

    vecN = Normalized((Normalized(vecPA) + Normalized(vecPC)) / 2)
    
    vecPCN = Normalized(vecPC)
    vecPAN = Normalized(vecPA)

    t = (np.dot(vecN, (posA - posC))) / np.dot(vecN, vecPCN)

    posB = posC + vecPCN * t 

    return posB 

def EllipsePeripheral(posA, posB, posC, posP, d, useDistribution = True):
    """
    Find the points on the ellipse and align it to the AB plane. 
    """
    
    # Util vectors 
    P_xy_projection = Normalized(np.array([posP[0], posP[1], 0]))
    vecCA = posA - posC
    
    # Lengths to calculate semi-major axis length 
    BB = (2 * d) * ((posP[2] - posB[2]) / posP[2])
    AC = np.linalg.norm(posA - posC)
    
    # Semi-major axis length 
    a = np.linalg.norm(posA - posB) / 2
    b = np.sqrt(BB * AC) / 2
    
    # Calculate the ellipse 
    if (useDistribution):
        points = CircularDistribution()
        points = np.transpose(np.transpose(points) * np.array([b, a, 0]))
        #print(points)
    else:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = b * np.cos(theta) 
        y = a * np.sin(theta) 
        z = np.zeros_like(theta)
        points = np.array([x, y, z])


    # Rotate the ellipse to it faces the right direction in the world xy plane,
    # i.e., one of its axis coincides with the tangential plane 
    theta_1 = angleBetweenVectors(posA, np.array([0, 1, 0]))
    trans_1 = ORotation(-theta_1, np.array([0, 0, 1]), points)
    
    # Move the points to be in tangent with A 
    trans_1 = Translate(trans_1, P_xy_projection * (d - a)) 
    
    # Rotate the ellipse around A it fits into the AB plane 
    theta = angleBetweenVectors(posB-posA, posC-posA)
    axis = Normalized(np.array([-vecCA[1], vecCA[0], 0]))
    trans_2 = Translate(trans_1, -posA)
    trans_2 = ORotation(-theta, axis, trans_2)
    trans_2 = Translate(trans_2, posA)
    
    return trans_2

def main():
    posP = np.array([3, 4, -10])
    d = 6
    P_xy_projection = np.array([posP[0], posP[1], 0])
    posA = d * ( P_xy_projection / np.linalg.norm(P_xy_projection) )
    posC = d * (-P_xy_projection / np.linalg.norm(P_xy_projection) )
    vecCA = posA - posC
    
    posB = FindB(posA, posC, posP, d)
    
    points = EllipsePeripheral(posA, posB, posC, posP, d) 

    pointsCuircle = EllipsePeripheral(posA, posB, posC, posP, d, False) 
    
    ax = PlotTest.Setup3Dplot()
    PlotTest.SetUnifScale(ax)
    PlotTest.AddXYZ(ax, 6)
    PlotTest.DrawCircle(ax, 6)
    PlotTest.DrawIncidentPlane(ax, posP, posB, d)

    PlotTest.Draw3D(ax, pointsCuircle[0], pointsCuircle[1], pointsCuircle[2])
    PlotTest.DrawPoint(ax, points)

    
    plt.show()


#Rotation(0.61, np.array([0, 1, 0]), np.array([1, 1, 0]))
main()