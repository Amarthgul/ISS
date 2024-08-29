#P = np.array([3, 4, -10])
r = 80
#d = 6 
from locale import normalize
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm


def Normalized(inputVec):
    return inputVec / np.linalg.norm(inputVec)


def line_plane_intersection(plane_normal, point_on_plane, line_direction, point_on_line):
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

def angle_between_vectors(v1, v2):
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
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


def CalTest(posP, d):
    P_xy_projection = np.array([posP[0], posP[1], 0])
    posA = d * ( P_xy_projection / np.linalg.norm(P_xy_projection) )
    posC = d * (-P_xy_projection / np.linalg.norm(P_xy_projection) ) 

    vecPA = posA - posP   
    vecPC = posC - posP  

    vecN = Normalized((Normalized(vecPA) + Normalized(vecPC)) / 2)

    # Calculate the position of point B 
    
    vecPCN = Normalized(vecPC)
    vecPAN = Normalized(vecPA)

    t = (np.dot(vecN, (posA - posC))) / np.dot(vecN, vecPCN)

    posB = posC + vecPCN * t 

    vecAB = Normalized(posB - posA)
    print("Positions P: \t", posP)
    print("Positions AC: \t", posA, posC)
    print("Vectors AC:  \t", vecPA, vecPC)
    print("Vector N: \t", vecN)
    print("Value of t:\t", t)
    print("Pos B:  \t", posB)
    print("Vector AB:  \t", vecAB)
    print("Angles:  \t", angle_between_vectors(vecPCN, vecN), angle_between_vectors(vecPAN, vecN))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([posP[0], posC[0]], [posP[1], posC[1]], [posP[2], posC[2]], label='3D Line', color='b', linewidth=3)
    ax.plot([posP[0], posA[0]], [posP[1], posA[1]], [posP[2], posA[2]], label='3D Line', color='b', linewidth=3)
    ax.plot([posC[0], posA[0]], [posC[1], posA[1]], [posC[2], posA[2]], label='3D Line', color='b', linewidth=3)
    
    #ax.plot([posP[0], posB[0]], [posP[1], posB[1]], [posP[2], posB[2]], label='3D Line', color='g')
    ax.plot([posA[0], posB[0]], [posA[1], posB[1]], [posA[2], posB[2]], label='3D Line', color='g')
    
    ax.plot([posP[0], posP[0]+vecN[0]],  [posP[1], posP[1]+vecN[1]],  [posP[2], posP[2]+vecN[2]], label='3D Line', color='r')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, -10)

    plt.show()

CalTest(np.array([3, 4, -10]), 6)