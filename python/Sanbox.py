

from locale import normalize
from mailbox import Babyl
from turtle import clear
from xml.dom.expatbuilder import theDOMImplementation
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

import PlotTest

origin = np.array([0, 0, 0])

#  ===========================================================================
"""
================================= Utility ====================================
"""

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

def angleBetweenVectors(v1, v2, use_degrees = False):
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
    if use_degrees:
        np.degrees(angle_radians)
    else:
        return angle_radians

def CircularDistribution(radius = 1, layer = 5, densityScale = 0.02, powerCoef = 0.8, shrink = 0.925):
    """
    Accquire a distribution based on polar coordinate. 

    :param radius: radius of the circle, it is suggested to keep it at 1. 
    :param layer: number of layers of samples, each layer is in a concentric circle, with different radius.
    :param densityScale: with increase of radius, the delta area is used to calculate the points needed, this parameter is used to divide the delta area and get the number of sample points. The lower it is, the more sample. 
    :param powerCoef: due to the use of delta area depending on radius, linear scale will make outer edges having more samples. This parameter reduces this unevenness. 
    :param shrink: shrink the distribution a bit to avoid edge clipping when used later in the projection. 
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
        layerTheta = np.arange(pointsInLayer) * ((np.pi * 2) / pointsInLayer) 

        layerPoints = np.array([layerH * np.cos(layerTheta), 
                                    layerH * np.sin(layerTheta), 
                                    np.zeros(pointsInLayer)])
        points = np.hstack((points, layerPoints))
        
    return points * radius * shrink

def SphericalNormal(sphere_radius, intersections, front_vertex):
    """
    Calculate the normal direction at the intersections. 

    :param sphere_radius: radius of the sperical surface. 
    :param intersections: points of intersections between incident rays and the surface. 
    :param front_vertex: vertex of the surface facing object side. 
    """

    # Offset from the front vertex to find the spherical origin 
    origin = front_vertex + np.array([0, 0, sphere_radius])

    return Normalized(intersections - origin)

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

def EllipsePeripheral(posA, posB, posC, posP, d, r, useDistribution = True):
    """
    Find the points on the ellipse and align it to the AB plane. 

    :param posA: position of point A. 
    :param posB: position of point B. 
    :param posC: position of point C. 
    :param posP: position of point P. 
    :param d: clear semi diameter of the surface. 
    :param useDistribution: when enabled, the method returns a distribution of points in the ellipse area instead of the points representing the outline of the ellipse. 
    """
    
    # Util vectors 
    P_xy_projection = Normalized(np.array([posP[0], posP[1], 0]))

    # On axis scenario 
    if (np.linalg.norm(P_xy_projection) == 0):
        P_xy_projection = np.array([d, 0, 0])

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
    else:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = b * np.cos(theta) 
        y = a * np.sin(theta) 
        z = np.zeros_like(theta)
        points = np.array([x, y, z])

    # On axis rays does not need to do the rotation 
    if ([posP[0] == 0 and posP[1] == 0]):
        offset = r - np.sqrt(r**2 - d**2)
        z = np.ones(len(points[0])) * offset
        return np.array([points[0], points[1], z]) 

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

def RaySphereIntersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    """
    Calculate the intersection points between a ray and a sphere.

    :param ray_origin: The origin of the ray (numpy array or list of 3 coordinates).
    :param ray_direction: The direction of the ray (numpy array or list of 3 coordinates).
    :param sphere_center: The center of the sphere (numpy array or list of 3 coordinates).
    :param sphere_radius: The radius of the sphere (float).
    :return: A tuple containing the intersection points or None if there are no intersections.
    """
    
    # Convert input to numpy arrays
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    sphere_center = np.array(sphere_center)

    # Normalize the direction vector
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Coefficients for the quadratic equation
    oc = ray_origin - sphere_center
    print("oc: ", oc)
    A = np.dot(ray_direction, ray_direction)
    #A = np.sum(ray_direction ** 2, axis=1)
    B = 2.0 * np.dot(oc, ray_direction)
    C = np.dot(oc, oc) - sphere_radius**2

    # Discriminant of the quadratic equation
    discriminant = B**2 - 4*A*C

    if discriminant < 0:
        return None  # No intersection
    elif discriminant == 0:
        # One point of intersection (tangent)
        # For this application this should never happen 
        t = -B / (2*A)
        intersection_point = ray_origin + t * ray_direction
        return intersection_point
    else:
        # Two points of intersection
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2*A)
        t2 = (-B + sqrt_discriminant) / (2*A)
        
        # Calculate the intersection points
        intersection_point1 = ray_origin + t1 * ray_direction
        intersection_point2 = ray_origin + t2 * ray_direction
        
        return intersection_point1, intersection_point2

def raySphereIntersectionArray(ray_origins, ray_directions, sphere_center, sphere_radius):
    """
    Calculate the intersection points between multiple rays and a sphere.

    :param ray_origins: An array of ray origins with shape (N, 3), where N is the number of rays.
    :param ray_directions: An array of ray directions with shape (N, 3).
    :param sphere_center: The center of the sphere (numpy array or list of 3 coordinates).
    :param sphere_radius: The radius of the sphere (float).
    :return: A list of intersection points for each ray. Each entry is either None (if no intersection),
             one point (if the ray is tangent), or two points (if the ray intersects the sphere at two points).
    """
    # Convert inputs to numpy arrays
    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)
    sphere_center = np.array(sphere_center)

    # Normalize the direction vectors
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]

    # Coefficients for the quadratic equation
    oc = ray_origins - sphere_center
    A = np.sum(ray_directions**2, axis=1)  # A = dot(ray_direction, ray_direction)
    B = 2.0 * np.sum(oc * ray_directions, axis=1)
    C = np.sum(oc**2, axis=1) - sphere_radius**2

    # Discriminant of the quadratic equation
    discriminant = B**2 - 4*A*C

    # Prepare the list to store results
    intersections = []

    # Loop over each ray and compute the intersection
    for i in range(len(ray_origins)):
        if discriminant[i] < 0:
            intersections.append(None)  # No intersection
        elif discriminant[i] == 0:
            # One point of intersection (tangent)
            t = -B[i] / (2*A[i])
            intersection_point = ray_origins[i] + t * ray_directions[i]
            intersections.append(intersection_point)
        else:
            # Two points of intersection
            sqrt_discriminant = np.sqrt(discriminant[i])
            t1 = (-B[i] - sqrt_discriminant) / (2*A[i])
            t2 = (-B[i] + sqrt_discriminant) / (2*A[i])
            
            # Calculate the intersection points
            intersection_point1 = ray_origins[i] + t1 * ray_directions[i]
            intersection_point2 = ray_origins[i] + t2 * ray_directions[i]
            
            intersections.append((intersection_point1, intersection_point2))

    return intersections

def IsolateIntersection(intersections, sphere_radius, clear_semi_diameter, cumulative_thickness):
    """
    Find the correct intersection points given 2 intersections between a sphere and a vector. 

    :param intersections: 2 intersection points. 
    :param sphere_radius: radius of the spherical surface. 
    :param clear_semi_diameter: ¯|_(ツ)_/¯
    :param cumulative_thickness: cumulative thickness from the first vertex. 
    """

    if (len(intersections) == 1): raise ValueError("No intersection is possible at all")
    if (len(intersections) == 1): return intersections 

    allowed_left = cumulative_thickness
    
    difference = (abs(sphere_radius) - np.sqrt(sphere_radius**2 - clear_semi_diameter**2))
    if (sphere_radius > 0):
        allowed_right = cumulative_thickness + difference
    else:
        allowed_right = cumulative_thickness - difference
        allowed_right, allowed_left = allowed_left, allowed_right 
        
    if (intersections[0][2] >= allowed_left and intersections[0][2] <= allowed_right):
        return intersections[0]
    elif (intersections[1][2] >= allowed_left and intersections[1][2] <= allowed_right):
        return intersections[1] 
    else:
        # This is either a value error or it's the ray being vignetted. 
        # For the sake of simulation, it is better to assume it's the ray refracted from last surface has too extreme of an angle and is vignetted. 
        return None 

def PruneIntersectionArray(intersections, sphere_radius, clear_semi_diameter, cumulative_thickness):
    """
    For a set of vectors intersecting with the spherical surface, find the correct intersections. 

    :param intersections: 2 intersection points. 
    :param sphere_radius: radius of the spherical surface. 
    :param clear_semi_diameter: ¯|_(ツ)_/¯
    :param cumulative_thickness: cumulative thickness from the first vertex. 
    """

    result = []

    for i in range(len(intersections)):
        passed = IsolateIntersection(intersections[i], sphere_radius, clear_semi_diameter, cumulative_thickness)
        if (passed is not None):
            result.append(IsolateIntersection(intersections[i], sphere_radius, clear_semi_diameter, cumulative_thickness))

    return np.array(result)

def VectorsRefraction(incident_vectors, normal_vectors, n1, n2):
    """
    Calculates the refracted vectors given incident vectors, normal vectors, and the refractive indices.
    
    :param incident_vectors: Array of incident vectors (shape: Nx3).
    :param normal_vectors: Array of normal vectors (shape: Nx3).
    :param n1: Refractive index of the first medium.
    :param n2: Refractive index of the second medium.
    :return: Array of refracted vectors or None if total internal reflection occurs.
    """
    # Normalize incident and normal vectors
    incident_vectors = incident_vectors / np.linalg.norm(incident_vectors, axis=1, keepdims=True)
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=1, keepdims=True)

    # Compute the ratio of refractive indices
    n_ratio = n1 / n2
    
    # Dot product of incident vectors and normal vectors
    cos_theta_i = -np.einsum('ij,ij->i', incident_vectors, normal_vectors)
    
    # Calculate the discriminant to check for total internal reflection
    discriminant = 1 - (n_ratio ** 2) * (1 - cos_theta_i ** 2)
    
    # Handle total internal reflection (discriminant < 0)
    total_internal_reflection = discriminant < 0
    if np.any(total_internal_reflection):
        print("Total internal reflection occurs for some vectors.")
        return None
    
    # Calculate the refracted vectors
    refracted_vectors = n_ratio * incident_vectors + (n_ratio * cos_theta_i - np.sqrt(discriminant))[:, np.newaxis] * normal_vectors
    
    return refracted_vectors


#  ===========================================================================
"""
==============================================================================
"""

def main():
    posP = np.array([1, 2, -10])
    d = 6
    r = 20
    P_xy_projection = np.array([posP[0], posP[1], 0])

    # For on axis rays, the p projection is 0, which needs to be modified 
    if (np.linalg.norm(P_xy_projection) == 0):
        P_xy_projection = np.array([d, 0, 0])

    # Finding the important points 
    posA = d * ( P_xy_projection / np.linalg.norm(P_xy_projection) )
    posC = d * (-P_xy_projection / np.linalg.norm(P_xy_projection) )
    vecCA = posA - posC
    posB = FindB(posA, posC, posP, d)

    # Sample for the first surface 
    points = EllipsePeripheral(posA, posB, posC, posP, d, r) # Sample points in the ellipse area 
    pointsCircle = EllipsePeripheral(posA, posB, posC, posP, d, r, False) # Points that form the edge of the ellipse 
    
    # Casting rays 
    vecs = Normalized(np.transpose(points) - posP)

    # Finding intersections between ray and 1st surface 
    duplicateOrigin = np.tile(posP, (points.shape[1], 1))
    thoroughIntersection = raySphereIntersectionArray(duplicateOrigin, vecs, np.array([0, 0, r]), r); 
    isoIntersection01 = PruneIntersectionArray(thoroughIntersection, r, d, 0)
    sphericalNormal = SphericalNormal(r, isoIntersection01, np.array([0, 0, 0]))

    # Finding refracted ray 
    refracted01 = VectorsRefraction(isoIntersection01-posP, sphericalNormal, 1, 1.8)

    
    r2 = -10
    t2 = 4
    d2 = 6.5
    thoroughIntersection02 = raySphereIntersectionArray(isoIntersection01, refracted01, np.array([0, 0, r2 +t2]), r2)
    isoIntersection02 = PruneIntersectionArray(thoroughIntersection02, r2, d2, t2)
    sphericalNormal02 = -SphericalNormal(r2, isoIntersection02, np.array([0, 0, t2]))
    refracted02 = VectorsRefraction(isoIntersection02-isoIntersection01, sphericalNormal02, 1.8, 1)
    print("isolate interections ", refracted02)
    #isoIntersection02 = PruneIntersectionArray(thoroughIntersection02, r2, d, 2)
    #sphericalNormal02 = SphericalNormal(r, isoIntersection02, np.array([0, 0, 2]))
    #refracted02 = VectorsRefraction(isoIntersection02-isoIntersection01, sphericalNormal02, 1.5, 1)

    # Plot the findings 
    ax = PlotTest.Setup3Dplot()
    PlotTest.SetUnifScale(ax)
    PlotTest.AddXYZ(ax, 6)
    PlotTest.DrawIncidentPlane(ax, posP, posB, d)

    PlotTest.Draw3D(ax, pointsCircle[0], pointsCircle[1], pointsCircle[2])
    #PlotTest.DrawLine(ax, posP, posP + randVec, "m")
    #PlotTest.DrawPoint(ax, points)
    PlotTest.DrawSpherical(ax, r, d, 0)
    PlotTest.DrawSpherical(ax, r2, d2, t2)
    
    for v in isoIntersection01:
        PlotTest.DrawLine(ax, posP, v, lineColor = "r", lineWidth = 0.5)
    for v, n in zip(isoIntersection01, isoIntersection02):
        PlotTest.DrawLine(ax, v, n, lineColor = "r", lineWidth = 0.5)
    
    for v, n in zip(isoIntersection02, refracted02):
        PlotTest.DrawLine(ax, v, v+20*n, lineColor = "r", lineWidth = 0.5)

    plt.show()


#Rotation(0.61, np.array([0, 1, 0]), np.array([1, 1, 0]))
if __name__ == "__main__":
    main() 