
import numpy as np

# Utility 
def Normalized(inputVec):
    return inputVec / np.linalg.norm(inputVec)

def Minus90(inputRadian):
    return np.pi / 2 - inputRadian

def Rotation(theta, axis, inputVertex):
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
 
    return np.matmul(R, inputVertex) 
    
def Translate(inputVertex, translation):
    return np.transpose(np.transpose(inputVertex) + translation)

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

def CircularDistribution(radius = 1, layer = 5, densityScale = 0.02, powerCoef = 0.8, shrink = 1):
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

def SphericalNormal(sphere_radius, intersections, front_vertex, sequential = True):
    """
    Calculate the normal direction at the intersections. 

    :param sphere_radius: radius of the sperical surface. 
    :param intersections: points of intersections between incident rays and the surface. 
    :param front_vertex: vertex of the surface facing object side. 
    """

    # Offset from the front vertex to find the spherical origin 
    origin = front_vertex + np.array([0, 0, sphere_radius])

    # Negative radius will by default having their normals pointing to the positive z direction 
    # For sequential simulation, use the sign to invert the normal so that negative radius points to negative z 
    if (sequential): sign = np.sign(sphere_radius)
    else: sign = 1 

    return sign * Normalized(intersections - origin)

