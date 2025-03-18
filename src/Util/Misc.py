

import math



from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import RNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, LambdaLines, RefreshRNG, Axis, UP_DIR, ORIGIN



def NumpyConversion(ary):
    if(backend_name == 'cupy'):
        return bd.asnumpy(ary)
    else: 
        return ary
    
    
# ==================================================================
""" ===================== 3D transformations =================== """
# ==================================================================


def Magnitude(inputVec):
    return bd.linalg.norm(inputVec)


def ArrayMagnitude(inputVec):
    return bd.linalg.norm(inputVec, axis=1)


def Normalized(inputVec):
    """
    Normalize a single vector. 
    """
    mag = bd.linalg.norm(inputVec)
    
    if (mag == 0):
        return 0
    
    return inputVec / mag


def ArrayNormalized(inputVec):
    """ 
    Normalize an array of vectors in shape (n, 3). 
    """
    return inputVec / bd.linalg.norm(inputVec, axis=1, keepdims=True)


def GridNormalized(inputVec):
    """
    This method is designed to normalized a (m, n, 3) array represneting many 3D vectors. 
    """
    norms = bd.linalg.norm(inputVec, axis=2, keepdims=True)

    # To avoid division by zero, add a small epsilon.
    epsilon = 1e-9
    norms = bd.maximum(norms, epsilon)

    # Normalize the vectors: each vector is divided by its norm.
    return inputVec / norms


def Partition(inputVec):
    return inputVec / (bd.sum(inputVec) + NEAR_ZERO)


def Minus90(inputRadian):
    return bd.pi / 2 - inputRadian


def Rotate(theta, axis, ibdutVertex):
    """
    Rotate the ibdut vertex by theta radians along y-axis.
    :param theta: Angle in radians.
    :param axis: Axis of rotation.
    :param ibdutVertex: Ibdut vertex.
    :return: Rotated vertex
    """
    R = bd.array([
        [bd.cos(theta) + axis[0]**2 * (1 - bd.cos(theta)), 
         axis[0] * axis[1] * (1 - bd.cos(theta)) - axis[2] * bd.sin(theta), 
         axis[0] * axis[2] * (1 - bd.cos(theta)) + axis[1]],
        [axis[0] * axis[1] * (1 - bd.cos(theta)) + axis[2] * bd.sin(theta), 
         bd.cos(theta) + axis[1]**2 * (1 - bd.cos(theta)),
         axis[1] * axis[2] * (1 - bd.cos(theta)) -axis[1]],
        [axis[2] * axis[0] * (1 - bd.cos(theta)) - axis[1] * bd.sin(theta), 
         axis[2] * axis[1] * (1 - bd.cos(theta)) + axis[0] * bd.sin(theta), 
         bd.cos(theta) + axis[2]**2 * (1 - bd.cos(theta))]
    ])
 
    return bd.matmul(R, ibdutVertex) 


def ArrayRotate(theta, axis, pivot, points):
    """
    Rotate 3D points around a specified pivot along a given axis.
    
    :param theta: (float) Rotation angle in radians.
    :param axis: (array-like) 3D unit vector representing the rotation axis.
    :param pivot: (array-like) 3D point representing the pivot (center of rotation).
    :param points: (ndarray) Array of shape (n, 3) containing the 3D points to rotate.
    
    :return: ndarray: Rotated points of shape (n, 3).
    """
    # Ensure axis is a unit vector
    axis = axis / bd.linalg.norm(axis)
    
    # Compute rotation matrix (Rodrigues' Rotation Formula)
    cos_theta = bd.cos(theta)
    sin_theta = bd.sin(theta)
    one_minus_cos = 1 - cos_theta
    
    ux, uy, uz = axis  # Extract components of the unit axis
    
    R = bd.array([
        [cos_theta + ux**2 * one_minus_cos, 
         ux * uy * one_minus_cos - uz * sin_theta, 
         ux * uz * one_minus_cos + uy * sin_theta],
        [uy * ux * one_minus_cos + uz * sin_theta, 
         cos_theta + uy**2 * one_minus_cos, 
         uy * uz * one_minus_cos - ux * sin_theta],
        [uz * ux * one_minus_cos - uy * sin_theta, 
         uz * uy * one_minus_cos + ux * sin_theta, 
         cos_theta + uz**2 * one_minus_cos]
    ])
    
    # Ensure points are a 2D array
    points = bd.asarray(points)  # Convert to array in case it's a list
    if points.ndim == 1:
        points = points[None, :]  # Reshape to (1, 3) if only one point is given
    
    # Translate points to pivot
    translated_points = points - pivot

    # Apply rotation matrix correctly using matrix multiplication
    rotated_points = bd.dot(translated_points, R.T)  # Correct matrix multiplication

    # Translate points back
    final_points = rotated_points + pivot

    return final_points


def Translate(ibdutVertex, translation):
    return bd.transpose(bd.transpose(ibdutVertex) + translation)


# ==================================================================
""" =========================== Math =========================== """
# ==================================================================


def Sigmoid(input, amp = 8, offset=-.5):
    return ONE / (ONE + bd.exp(-amp * (input+offset) ))


def InvSigmoid(input, amp = 10, offset=0.5):
    return offset + bd.log(input / (ONE-input)) / 10


def CartesianToPolar(point, origin):
        delta = point - origin
        angle = bd.arctan2(delta[1], delta[0])
        radius = bd.linalg.norm(delta)
        return angle, radius


def MovingAverageSmoothing(values):
    """
    Smooth the data with end points preservation. 

    :param values: data values to be smoothed. 

    :return: copied and smoothed data. 
    """

    smoothed = values.copy()
    smoothed[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3
    return smoothed


def GaussianSmooth(values, sigma=1):
    """
    Apply Gaussian smoothing to a set of values. 

    :param values: data values to be smoothed. 
    :param sigma: standard deviation that controls how strong the smooth effect is, the higher the value, the stronger the smoothing. 

    :return: new set of data that have been smoothed. 
    """

    if(backend_name=='cupy'):
        from cupyx.scipy.ndimage import gaussian_filter1d
    else:
        from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(values, sigma=sigma, mode='nearest')
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    
    return smoothed


# ==================================================================
""" ========================= Geometries ======================= """
# ==================================================================


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
    dot_product = bd.dot(plane_normal, line_direction)
    
    # If the dot product is 0, the line is parallel to the plane (no intersection or line lies on the plane)
    if bd.isclose(dot_product, 0):
        print("The line is parallel to the plane and does not intersect.")
        return None
    
    # Calculate the parameter t for the line equation
    t = bd.dot(plane_normal, (point_on_plane - point_on_line)) / dot_product
    
    # Calculate the intersection point using the parametric line equation
    intersection_point = point_on_line + t * line_direction
    
    return intersection_point


def SphericalNormal(sphere_radius, intersections, front_vertex, sequential = True):
    """
    Calculate the normal direction at the intersections. 

    :param sphere_radius: radius of the sperical surface. 
    :param intersections: points of intersections between incident rays and the surface. 
    :param front_vertex: vertex of the surface facing object side. 
    """

    # Offset from the front vertex to find the spherical origin 
    origin = front_vertex + bd.array([0, 0, sphere_radius])

    # Negative radius will by default having their normals pointing to the positive z direction 
    # For sequential simulation, use the sign to invert the normal so that negative radius points to negative z 
    if (sequential): sign = bd.sign(sphere_radius)
    else: sign = 1 

    return sign * Normalized(intersections - origin)


def angleBetweenVectors(v1, v2, use_degrees = False):
    """
    Calculates the angle between two vectors in radians.
    
    :param v1: First vector (numpy array).
    :param v2: Second vector (numpy array).
    :return: The angle between the vectors in degrees.
    """
    # Normalize vectors to avoid floating-point issues
    v1_normalized = bd.linalg.norm(v1)
    v2_normalized = bd.linalg.norm(v2)
    
    # Calculate the dot product of the two vectors
    dot_product = bd.dot(v1, v2)
    
    # Calculate the magnitudes (norms) of the vectors
    norm_v1 = bd.linalg.norm(v1)
    norm_v2 = bd.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure cos_theta is in the valid range [-1, 1] to avoid errors due to floating-point precision
    cos_theta = bd.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = bd.arccos(cos_theta)
    
    # Convert to degrees
    if use_degrees:
        bd.degrees(angle_radians)
    else:
        return angle_radians


def AxialDistance(vectors, axis):
    """
    Calculate the distance between the max and min values of 3D vectors along a specified axis.

    Parameters:
    vectors (ndarray): N x 3 array of 3D vectors.
    axis (str): Axis to calculate the distance ('x', 'y', 'z').

    Returns:
    float: Distance between max and min values along the specified axis.
    """

    # Extract the relevant column for the specified axis
    axis_values = vectors[:, axis]
    
    # Calculate max and min along the axis
    max_value = bd.max(axis_values)
    min_value = bd.min(axis_values)
    
    # Calculate and return the distance
    return max_value - min_value


def AxialSnap(points):
    points[points[:, 0] < AXIAL_ZERO, 1] = 0
    points[points[:, 1] < AXIAL_ZERO, 1] = 0


def TransversalDistance(vectors):
    """
    Find the distance of vectors to the optical axis (z axis). 
    """

    return ArrayMagnitude(vectors[:, :2])


def RayTriangleIntersection(origin, direction, A, B, C, epsilon=1e-6):
    """
    Check if a ray intersects a triangle in 3D using the Möller–Trumbore algorithm.
    
    Parameters:
    origin (ndarray): Origin of the ray (3D point).
    direction (ndarray): Direction of the ray (3D vector, normalized).
    A, B, C (ndarray): Vertices of the triangle (3D points).
    epsilon (float): Small threshold for numerical stability.
    
    Returns:
    bool: True if the ray intersects the triangle, False otherwise.
    float: The intersection distance `t` along the ray if an intersection occurs.
    """
    # Compute edge vectors
    E1 = B - A
    E2 = C - A
    
    # Compute determinant
    P = bd.cross(direction, E2)
    det = bd.dot(E1, P)
    
    # If determinant is near zero, the ray is parallel to the triangle
    if abs(det) < epsilon:
        return False, None
    
    inv_det = 1.0 / det
    
    # Compute the distance from A to the ray origin
    T = origin - A
    
    # Compute barycentric coordinate u
    u = bd.dot(T, P) * inv_det
    if u < 0 or u > 1:
        return False, None
    
    # Compute barycentric coordinate v
    Q = bd.cross(T, E1)
    v = bd.dot(direction, Q) * inv_det
    if v < 0 or u + v > 1:
        return False, None
    
    # Compute intersection distance t
    t = bd.dot(E2, Q) * inv_det
    if t < 0:
        return False, None
    
    return True, t


def PointsInTriangle(points, A, B, C):
    """
    Check if points are inside a triangle in 3D space.
    
    Parameters:
    points (ndarray): Array of shape (n, 3) representing the points to test.
    A, B, C (ndarray): Vertices of the triangle (3D points).
    
    Returns:
    ndarray: Boolean array of shape (n,) indicating whether each point is inside the triangle.
    """
    # Compute edge vectors of the triangle
    E1 = B - A
    E2 = C - A

    # Compute normal vector of the triangle
    normal = bd.cross(E1, E2)
    normal = normal / bd.linalg.norm(normal)

    # Project points onto the triangle plane
    AP = points - A
    plane_distances = bd.dot(AP, normal)
    projected_points = points - bd.outer(plane_distances, normal)

    # Convert to 2D coordinates in the triangle plane
    E1_norm = E1 / bd.linalg.norm(E1)
    E2_proj = E2 - bd.dot(E2, E1_norm) * E1_norm
    E2_norm = E2_proj / bd.linalg.norm(E2_proj)
    local_coords = bd.column_stack([bd.dot(AP, E1_norm), bd.dot(AP, E2_norm)])

    # Compute barycentric coordinates
    E1_2D = bd.array([bd.dot(E1, E1_norm), bd.dot(E1, E2_norm)])
    E2_2D = bd.array([bd.dot(E2_proj, E1_norm), bd.dot(E2_proj, E2_norm)])
    T = bd.linalg.solve(bd.column_stack([E1_2D, E2_2D]), local_coords.T).T

    u, v = T[:, 0], T[:, 1]
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    return inside




def main():
    points = bd.array([[1, 0, 0], [1, 2, 3]])
    colors = bd.array([[1, 1, 1], [1, 0.5, 0.25]])

    




if __name__ == "__main__":
    main()