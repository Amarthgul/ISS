

import math



from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import RNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, LambdaLines, RefreshRNG, Axis


# ==================================================================
""" ====================== Memory Management =================== """
# ==================================================================

class MemoryManagement():

    MaxMemory = 40 # In gig

    MaxRayBatchRatio = 0.5 # Leave space for other variables 

    RayBatchFloat32Szie = 4   # 4 bytes for float 32
    RayBatchComponents = 10   # 10 float32 in one raybatch entry 
    RayBatchUnitSize = 4 * RayBatchComponents # Memory size for 1 raybatch entry 

    _GigToByteConversion = 1024**3

    @classmethod
    def AllowedRaybatchSize(self):
        allowedMemorySize = self.MaxMemory * self.MaxRayBatchRatio
        allowedMemorySizeByte = allowedMemorySize * self._GigToByteConversion
        return allowedMemorySizeByte / self.RayBatchUnitSize


# ==================================================================
""" ===================== 3D transformations =================== """
# ==================================================================


def Magnitude(inputVec):
    return bd.linalg.norm(inputVec)


def ArrayMagnitude(inputVec):
    return bd.linalg.norm(inputVec, axis=1, keepdims=True)


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
    Normalize an array of vectors. 
    """
    return inputVec / bd.linalg.norm(inputVec, axis=1, keepdims=True)


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


# ==================================================================
""" ================== Color and conversions =================== """
# ==================================================================


def LumiPeak(RGB, bitDepth = 8):
    """
    Naive solution for calculating the luminance based on the RGB channel of a pixel/point. 

    :param RGB: RGB value of a pixel as [R, G, B]. 
    :param bitDepth: bitDepth if the RGB array is not in the [0, 1] range. 
    """
    if(bd.sum(RGB) > 3):
        RGB = RGB / (2**bitDepth)

    lumi = 0.2126*RGB[0] + 0.7152*RGB[1] + 0.0722 *RGB[2]
    return lumi


def LumiPeakArray(RGB, bitDepth = 8):
    """
    Naive solution for calculating the luminance based on the RGB channel of an image/array.  

    :param RGB: RGB array in the shape of (m, n, 3). 
    :param bitDepth: bitDepth if the RGB array is not in the [0, 1] range.
    """
    if(RGB.max() > 1):
        RGB = RGB / (2**bitDepth)

    return 0.2126*RGB[:, :, 0] + 0.7152*RGB[:, :, 1] + 0.0722 *RGB[:, :, 2] 


def RGBToWavelength(RGB, 
                    primaries = {"R": "C'", "G": "e", "B":"g"}, 
                    secondaries = ["F", "D"], 
                    UVIRcut = ["i", "A'"],
                    bitDepth=8):
    """
    Convert an RGB values to corresponding wavelengths and intensity/radiant flux.
    
    :param RGB: RGB values
    :param primaries: A dictionary mapping RGB to primary wavelength lines (default: {"R": "C'", "G": "e", "B": "g"})
    :param secondaries: A dictionary mapping secondary colors to wavelength lines (optional)
    :param UVIRcut: Cut wavelength for ultraviolet and infrared, the first term is UV and the second is IR. 
    :return: A NumPy array of wavelengths corresponding to the ibdut RGB array
    """

    # Normalize RGB values to the range [0, 1]
    bits = 2.0 ** bitDepth - 1

    wavelengths = bd.array([
        LambdaLines[primaries["R"]], 
        LambdaLines[primaries["G"]], 
        LambdaLines[primaries["B"]]
    ])

    radiants = bd.array(RGB)
    #print("INSIDE RAD", radiants)

    if (len(secondaries) > 0):
        for secondary in secondaries:
            if(type(secondary) is str):
                currentWavelength = LambdaLines[secondary]
            else:
                currentRadiant = secondary # When passed as numbers 
            currentRadiant = 0
            wavelengths = bd.append(wavelengths, currentWavelength)

            # Between IR limit and Red line 
            if(currentWavelength < LambdaLines[UVIRcut[1]] and currentWavelength > LambdaLines[primaries["R"]]):
                # Using red radiant and reduce the intensity depending on how far it is away from the red line 
                currentRadiant = radiants[0] * ( (currentWavelength - LambdaLines[primaries["R"]]) / (LambdaLines[UVIRcut[1]] - LambdaLines[primaries["R"]]) )

            # Between Red line and Green line 
            elif(currentWavelength < LambdaLines[primaries["R"]] and currentWavelength > LambdaLines[primaries["G"]]):
                # Find the ratio between red and green 
                ratio = (currentWavelength - LambdaLines[primaries["G"]]) / (LambdaLines[primaries["R"]] - LambdaLines[primaries["G"]])

                currentRadiant = radiants[0] * ratio + radiants[1] * (1 - ratio)

            # Between Green line and Blue line 
            elif(currentWavelength < LambdaLines[primaries["G"]] and currentWavelength > LambdaLines[primaries["B"]]):
                # Find the ratio between green and blue 
                ratio = (currentWavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]] - LambdaLines[primaries["B"]])

                currentRadiant = radiants[1] * ratio + radiants[2] * (1 - ratio)

            # Between Blue line and UV limit 
            elif(currentWavelength < LambdaLines[primaries["B"]] and currentWavelength > LambdaLines[UVIRcut[0]]):
                currentRadiant = radiants[0] * ( (currentWavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]]) )

            radiants = bd.append(radiants, currentRadiant)

    return (wavelengths, radiants)


def RGBToWavelengthArray(RGB, 
                primaries = {"R": "C'", "G": "e", "B":"g"}, 
                secondaries = ["F", "D"], 
                UVIRcut = ["i", "A'"],
                bitDepth=8):
    """
    Convert an RGB values to corresponding wavelengths and intensity/radiant flux.
    
    :param RGB: a 3D array in shape (m, n, 3) representing the RGB of an image. 
    :param primaries: A dictionary mapping RGB to primary wavelength lines (default: {"R": "C'", "G": "e", "B": "g"}).
    :param secondaries: A dictionary mapping secondary colors to wavelength lines (optional)
    :param UVIRcut: Cut wavelength for ultraviolet and infrared, the first term is UV and the second is IR. 

    :return: A NumPy array of wavelengths corresponding to the input RGB array. 
    """

    if(RGB.max() > 1):
        RGB = RGB / (2**bitDepth)

    width = RGB.shape[0]
    height = RGB.shape[1]

    wavelengths = bd.array([
        LambdaLines[primaries["R"]], 
        LambdaLines[primaries["G"]], 
        LambdaLines[primaries["B"]]
    ])
    wavelengths = bd.tile(wavelengths, (width, height, 1))

    radiants = bd.array(RGB)

    # TODO: Add secondary support? 

    return (wavelengths, radiants)


def WavelengthToRGB(wavelength, 
                    primaries={"R": "C'", "G": "e", "B": "g"},
                    UVIRcut=["i", "A'"],
                    useBits=False, bitDepth=8):
    """
    Convert a wavelength to RGB values.
    
    :param wavelength: Wavelength to convert (in nm).
    :param primaries: A dictionary mapping RGB to primary wavelength lines.
    :param UVIRcut: List with UV and IR limits for cutoff.
    :param bitDepth: Bit depth for RGB values.
    :return: RGB values as integers in the range [0, 255].
    """

    # TODO: Note that this function works in conjunction with RGBToWavelength, \
    # thus should be edited together with RGBToWavelength to ensure the results. 
    # For this reason, there is no secondaries here since they are just linear interpolations and can be just ignored  

    # Default RGB values (intensity normalized to 0-1)
    R, G, B = ZERO, ZERO, ZERO

    # Calculate bit depth scaling factor
    bits = 2 ** bitDepth - 1

    # Check where the wavelength falls and interpolate
    if (LambdaLines[primaries["R"]] <= wavelength <= LambdaLines[UVIRcut[1]]):
        # Between Red primary and IR cutoff
        R  = (LambdaLines[UVIRcut[1]] - wavelength) / (LambdaLines[UVIRcut[1]] - LambdaLines[primaries["R"]])

    elif (LambdaLines[primaries["G"]] <= wavelength < LambdaLines[primaries["R"]]):
        # Between Green and Red
        ratio = (wavelength - LambdaLines[primaries["G"]]) / (LambdaLines[primaries["R"]] - LambdaLines[primaries["G"]])
        R = ratio
        G = ONE - ratio

    elif (LambdaLines[primaries["B"]] <= wavelength < LambdaLines[primaries["G"]]):
        # Between Blue and Green
        ratio = (wavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]]- LambdaLines[primaries["B"]])
        G = ratio
        B = ONE - ratio

    elif (LambdaLines[UVIRcut[0]] <= wavelength < LambdaLines[primaries["B"]]):
        # Between Blue primary and UV cutoff
        B = ONE
        G = (wavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]])


    R, G, B = bd.array(R), bd.array(G), bd.array(B)

    if (useBits):
        return bd.clip(bd.array([R, G, B]) * bits, ZERO, bits)
    else:
        return bd.array([R, G, B])

        
def ColorTuplePLT(arrayRGB):
    if(backend_name == 'cupy'):
        arrayRGB = bd.asnumpy(arrayRGB)

    return tuple(arrayRGB)



def main():
    print(WavelengthToRGB(550.0))




if __name__ == "__main__":
    main()