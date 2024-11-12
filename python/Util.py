
import numpy as np
import math
from enum import Enum


# ==================================================================
""" ============= Consts, flags, and definitions =============== """
# ==================================================================


# Global flag for developing and debugging features 
DEVELOPER_MODE = True 


# Creates a RNG for the entire program to use 
RANDOM_SEED = 42 
rng = np.random.default_rng(seed=RANDOM_SEED)


# The threashold by which a raybatch will no longer propagate 
RADIANT_KILL = 0.001
# Changing this could increase accuracy, at the cost of increase time 


# Global variable for per spot sampling in image formation. 
PER_POINT_MAX_SAMPLE = 100
# This can be used to estimate and normalize the colors. 


# Fraunhofer symbols used in imager wavelength-RGB conversion 
# and material RI and Abbe calculation 
LambdaLines = {
    "i" : 365.01,  # Default UV cut 
    "h" : 404.66, 
    "g" : 435.84,  # Default B
    "F'": 479.99, 
    "F" : 486.13,  # Default secondary 
    "e" : 546.07,  # Default G
    "d" : 587.56,  
    "D" : 589.3,   # Default secondary
    "C'": 643.85,  # Default R 
    "C" : 656.27, 
    "r" : 706.52, 
    "A'": 768.2,   # Default IR cut 
    "s" : 852.11,
    "t" : 1013.98,
}


# Ways to fit a gate when input and gate has different aspect ratio 
class Fit(Enum):
    FILL = 1  # Stretch both horizontal and vertical direction to fill the entire gate 
    FIT = 2   # Proportionally scale the image to fit one axis and ensure the image is not cropped 


# Placeholder varible for arguments 
SOME_BIG_CONST = 1024
SOME_SML_CONST = 1e-10


# ==================================================================
""" ====================== Memory Management =================== """
# ==================================================================


class MemoryManagement():

    MaxMemory = 40 # In gig

    MaxRayBatchRatio = 0.5 # Leave space for other variables 

    RayBatchFloat32Szie = 4   # 4 bytes for float 32
    RayBatchComponents = 10   # 10 float32 in one raybatch entry 
    RayBatchUnitSize = 4 * RayBatchComponents # Memory size for 1 raybatch entry 

    @classmethod
    def AllowedRaybatchSize(self):
        allowedMemorySize = self.MaxMemory * self.MaxRayBatchRatio
        allowedMemorySizeByte = allowedMemorySize * (1024**3)
        return allowedMemorySizeByte / self.RayBatchUnitSize

# ==================================================================
""" ===================== 3D transformations =================== """
# ==================================================================


def Normalized(inputVec):
    return inputVec / np.linalg.norm(inputVec)


def ArrayNormalized(inputVec):
    """ Normalize an array of vectors """
    return inputVec / np.linalg.norm(inputVec, axis=1, keepdims=True)


def Partition(inputVec):
    return inputVec / (np.sum(inputVec) + SOME_SML_CONST)


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


# ==================================================================
""" ================== Generate Sample Points ================== """
# ==================================================================


# =========================== Presets ==========================
# Light:   layer = 5,    densityScale = 0.02,    powerCoef = 0.8
# Medium:  layer = 10,   densityScale = 0.0095,  powerCoef = 0.9
# Heavy:   layer = 60,   densityScale = 0.0004,  powerCoef = 0.7
# Prodc:   layer = 100,  densityScale = 0.0002,  powerCoef = 0.7
def CircularDistribution(radius = 1, layer = 5,    densityScale = 0.02,    powerCoef = 0.8, shrink = 0.95):
    """
    Accquire a distribution based on polar coordinate. 

    :param radius: radius of the circle, it is suggested to keep it at 1. 
    :param layer: number of layers of samples, each layer is in a concentric circle, with different radius.
    :param densityScale: with increase of radius, the delta area is used to calculate the points needed, this parameter is used to divide the delta area and get the number of sample points. The lower it is, the more sample. 
    :param powerCoef: due to the use of delta area depending on radius, linear scale will make outer edges having more samples. This parameter reduces this unevenness. 
    :param shrink: shrink the distribution a bit to avoid edge clipping when used later in the projection. 
    """

    # This function is not that controllable due to it based on scale and not the exact number of points. 
    # It also tend to have meridional or sagittal uneveness when the incoing rays have an extreme angle. 
    partitionLayer = (np.arange(layer) + 1.0) / layer
    lastArea = 0
    
    points = np.array([[0], [0], [0]])
    
    for current in partitionLayer:
        area = np.pi * current ** 2
        deltaArea = area - lastArea
        lastArea = area 
        
        pointsInLayer = int((deltaArea / densityScale) ** powerCoef)

        # Avoid zero point in layer situation 
        if(pointsInLayer == 0): pointsInLayer = 1

        layerH = current 
        layerTheta = np.arange(pointsInLayer) * ((np.pi * 2) / pointsInLayer) 

        layerPoints = np.array([layerH * np.cos(layerTheta), 
                                    layerH * np.sin(layerTheta), 
                                    np.zeros(pointsInLayer)])
        points = np.hstack((points, layerPoints))
        
    return points * radius * shrink


def RandomEllipticalDistribution(major_axis=1, minor_axis=1, samplePoints=500, shrink=0.95):
    """
    Generate a random, even distribution of points on an ellipse.
    
    :param major_axis: The radius of the major axis of the ellipse.
    :param minor_axis: The radius of the minor axis of the ellipse.
    :param samplePoints: Total number of points to generate.
    :param shrink: Shrink factor to avoid edge clipping when projecting later.
    :return: NumPy array of shape (2, samplePoints) representing the points on the ellipse.
    """
    
    # Step 1: Generate random angles between [0, 2*pi]
    angles = rng.uniform(0, 2 * np.pi, samplePoints)
    
    # Step 2: Generate random radial distances with a uniform distribution
    # sqrt to ensure even distribution over the circle
    radii = np.sqrt(rng.uniform(0, 1, samplePoints))
    
    # Step 3: Convert polar coordinates to Cartesian coordinates assuming a unit circle
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Step 4: Scale the points to match the major and minor axes of the ellipse
    x *= major_axis * shrink
    y *= minor_axis * shrink

    z = np.zeros(len(x)) # z defaults to 0
    
    # Step 5: Return the points as a (2, samplePoints) array
    points = np.vstack((x, y, z))
    
    return points


# ==================================================================
""" ================== Color and conversions =================== """
# ==================================================================

def LumiPeak(RGB):
    lumi = 0.2126*RGB[0] + 0.7152*RGB[1] + 0.0722 *RGB[2]
    return lumi


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
    :return: A NumPy array of wavelengths corresponding to the input RGB array
    """

    # Normalize RGB values to the range [0, 1]
    bits = 2.0 ** bitDepth - 1

    wavelengths = np.array([
        LambdaLines[primaries["R"]], 
        LambdaLines[primaries["G"]], 
        LambdaLines[primaries["B"]]
    ])

    radiants = np.array(RGB)
    #print("INSIDE RAD", radiants)

    if (len(secondaries) > 0):
        for secondary in secondaries:
            if(type(secondary) is str):
                currentWavelength = LambdaLines[secondary]
            else:
                currentRadiant = secondary # When passed as numbers 
            currentRadiant = 0
            wavelengths = np.append(wavelengths, currentWavelength)

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

            radiants = np.append(radiants, currentRadiant)

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
    R, G, B = 0.0, 0.0, 0.0

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
        G = 1 - ratio

    elif (LambdaLines[primaries["B"]] <= wavelength < LambdaLines[primaries["G"]]):
        # Between Blue and Green
        ratio = (wavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]]- LambdaLines[primaries["B"]])
        G = ratio
        B = 1 - ratio

    elif (LambdaLines[UVIRcut[0]] <= wavelength < LambdaLines[primaries["B"]]):
        # Between Blue primary and UV cutoff
        B = 1.0
        G = (wavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]])

    # Scale to bit depth and return
    
    if (useBits):
        R = int(np.clip(R * bits, 0, bits))
        G = int(np.clip(G * bits, 0, bits))
        B = int(np.clip(B * bits, 0, bits))

    return np.array([R, G, B])

    


def main():
    RGB = WavelengthToRGB(643.85) 
    wavelngth = RGBToWavelength(RGB)
    RGB1 = WavelengthToRGB(589.3) 

    print("RGB result: \t", RGB)
    print("wavelength: \t", wavelngth)
    print("RGB result: \t", RGB1)

    print(MemoryManagement.AllowedRaybatchSize())



if __name__ == "__main__":
    main()