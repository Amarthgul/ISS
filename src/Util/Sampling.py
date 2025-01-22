

from scipy.stats import qmc
from scipy.stats.qmc import PoissonDisk

from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import RNG, RefreshRNG, ONE

# =========================== Presets ==========================
# Light:   layer = 5,    densityScale = 0.02,    powerCoef = 0.8
# Medium:  layer = 10,   densityScale = 0.0095,  powerCoef = 0.9
# Heavy:   layer = 60,   densityScale = 0.0004,  powerCoef = 0.7
# Prodc:   layer = 100,  densityScale = 0.0002,  powerCoef = 0.7
def CircularDistribution(radius = 1, layer = 5,    densityScale = 0.02,    powerCoef = 0.8, shrink = 1):
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
    partitionLayer = (bd.arange(layer) + 1.0) / layer
    lastArea = 0
    
    points = bd.array([[0], [0], [0]])
    
    for current in partitionLayer:
        area = bd.pi * current ** 2
        deltaArea = area - lastArea
        lastArea = area 
        
        pointsInLayer = int((deltaArea / densityScale) ** powerCoef)

        # Avoid zero point in layer situation 
        if(pointsInLayer == 0): pointsInLayer = 1

        layerH = current 
        layerTheta = bd.arange(pointsInLayer) * ((bd.pi * 2) / pointsInLayer) 

        layerPoints = bd.array([layerH * bd.cos(layerTheta), 
                                    layerH * bd.sin(layerTheta), 
                                    bd.zeros(pointsInLayer)])
        points = bd.hstack((points, layerPoints))
        
    return points * radius * shrink


def RandomEllipticalDistribution(major_axis=1, minor_axis=1, samplePoints=500, zDepth = 0, shrink=1, groupByPoint=False):
    """
    Generate a random, even distribution of points on an ellipse.
    
    :param major_axis: The radius of the major axis of the ellipse.
    :param minor_axis: The radius of the minor axis of the ellipse.
    :param samplePoints: Total number of points to generate.
    :param shrink: Shrink factor to avoid edge clipping when projecting later.
    :param groupByPoint: If True, the points are grouped by point as (n, 3), if False, the points are grouped by axis.

    :return: Array representing the points on the ellipse in shape (n, 3) or (3, n).
    """
    # RNG is refreshed for every call while remaining deterministic 
    RefreshRNG()

    # Generate random angles between [0, 2*pi]
    angles = RNG.uniform(0, 2 * bd.pi, samplePoints)
    
    # Generate random radial distances with a uniform distribution
    # sqrt to ensure even distribution over the circle
    radii = bd.sqrt(RNG.uniform(0, 1, samplePoints))

    # Convert polar coordinates to Cartesian coordinates assuming a unit circle
    x = radii * bd.cos(angles)
    y = radii * bd.sin(angles)
    
    # Scale the points to match the major and minor axes of the ellipse
    x *= major_axis * shrink
    y *= minor_axis * shrink

    zDepth = bd.ones(len(x)) * zDepth
    
    if(groupByPoint):
        return bd.vstack((x, y, zDepth)).T
    else:
        return bd.vstack((x, y, zDepth))


def PoissonDiskDistribution(semiDiameter=1, zDepth = 0, samplePoints=128):


    
    import numpy as np 

    theoreticalSampleCount = int(samplePoints * (4.0 / bd.pi))
    radius =  (2 *semiDiameter) / np.sqrt(theoreticalSampleCount)

    centerOffset = np.array([0.5, 0.5])

    engine = qmc.PoissonDisk(d=2, radius=radius)

    while(True):
        sample = engine.fill_space() 
        radius /= np.sqrt(1.41)
        engine = qmc.PoissonDisk(d=2, radius=radius)

        if(sample.shape[0] >= theoreticalSampleCount):
            sample -= centerOffset
            within = sample[np.linalg.norm(sample, axis=1) < 0.5]
            while(True):
                if(within.shape[0] >= samplePoints):
                    break
                engine.reset()
                sample = engine.fill_space() - centerOffset

    selectedIndices = bd.random.choice(sample.shape[0], samplePoints, replace=False)
    planarPoints = sample[selectedIndices]
    return bd.vstack((planarPoints.T, bd.ones(samplePoints) * zDepth)).T * semiDiameter


