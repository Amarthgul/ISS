




"""
This file handles the initial generation of rays. 

There are two method of doing so. One is to project the rays toward the first surface, and the second is to project them towards the entrence pupil. 

Projecting to first surface is more faithful as it is less discriminative, and is better when glare and flares are needed. But this method is slower. 

Projecting to the entrance pupil is faster, but it will offer less flare and glares. 

"""


from Util.Backend import backend as bd
from Util.Backend import constant
from Util.Misc import Normalized, ArrayNormalized, CircularDistribution, angleBetweenVectors, Rotate, Translate, RandomEllipticalDistribution
from Util.Globals import NORMAL_RADIANT, INIT_PHASE_DIFF, ZERO, ONE, FAR_DISTANCE

from Util.PlotTest import DrawRaybatch

from .RayBatch import RayBatch 


# ==================================================================
""" ==================== First Surface Method ================== """
# ==================================================================


def FindB(posA, posC, posP):
    """
    Find the position of point B. 
    :param posA: position of point A.
    :param posC: position of point C.
    :param posP: position of point P.
    """
    vecPA = posA - posP   
    vecPC = posC - posP  
    vecN = Normalized((Normalized(vecPA) + Normalized(vecPC)) / 2)
    
    vecPCN = Normalized(vecPC)
    t = (bd.dot(vecN, (posA - posC))) / bd.dot(vecN, vecPCN)

    return posC + vecPCN * t 


def GenerateEllipse(posA, posB, posC, posP, sd, r, useDistribution = True):
    """
    Find the points on the ellipse and align it to the AB plane. 

    :param posA: position of point A. 
    :param posB: position of point B. 
    :param posC: position of point C. 
    :param posP: position of point P. 
    :param sd: clear semi diameter of the surface. 
    :param r: radius of the surface. 
    :param useDistribution: when enabled, the method returns a distribution of points in the ellipse area instead of the outline of the ellipse. 
    """

    offset = bd.zeros(3)
    offset[2] = posA[2]
    onAxis = False 

    one = constant(1.0)
    zero = constant(0)

    # On axis scenario 
    if (bd.isclose(posP[0], zero) and bd.isclose(posP[1], zero)):
        P_xy_projection = bd.array([sd, zero, zero])
        onAxis = True 
    else:
        # Util vectors 
        P_xy_projection = posP.copy()
        P_xy_projection[2] = zero

    vecCA = posA - posC
    
    # On axis rays can be grealty simplified 
    if(onAxis):
        if (useDistribution):
            # Move the point along the z axis 
            points = bd.transpose(RandomEllipticalDistribution()) + offset
            # Scale it on the two semi-major axis 
            points = bd.transpose(points * bd.array([sd, sd, one]))
            
        else:
            # Generate the contour of the ellipse 
            theta = bd.linspace(0, 2 * bd.pi, 100)
            x = bd.cos(theta) 
            y = bd.sin(theta) 
            z = bd.ones(len(x)) * posA[2]
            points = bd.array([x, y, z])
        return points 
    
    else:
        # Lengths to calculate semi-major axis length 
        BB = abs((2 * sd) * ((posP[2] - posB[2]) / posP[2]))
        AC = bd.linalg.norm(posA - posC)
        
        # Semi-major axis length 
        a = bd.linalg.norm(posA - posB) / 2
        b = bd.sqrt(BB * AC) / 2
        
        # Temporary array to store the semi-major and semi-minor axis lengths
        temp = bd.ones(3) 
        # Just for cupy compatibility 

        # Calculate the ellipse 
        if (useDistribution):
            temp[0] = a
            temp[1] = b 
            # Move the point along the z axis 
            points = bd.transpose(CircularDistribution()) + offset
            # Scale it on the two semi-major axis 
            points = bd.transpose(points * temp)
            
        else:
            # Generate the contour of the ellipse 
            theta = bd.linspace(0, 2 *bd.pi, 100)
            x = b * bd.cos(theta) 
            y = a * bd.sin(theta) 
            z = bd.ones(len(x)) * posA[2]
            points = bd.array([x, y, z])
        
        # Rotate the ellipse to it faces the right direction in the world xy plane,
        # i.e., one of its axis coincides with the tangential plane 
        theta_1 = angleBetweenVectors(posA, bd.array([0, 1, 0]))
        trans_1 = Rotate(-theta_1, bd.array([0, 0, 1]), points)
        
        # Move the points to be in tangent with A 
        trans_1 = Translate(trans_1, P_xy_projection * (sd - a)) 
        
        # Rotate the ellipse around A it fits into the AB plane 
        theta = angleBetweenVectors(posB-posA, posC-posA)

        temp = bd.zeros(3) # Another temporary array for cupy compatibility
        temp[0] = -vecCA[1]
        temp[1] = vecCA[0]
        axis = Normalized(temp)

        trans_2 = Translate(trans_1, -posA)
        trans_2 = Rotate(-theta, axis, trans_2)
        trans_2 = Translate(trans_2, posA)

        return trans_2


def InitRays(r, sd, posP, wavelength = 550):

    zero = 0 # For cupy compatibility

    P_xy_projection = posP.copy()
    P_xy_projection[2] = 0

    offset = bd.zeros(3)
    offset[2] = abs(r) - bd.sqrt(r**2 - sd**2)
    offset *= bd.sign(r)

    if(bd.isclose(bd.linalg.norm(P_xy_projection), 0)):
        # On aixs 
        posA = bd.array([sd, zero, zero]) + offset
        posC = bd.array([sd, zero, zero]) + offset
    else:
        # Off axis
        posA = sd * ( P_xy_projection / bd.linalg.norm(P_xy_projection) ) + offset
        posC = sd * (-P_xy_projection / bd.linalg.norm(P_xy_projection) ) + offset
    posB = FindB(posA, posC, posP)

    points = bd.transpose(GenerateEllipse(posA, posB, posC, posP, sd, r)) # Sample points in the ellipse area 
    _temp = GenerateEllipse(posA, posB, posC, posP, sd, r, False) # Points that form the edge of the ellipse 

    vecs = ArrayNormalized(points - posP)

    # Creating the ray batch.  
    # For some reason vecs is often not registered with indexing assignment, hstack is thus used to force the composition of the raybatch. 
    mat1 = bd.tile(bd.array([posP[0], posP[1], posP[2]]), (vecs.shape[0], 1))

    temp = bd.zeros(5)
    temp[0] = wavelength
    temp[1] = NORMAL_RADIANT    # Sagittal radiant
    temp[2] = NORMAL_RADIANT    # Tangential radiant
    temp[3] = INIT_PHASE_DIFF   # Phase difference 
    mat2 = bd.tile(temp, (vecs.shape[0], 1))
    mat = bd.hstack((bd.hstack((mat1, vecs)), mat2))

    rayBatch = RayBatch( mat )

    # Set the initial rays' direction into the newly spawned rays 
    # TODO: this does not seem to work when y value of the point is set to bigger than 1? 
    #self.rayBatch.value[:, 3:6] = vecs

    # if(ENABLE_RAYPATH):
    #     # Append the starting point into ray path 
    #     self.rayPath.append(np.copy(mat1))

    return rayBatch 


# TODO: implement a new set of first surface method that takes an array of sources 



# ==================================================================
""" =================== Entrance Pupil Method ================== """
# ==================================================================

# TODO: Implement the entrance pupil method


# ==================================================================
""" ======================= Emit From Stop ===================== """
# ==================================================================


def EmitFromStop(stopIndex, stopVertex, previousSD, nextSD, previousSDT, nextSDT, numRays=20, wavelength = 550.0):
    """
    Emit rays from the center of the stop towards the object and image side.
    The angle of the rays are determined by the edge of previous and next surface.

    :param stopIndex: index of the stop among the lens surfaces.
    :param stopVertex: vector location of the center of stop in world space.
    :param previousSD: clear semi-diameter of the previous surface.
    :param nextSD: clear semi-diameter of the next surface.
    :param previousSDT: z-position of the edge of the previous surface.
    :param nextSDT: z-position of the edge of the next surface.
    :param numRays: number of rays to be emitted.
    :param wavelength: wavelength of the rays in nm.

    :return: two RayBatch objects, one for the object side and the other for the image side.
    """
    
    stopCT = stopVertex[2]

    objectSideTheta = bd.arctan(previousSD / bd.abs(previousSDT-stopCT))
    imageSideTheta = bd.arctan(nextSD / bd.abs(nextSDT-stopCT))

    # Determine the max angle of the rays
    theta = objectSideTheta if (objectSideTheta < imageSideTheta) else imageSideTheta

    angularSteps = bd.linspace(0, theta, numRays)
    
    vectors = bd.array([(ZERO, bd.sin(a), bd.cos(a)) for a in angularSteps])

    vertices = bd.tile(stopVertex, (vectors.shape[0], 1))
    temp = bd.zeros(3)
    temp[0] = wavelength
    temp[1] = NORMAL_RADIANT    # Sagittal radiant
    temp[2] = NORMAL_RADIANT    # Tangential radiant
    temp = bd.tile(temp, (vectors.shape[0], 1))
    

    # concatenate creates a new array so the two new arry should be independent of each other
    objectSideRB = RayBatch(bd.concatenate([
        vertices, 
        -vectors, 
        temp, 
        angularSteps[:, bd.newaxis], 
        bd.tile(constant(stopIndex), (vectors.shape[0], 1))],
    axis=1))
    imageSideRB = RayBatch(bd.concatenate([
        vertices, 
        vectors, 
        temp, 
        angularSteps[:, bd.newaxis], 
        bd.tile(constant(stopIndex), (vectors.shape[0], 1))],
    axis=1))
    # Note that phase difference is replaced with angles. 
    # This record the angle of the rays so that after propagation, the angle can be used to find the entrance pupil. 

    return objectSideRB, imageSideRB



# ==================================================================
""" =================== Emit From Object Space ================= """
# ==================================================================

def EmitFromObjectSpace(firstSurfaceSD, planar=True, numRays=21, wavelength = 550.0):
    """
    Emit rays from the object space infinitely towards the 1st surface of the lens.

    :param firstSurfaceSD: clear semi-diameter of the first surface.
    :param planar: if True, rays are emitted along the y-plane.
    :param numRays: number of rays to be emitted, it is suggested to have an odd number so that there is one ray along the optical axis.
    :param wavelength: wavelength of the rays in nm.

    :return: a RayBatch object pointing from infinite object space to the 1st surface.
    """

    if(planar):
        # Emit rays along the y plane
        y_values = bd.linspace(firstSurfaceSD, -firstSurfaceSD, numRays)
        position = bd.array([[ZERO, y, FAR_DISTANCE] for y in y_values])
    else: 
        position = RandomEllipticalDistribution(samplePoints=20) * firstSurfaceSD

    direction = bd.tile(bd.array([ZERO, ZERO, ONE]), (numRays, 1))

    temp = bd.zeros(5)
    temp[0] = wavelength
    temp[1] = NORMAL_RADIANT    # Sagittal radiant
    temp[2] = NORMAL_RADIANT    # Tangential radiant
    temp[3] = INIT_PHASE_DIFF   # Phase difference 
    
    return RayBatch(
        bd.concatenate([position, direction, bd.tile(temp, (numRays, 1))], axis=1)
        )


# ==================================================================
""" ==================== Emit From Image Space ================= """
# ==================================================================


def main():
    pass 

if __name__ == "__main__":
    main()




