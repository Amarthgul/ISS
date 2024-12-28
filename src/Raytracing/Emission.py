




"""
This file handles the initial generation of rays. 

There are two method of doing so. One is to project the rays toward the first surface, and the second is to project them towards the entrence pupil. 

Projecting to first surface is more faithful as it is less discriminative, and is better when glare and flares are needed. But this method is slower. 

Projecting to the entrance pupil is faster, but it will offer less flare and glares. 

"""


from Util.Backend import backend as bd
from Util.Misc import Normalized, ArrayNormalized, CircularDistribution, angleBetweenVectors, Rotation, Translate
from Util.Globals import NORMAL_RADIANT, INIT_PHASE_DIFF

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

    one = 1.0
    zero = 0

    # On axis scenario 
    if (bd.isclose(posP[0], 0) and bd.isclose(posP[1], 0)):
        P_xy_projection = bd.array([sd, 0, 0])
        onAxis = True 
    else:
        # Util vectors 
        P_xy_projection = posP.copy()
        P_xy_projection[2] = 0

    vecCA = posA - posC
    
    # On axis rays can be grealty simplified 
    if(onAxis):
        if (useDistribution):
            # Move the point along the z axis 
            points = bd.transpose(CircularDistribution()) + offset
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
            print("temp: ", temp)
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
        trans_1 = Rotation(-theta_1, bd.array([0, 0, 1]), points)
        
        # Move the points to be in tangent with A 
        trans_1 = Translate(trans_1, P_xy_projection * (sd - a)) 
        
        # Rotate the ellipse around A it fits into the AB plane 
        theta = angleBetweenVectors(posB-posA, posC-posA)

        temp = bd.zeros(3) # Another temporary array for cupy compatibility
        temp[0] = -vecCA[1]
        temp[1] = vecCA[0]
        axis = Normalized(temp)

        trans_2 = Translate(trans_1, -posA)
        trans_2 = Rotation(-theta, axis, trans_2)
        trans_2 = Translate(trans_2, posA)

        return trans_2


def InitRays(r, sd, posP, wavelength = 550):

    P_xy_projection = posP.copy()
    P_xy_projection[2] = 0

    offset = bd.zeros(3)
    offset[2] = abs(r) - bd.sqrt(r**2 - sd**2)
    offset *= bd.sign(r)

    if(not bd.isclose(bd.linalg.norm(P_xy_projection), 0)):
        posA = sd * ( P_xy_projection / bd.linalg.norm(P_xy_projection) ) + offset
        posC = sd * (-P_xy_projection / bd.linalg.norm(P_xy_projection) ) + offset
    else:
        posA = bd.array([sd, 0, 0]) + offset
        posC = bd.array([sd, 0, 0]) + offset
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


def main():
    pass 

if __name__ == "__main__":
    main()




