

from Util.Backend import backend as bd 
from Util.Misc import ArrayNormalized



def Reflect(incident, normal):
    """
    Calculates the reflected vectors given incident vectors and normal vectors. Note that this method is for mirror reflection. 

    :param incident: Array of incident vectors (shape: Nx3).
    :param normal: Array of normal vectors (shape: Nx3).

    :return: directions of reflected rays (shape: Nx3). 
    """
    # Make sure both are normalized 
    incident = ArrayNormalized(incident)
    normal = ArrayNormalized(normal)
    
    # Compute dot product of incident vectors with surface normals
    dotProduct = bd.sum(incident * normal, axis=1, keepdims=True)
    
    # Apply reflection formula: R = I - 2 * (I ⋅ N) * N
    reflected = incident - 2 * dotProduct * normal
    
    return reflected


def LambertianReflect(normal, outputCount=1):
    """
    Pure Lambertian reflection evenly reflcts the ray back and no incident infomation is needed. 

    :param normal: normal direction at the point where this reflection happens (shape: Nx3). 
    :param outputCount: number of output directions per normal. It is suggested to keep this as 1, otherwise the calculation and memory useage could increase dramatically. 

    :return: directions of reflected rays. 
    """
    pass 
