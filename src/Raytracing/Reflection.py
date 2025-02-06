

from Util.Backend import backend as bd 
from Util.Misc import ArrayNormalized



def Reflect(incident, normal):
    """
    Calculates the reflected vectors given incident vectors and normal vectors. Note that this method is for mirror reflection. 

    :param incident: Array of incident vectors (shape: Nx3).
    :param normal: Array of normal vectors (shape: Nx3).
    """
    # Make sure both are normalized 
    incident = ArrayNormalized(incident)
    normal = ArrayNormalized(normal)
    
    # Compute dot product of incident vectors with surface normals
    dotProduct = bd.sum(incident * normal, axis=1, keepdims=True)
    
    # Apply reflection formula: R = I - 2 * (I ⋅ N) * N
    reflected = incident - 2 * dotProduct * normal
    
    return reflected


def Lambertian(incident, normal, outputCount=1):
    pass 
