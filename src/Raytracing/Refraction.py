



from Util.Backend import backend as bd 



def Refract(incident, normal, n1, n2):
    """
    Calculates the refracted vectors given incident vectors, normal vectors, and the refractive indices. Since all rays arriving should land on the surface, there should not be any vignetted rays. 
    
    :param incident: Array of incident vectors (shape: Nx3).
    :param normal: Array of normal vectors (shape: Nx3).
    :param n1: Refractive index of the first medium.
    :param n2: Refractive index of the second medium.

    :return: Array of refracted vectors (shape: Nx3), index of TIR rays, a Null vignette array.
    """

    # Normalize incident and normal vectors
    incident = incident / bd.linalg.norm(incident, axis=1, keepdims=True)
    normal = normal / bd.linalg.norm(normal, axis=1, keepdims=True)

    # Compute the ratio of refractive indices
    n_ratio = n1 / n2
    
    # Dot product of incident vectors and normal vectors
    cos_theta_i = -bd.einsum('ij,ij->i', incident, normal)
    
    # Calculate the discriminant to check for total internal reflection
    discriminant = 1 - (n_ratio ** 2) * (1 - cos_theta_i ** 2)
    
    # Handle total internal reflection (discriminant < 0)
    TIR = discriminant < 0


    # Calculate the refracted vectors
    refracted = n_ratio[:, bd.newaxis][~TIR] * incident[~TIR] + (n_ratio[~TIR] * cos_theta_i[~TIR] - bd.sqrt(discriminant[~TIR]))[:, bd.newaxis] * normal[~TIR]
    

    # refracted = n_ratio[:, bd.newaxis] * incident[~TIR] + (n_ratio * cos_theta_i[~TIR] - bd.sqrt(discriminant[~TIR]))[:, bd.newaxis] * normal[~TIR]

    # No rays are supposed to be vignetted in this operation as they should all be on the surface accroding to the method contract. 
    return refracted, TIR, None 







