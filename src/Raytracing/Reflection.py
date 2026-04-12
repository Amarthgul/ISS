

from Util.Backend import backend as bd 
from Util.Misc import ArrayNormalized
from Util.PltPlot import DrawDirection


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
    
    # print(incident[bd.sum(incident * normal, axis=1) > 0] )

    # Compute dot product of incident vectors with surface normals
    dotProduct = bd.sum(incident * normal, axis=1, keepdims=True)
    
    # Apply reflection formula: R = I - 2 * (I ⋅ N) * N
    reflected = incident - 2 * dotProduct * normal

    return reflected


def LambertianReflect(normal, outputPer=1):
    """
    Pure Lambertian reflection evenly reflects the ray back and no incident information is needed.

    :param normal: normal direction at the point where this reflection happens (shape: Nx3). 
    :param outputPer: number of output directions per normal. It is suggested to keep this as 1, otherwise the calculation and memory usage could increase dramatically.

    :return: an array for directions of reflected rays, another array for their intensity.
    """

    normal = bd.asarray(normal)
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise ValueError("normal must have shape (N, 3)")
    if outputPer < 1:
        raise ValueError("outputPer must be >= 1")

    normal = ArrayNormalized(normal)
    count = normal.shape[0]
    total = count * outputPer

    # Duplicate normals when more than one output is required per input normal.
    if outputPer != 1:
        normalRep = bd.repeat(normal, outputPer, axis=0)
    else:
        normalRep = normal

    dtype = normalRep.dtype

    # Cosine-weighted hemisphere sampling in local coordinates.
    # This is a natural Monte Carlo match for Lambertian reflection, so every
    # returned sample can carry the same intensity weight.
    u1 = bd.random.random(total)
    u2 = bd.random.random(total)

    r = bd.sqrt(u1)
    phi = 2.0 * bd.pi * u2

    xLocal = r * bd.cos(phi)
    yLocal = r * bd.sin(phi)
    zLocal = bd.sqrt(bd.maximum(0.0, 1.0 - u1))

    # Build a robust orthonormal basis around each normal.
    # Pick a helper axis that is not too parallel to the normal.
    useZ = bd.abs(normalRep[:, 2]) < 0.999
    helper = bd.zeros_like(normalRep)
    helper[:, 2] = 1.0
    helper[~useZ, 1] = 1.0
    helper[~useZ, 2] = 0.0

    tangent = bd.cross(helper, normalRep)
    tangentNorm = bd.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / bd.maximum(tangentNorm, bd.asarray(1e-12, dtype=dtype))

    bitangent = bd.cross(normalRep, tangent)
    bitangentNorm = bd.linalg.norm(bitangent, axis=1, keepdims=True)
    bitangent = bitangent / bd.maximum(bitangentNorm, bd.asarray(1e-12, dtype=dtype))

    reflected = (
        tangent * xLocal[:, None]
        + bitangent * yLocal[:, None]
        + normalRep * zLocal[:, None]
    )
    reflected = ArrayNormalized(reflected)

    # For cosine-weighted Lambertian sampling, each sample carries equal weight.
    # Split the total reflected energy evenly when multiple outputs are emitted
    # from the same input normal.
    intensity = bd.ones(total, dtype=dtype) / outputPer

    return reflected, intensity
