


from Util.Backend import backend as bd
from Util.Globals import ONE, NEAR_ZERO
from Util.Misc import ArrayNormalized, Magnitude
from Raytracing.RayBatch import RayBatch



def ModifyEllipse(A, v, add=False):
    """
    Modify an ellipse to expand or contract in the direction of vector `v`.
    
    :param A: 2x2 positive definite matrix defining the original ellipse (ndarray).
    :param v: 2D vector specifying direction and magnitude (ndarray). 
    :param add: when set to True, ellipse will be expanded, otherwise contracted. 

    :return: bd.ndarray: New 2x2 matrix defining the modified ellipse
    """

    # Number of ellipses / vectors
    n = v.shape[0]

    # Compute direction angles (theta) for each vector in v.
    theta = bd.arctan2(v[:, 1], v[:, 0])  # shape (n,)
    c = bd.cos(theta)
    s = bd.sin(theta)
    
    # Construct rotation matrices R for each sample, shape (n, 2, 2)
    R = bd.stack([bd.stack([c, -s], axis=1),
                  bd.stack([s,  c], axis=1)], axis=1)
    
    # Compute vector magnitudes m for each vector v.
    m = bd.linalg.norm(v, axis=1)  # shape (n,)
    
    # Compute the quadratic form v^T * A * v for each sample.
    # This yields a 1D array of length n.
    vAv = bd.einsum('ni,nij,nj->n', v, A, v)
    
    # Compute scaling factor s_factor for the modification.
    if add:
        # For expansion, s_factor = sqrt(v^T A v)
        s_factor = bd.sqrt(vAv)
    else:
        # For contraction, compute the original length L in direction v:
        L = m / bd.sqrt(vAv)
        # New length L_new is reduced by m:
        L_new = L - m
        s_factor = L_new / L
    
    # Build a diagonal scaling matrix S for each sample.
    # S[i] = diag(s_factor[i], 1), shape (n, 2, 2)
    S = bd.tile(bd.eye(2), (n, 1, 1))
    S[:, 0, 0] = s_factor

    # Compute the inverse of S.
    S_inv = bd.tile(bd.eye(2), (n, 1, 1))
    S_inv[:, 0, 0] = 1.0 / s_factor

    # Compute the combined transformation matrix: M_inv = R * S_inv * R^T.
    R_T = bd.transpose(R, (0, 2, 1))
    M_inv = bd.matmul(bd.matmul(R, S_inv), R_T)
    
    # Finally, compute the new ellipse matrix:
    # A_new = M_inv^T * A * M_inv for each sample.
    M_inv_T = bd.transpose(M_inv, (0, 2, 1))
    A_new = bd.matmul(bd.matmul(M_inv_T, A), M_inv)
    
    return A_new


def SenkrechtUndParallel(incident, normal):
    """
    Berechnen Sie die p und s Polarisationsrichtung bei der gegebenen Einfalls- und Normalrichtung.
    """
    # Oh nein es gibt ein deutsche Verfahren! 
    
    senkrecht = ArrayNormalized(bd.cross(incident, normal))

    # Manuelles Zuweisen des senkrechten Vektors
    senkrecht[bd.all(bd.isnan(senkrecht), axis=1)] = bd.array([1, 0, 0])

    # Note that if incident is the reverse of normal, this means a perpendicular ray. Which will not have polarization effects, only partial reflection based on the transmission. For these rays, the senkrecht calculation above will return Nan, so will the the parallel.

    return senkrecht, ArrayNormalized(bd.cross(normal, senkrecht))


def FresnelReflectance(normals, incident, refracted, n1, n2):
    """
    
    """

    cosThetaI = bd.sum(normals * incident, axis=1) / (Magnitude(normals) * Magnitude(incident))
    cosThetaT = bd.sum(normals * refracted, axis=1) / (Magnitude(normals) * Magnitude(refracted))

    # Reflectance ratio along senkrecht and parallel direction (Fresnel equation)
    R_s = bd.abs( (n1*cosThetaI-n2*cosThetaI)/(n1*cosThetaI+n2*cosThetaT) ) ** 2
    R_p = bd.abs( (n1*cosThetaT-n2*cosThetaI)/(n1*cosThetaT+n2*cosThetaI) ) ** 2

    return R_s, R_p


def EllipseHeights(A, u, v):
    """
    Compute heights for two perpendicular vectors simultaneously.
    
    :param A: (n, 2, 2) ellipse matrices
    :param u: (n, 2) first vectors
    :param v: (n, 2) second vectors (⊥ to u)
        
    :return: (h_u, h_v) heights in u and v directions
    """

    # Combined computation
    M = bd.stack([u, v], axis=-1)  # Shape: (..., 2, 2)
    quad_terms = bd.einsum('...ki,...kj,...ij->...k', M, M, A)

    norms = bd.linalg.norm(M, axis=-2)  # Shape: (..., 2)
    h_u = norms[..., 0] / bd.sqrt(quad_terms[..., 0])
    h_v = norms[..., 1] / bd.sqrt(quad_terms[..., 1])
    
    return h_u, h_v


def QuantitativePolarize(A, s, p, R_s, R_p):
    """
    Given the incident polarized radiance ellipse, the local s and p direction, and the corresponding s and p direction reflectance ratio, calculate the quantitative reflectance on the local s and p direction. 
    """

    # s and p should have already been normalized since SenkrechtUndParallel() contains a normalization process 
    # s = ArrayNormalized(s)
    # p = ArrayNormalized(p)

    baseHeightS, baseHeightP = EllipseHeights(A, s, p)

    return s * (baseHeightS * R_s)[:, bd.newaxis], p * (baseHeightP * R_p)[:, bd.newaxis]



def PolarizeRB(rb, v_s, v_p, add=False):
    """
    Given a raybatch, modify it on senkrecht and parallel direction. 

    :return: modified raybatch. 
    """

    ellipseM = rb.PolarizationMat()

    ellipseM = ModifyEllipse(ellipseM, v_s, add)
    ellipseM = ModifyEllipse(ellipseM, v_p, add)

    rb.SetPolarization(ellipseM)

    return rb


def CreateEllipseFromFectors(u, v):
    """
    Creates ellipse matrices for multiple pairs of perpendicular vectors.
    
    :param u: (N, 2) array of 2D vectors
    :param v: (N, 2) array of 2D vectors (each pair must be perpendicular)
        
    :return: bd.ndarray: (N, 2, 2) array of ellipse matrices
    """

    batch_size = u.shape[0]
    
    # Compute magnitudes
    norms_u = bd.linalg.norm(u, axis=1, keepdims=True)
    norms_v = bd.linalg.norm(v, axis=1, keepdims=True)
    
    # Handle zero vectors gracefully
    norms_u = bd.where(norms_u == 0, 1, norms_u)  # Prevent division by zero
    norms_v = bd.where(norms_v == 0, 1, norms_v)
    
    # Create orthonormal basis
    e1 = u / norms_u
    e2 = v / norms_v
    
    # Construct rotation matrices (batch_size, 2, 2)
    R = bd.stack([e1, e2], axis=2)
    
    # Create scaling matrices (batch_size, 2, 2)
    D = bd.zeros((batch_size, 2, 2))
    D[:, 0, 0] = bd.where(norms_u.ravel() == 0, 0, 1/(norms_u**2).ravel())
    D[:, 1, 1] = bd.where(norms_v.ravel() == 0, 0, 1/(norms_v**2).ravel())
    
    # Compute transformed matrices using Einstein summation
    A = bd.einsum('nij,njk,nlk->nil', R, D, R)
    
    return A


def ResidueRB(rb, v_s, v_p):
    """
    Given a raybatch as the base, create an raybatch whose polarization component is based on the given senkrecht and parallel component. 
    """

    ms = CreateEllipseFromFectors(v_s, v_p)

    rb.SetPolarization(ms)

    return rb


def main():
    inc = ArrayNormalized(bd.array([[0.1, 0.1, 0.9], [0, 0, 1]]))
    nor = ArrayNormalized(bd.array([[0, 0, -1], [0, 0, -1]]))

    #print(SenkrechtUndParallel(inc, nor))

    A = bd.array([[[2, 0.5],
                   [0.5, 1.0]],
                  [[1.5, 0.2],
                   [0.2, 1.2]]])  # shape (2, 2, 2)
    
    A = bd.array([[[1e-9, 0],
                   [0, 1e-9]],
                  [[1e-9, 0],
                   [0, 1e-9]]])  # shape (2, 2, 2)
    
    v = bd.array([[1.0, 0.5],
                [-0.3, 0.8]])  # shape (2, 2)

    # Expand the ellipses in the direction of v.
    A_modified = ModifyEllipse(A, v, add=True)
    print("Modified ellipses:\n", A_modified.get())


if __name__ == "__main__":
    main()