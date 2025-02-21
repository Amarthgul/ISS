


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


def ResidueRB(rb, v_s, v_p):
    """
    Given a raybatch as the base, create an raybatch whose polarization component is based on the given senkrecht and parallel component. 
    """
    residue = RayBatch(bd.copy(rb.value))
    
    len = residue.value.shape[0]
    residue.SetPolarizationPerTerm(
        diag1 = bd.ones(len) * NEAR_ZERO, 
        diag2 = bd.ones(len) * NEAR_ZERO, 
        tilt  = bd.zeros(len)
    )
    residue = PolarizeRB(residue, v_s, v_p, add=True)

    return residue


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