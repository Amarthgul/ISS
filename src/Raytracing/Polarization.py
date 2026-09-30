"""Real coherency transport; terms are (Cxx, Cxy, Cyy) in a ray-local frame.

Directions and normals are unit vectors. Indices are positive and real.
Circular polarization and phase retardance (including TIR) are omitted.
Operations accept empty batches and use the configured NumPy/CuPy backend.
"""

from Util.Backend import backend as bd


def TransverseBasis(direction):
    """Deterministic right-handed (x, y, direction) frame, with no stored axes."""
    reference = bd.eye(3, dtype=direction.dtype)[bd.argmin(bd.abs(direction), axis=1)]
    x = reference - bd.sum(reference * direction, axis=1)[:, None] * direction
    x /= bd.linalg.norm(x, axis=1)[:, None]
    return x, bd.cross(direction, x)


def TransformCoherency(terms, j00, j01, j10, j11):
    """Apply a real Jones map J C J.T directly to the three coefficients."""
    a, b, c = terms.T
    return bd.stack((
        j00*j00*a + 2*j00*j01*b + j01*j01*c,
        j00*j10*a + (j00*j11 + j01*j10)*b + j01*j11*c,
        j10*j10*a + 2*j10*j11*b + j11*j11*c,
    ), axis=1)


def SenkrechtUndParallel(incident, normal):
    """Incident s/p frame; at normal incidence choose the ray's x axis as s."""
    s = bd.cross(incident, normal)
    length = bd.linalg.norm(s, axis=1)
    axial = length <= 1e-12
    s /= bd.where(axial, 1.0, length)[:, None]
    fallback, _ = TransverseBasis(incident)
    s = bd.where(axial[:, None], fallback, s)
    return s, bd.cross(incident, s)


def FresnelAmplitudes(incident, normal, n_in, n_out):
    """Signed r_s/r_p and power-normalized t_s/t_p.

    Local p = cross(direction, s) on both sides. TIR uses r_s=r_p=1,
    neglecting differential phase while preserving reflected power.
    """
    ci = bd.clip(bd.abs(bd.sum(incident * normal, axis=1)), 0.0, 1.0)
    discriminant = 1.0 - (n_in / n_out)**2 * (1.0 - ci*ci)
    ct = bd.sqrt(bd.maximum(discriminant, 0.0))
    ds = n_in*ci + n_out*ct
    dp = n_out*ci + n_in*ct
    rs = (n_in*ci - n_out*ct) / bd.where(ds == 0, 1.0, ds)
    rp = (n_out*ci - n_in*ct) / bd.where(dp == 0, 1.0, dp)
    same_medium = n_in == n_out
    rs = bd.where(same_medium, 0.0, rs)
    rp = bd.where(same_medium, 0.0, rp)
    tir = discriminant < 0.0
    rs = bd.where(tir, 1.0, rs)
    rp = bd.where(tir, 1.0, rp)
    return rs, rp, bd.sqrt(bd.maximum(1.0-rs*rs, 0.0)), bd.sqrt(bd.maximum(1.0-rp*rp, 0.0))


def FresnelReflectance(normals, incident, refracted, n1, n2):
    """Power reflectances with n1 incident and n2 outgoing; legacy signature."""
    rs, rp, _, _ = FresnelAmplitudes(incident, normals, n1, n2)
    return rs*rs, rp*rp


def InterfaceCoherency(terms, incident, normal, outgoing, amplitude_s, amplitude_p):
    """Map canonical incident coefficients through a specular interface."""
    s, pi = SenkrechtUndParallel(incident, normal)
    po = bd.cross(outgoing, s)
    xi, yi = TransverseBasis(incident)
    xo, yo = TransverseBasis(outgoing)
    # J = B_out.T [s, p_out] diag(amplitudes) [s, p_in].T B_in.
    si_x = bd.sum(s*xi, axis=1)
    si_y = bd.sum(s*yi, axis=1)
    pi_x = bd.sum(pi*xi, axis=1)
    pi_y = bd.sum(pi*yi, axis=1)
    so_x = amplitude_s * bd.sum(s*xo, axis=1)
    so_y = amplitude_s * bd.sum(s*yo, axis=1)
    po_x = amplitude_p * bd.sum(po*xo, axis=1)
    po_y = amplitude_p * bd.sum(po*yo, axis=1)
    return TransformCoherency(terms,
        so_x*si_x + po_x*pi_x, so_x*si_y + po_x*pi_y,
        so_y*si_x + po_y*pi_x, so_y*si_y + po_y*pi_y)


def DielectricCoherency(terms, incident, normal, outgoing, n_in, n_out, reflection=False):
    """One reflected or transmitted branch, always using the incident state."""
    rs, rp, ts, tp = FresnelAmplitudes(incident, normal, n_in, n_out)
    if reflection:
        return InterfaceCoherency(terms, incident, normal, outgoing, rs, rp)
    return InterfaceCoherency(terms, incident, normal, outgoing, ts, tp)


def TransportCoherency(terms, incident, outgoing):
    """Minimal-rotation transport for prescribed direction changes.

    This is the nondepolarizing approximation used by haze/diffuse samplers.
    Exact reversal rotates by pi about the incident canonical x axis.
    """
    xi, _ = TransverseBasis(incident)
    v = bd.cross(incident, outgoing)
    cosine = bd.clip(bd.sum(incident*outgoing, axis=1), -1.0, 1.0)
    reverse = cosine <= -1.0 + 1e-12
    denominator = bd.where(reverse, 1.0, 1.0+cosine)[:, None]
    xr = xi + bd.cross(v, xi) + bd.cross(v, bd.cross(v, xi)) / denominator
    xr = bd.where(reverse[:, None], xi, xr)
    xr -= bd.sum(xr*outgoing, axis=1)[:, None] * outgoing
    xr /= bd.linalg.norm(xr, axis=1)[:, None]
    yr = bd.cross(outgoing, xr)
    xo, yo = TransverseBasis(outgoing)
    return TransformCoherency(terms,
        bd.sum(xo*xr, axis=1), bd.sum(xo*yr, axis=1),
        bd.sum(yo*xr, axis=1), bd.sum(yo*yr, axis=1))


def RotateCoherency(terms, incident, outgoing, rotation):
    """Rotate the ray and electric-field axes by a rigid 3D rotation."""
    xi, yi = TransverseBasis(incident)
    xo, yo = TransverseBasis(outgoing)
    xr, yr = xi @ rotation.T, yi @ rotation.T
    return TransformCoherency(terms,
        bd.sum(xo*xr, axis=1), bd.sum(xo*yr, axis=1),
        bd.sum(yo*xr, axis=1), bd.sum(yo*yr, axis=1))
