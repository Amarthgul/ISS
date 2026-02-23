


from Util.Backend import backend as bd 
from Util.Globals import ONE, TWO


def NewtonSolver(f, df, x0, tol=1e-6, max_iter=100):
    """
    A simple Newton-Raphson solver using Cupy.
    
    Parameters:
        f: function accepting a Cupy array and returning a Cupy array.
        df: derivative of f, also operating on Cupy arrays.
        x0: initial guess (Cupy array).
        tol: tolerance for convergence.
        max_iter: maximum number of iterations.
        
    Returns:
        x: the computed root (Cupy array).
    """
    
    x = x0.copy()
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        # Avoid division by zero by adding a small epsilon if needed.
        x_new = x - fx / (dfx + 1e-12)
        if bd.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x



def Erf(x):
    """
    erf approximation (Abramowitz-Stegun 7.1.26), bd-friendly (NumPy/CuPy)
    """
    # constants
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = bd.sign(x)
    ax = bd.abs(x)
    t = ONE / (ONE + p * ax)

    # Horner polynomial
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    y = ONE - poly * bd.exp(-ax * ax)

    return sign * y


def Phi(x):
    # Standard normal CDF: 0.5*(1 + erf(x/sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    return 0.5 * (ONE + Erf(x * inv_sqrt2))


def SkewNormPDF(x, mu, sigma, alpha):
    # Azzalini skew-normal PDF:
    # f(x) = (2/σ) * φ(z) * Φ(α z), z=(x-μ)/σ
    z = (x - bd.array(mu)) / bd.array(sigma)
    phi = (ONE / bd.sqrt(TWO * bd.pi)) * bd.exp(-0.5 * z * z)
    p = Phi(bd.array(alpha) * z)
    return (TWO / bd.array(sigma)) * phi * p


