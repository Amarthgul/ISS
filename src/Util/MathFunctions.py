


from Util.Backend import backend as bd 



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