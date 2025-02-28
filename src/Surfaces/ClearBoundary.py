

import matplotlib.pyplot as plt

from Util.Backend import backend as bd 
from Util.Backend import constant
from Util.MathFunctions import NewtonSolver
from Util.Misc import ArrayNormalized
from Util.SpatialEllipse import SpatialEllipse
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, SetUnifScale, RemoveBG, AddXYZ, DrawEllipse, DrawClearBoundary



class ClearBoundary():
    """
    A clear boundary is the cylinder shaped surface that connects 2 lens surfaces. 
    """

    def __init__(self, E1, E2):

        """2 rings of class SpatialEllipse that defines this clear boundary"""
        self.E1 = E1 
        self.E2 = E2


        """Describes the material at the other side. When set to None, treat it as Air; it can also be set to a constant float that represents IOR; alternatively it could be a material class."""
        self.exteriorCoating = None 


        """Weight of [0, 1] that controls total diffuse and total mirror reflection. When set to 0, surface reflects as lambertian, when set to 1, reflects like mirror"""
        self.specularReflection = 0.5


    def DrawSurface(self):
        DrawClearBoundary(self.E1, self.E2)
        


    def Intersection(self, incomingRaybatch):
        pass 


    def Normal(self, intersections):
        pass 


    def Trace(self, incidentRaybatch):
        tempO = bd.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        tempD = ArrayNormalized(bd.array([[0, 1, 1], [0, 1, 0.75], [0, 2, 0.8]]))
        DrawDirection(tempO, tempD, lineLength=25)

        print("Datas: ", "\nE1 z ", self.E1.ZCoord(), 
            "\nE2 z ", self.E2.ZCoord(), 
            "\nE1 r ", self.E1.SemiAxisMagnititude(), 
            "\nE2 r ", self.E2.SemiAxisMagnititude())
        
        points, _bool = self._RayCircularFrustumIntersection(tempO, tempD, 
                                                  self.E1.ZCoord(), 
                                                  self.E2.ZCoord(), 
                                                  self.E1.SemiAxisMagnititude(), 
                                                  self.E2.SemiAxisMagnititude())
        
        DrawPoints(points)
        print(points, "\n", _bool)


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _RayFrustumIntersection(self, ray_origin, ray_dir, z1, z2, xc, yc, a1, b1, a2, b2):
        oz, dz = ray_origin[2], ray_dir[2]

        from scipy.optimize import newton

        # Check if ray intersects the z-range [z1, z2]
        if dz == 0:
            if oz < z1 or oz > z2:
                return False, []
            t_in = t_out = 0.0  # Ray is horizontal within [z1, z2]
        else:
            t_in = (z1 - oz) / dz
            t_out = (z2 - oz) / dz
            if t_in > t_out:
                t_in, t_out = t_out, t_in
            if t_out < 0:
                return False, []
            t_in = max(t_in, 0)
        
        # Define the quartic equation to solve
        def equation(t):
            z = oz + t * dz
            s = (z - z1) / (z2 - z1)
            a = a1 + s * (a2 - a1)
            b = b1 + s * (b2 - b1)
            x = ray_origin[0] + t * ray_dir[0] - xc
            y = ray_origin[1] + t * ray_dir[1] - yc
            return (x/a)**2 + (y/b)**2 - 1
        
        # Solve using Newton-Raphson within [t_in, t_out]
        try:
            # This is what makes this method less desirable, the newton method is fairly expensive and will slow down the process dramatically, consider using 
            t_sol = newton(equation, x0=(t_in + t_out)/2, xmin=t_in, xmax=t_out)
            return True, [t_sol]
        except RuntimeError:
            return False, []

    def _RayCircularFrustumIntersection(self, ray_origins, ray_dirs, z1, z2, r1, r2):
        """
        Check if multiple rays intersect a conical frustum defined by two circles on the z-axis.

        Parameters:
            ray_origins (bd.ndarray): (N, 3) array of ray origins.
            ray_dirs (bd.ndarray): (N, 3) array of ray directions.
            z1, z2 (float): z-coordinates of the two circles.
            r1, r2 (float): Radii of the two circles.

        Returns:
            bd.ndarray: (N,) boolean array indicating intersection.
            list: List of arrays containing t-values for each ray.
        """
        # Extract z-components
        oz = ray_origins[:, 2]
        dz = ray_dirs[:, 2]

        # Compute z-boundary intersection parameters t_in and t_out
        t_in = bd.where(dz != 0, (z1 - oz) / dz, bd.inf)
        t_out = bd.where(dz != 0, (z2 - oz) / dz, bd.inf)

        # Swap t_in and t_out where needed
        swap_mask = t_in > t_out
        t_in[swap_mask], t_out[swap_mask] = t_out[swap_mask], t_in[swap_mask]

        # Define slope of the frustum surface
        k = (r2 - r1) / (z2 - z1)

        # Compute quadratic coefficients for ray-frustum intersection
        A = ray_dirs[:, 0]**2 + ray_dirs[:, 1]**2 - (k * ray_dirs[:, 2])**2
        B = 2 * (ray_origins[:, 0] * ray_dirs[:, 0] + ray_origins[:, 1] * ray_dirs[:, 1]) - \
            2 * k * ray_dirs[:, 2] * (r1 + k * (oz - z1))
        C = ray_origins[:, 0]**2 + ray_origins[:, 1]**2 - (r1 + k * (oz - z1))**2

        # Compute the discriminant
        discriminant = B**2 - 4 * A * C
        valid_mask = discriminant >= 0

        # Allocate space for intersection points (initialize with NaN)
        intersections = bd.full_like(ray_origins, bd.nan)

        # Process valid intersections
        if bd.any(valid_mask):
            sqrt_discriminant = bd.sqrt(discriminant[valid_mask])

            # Compute the two possible intersection t-values
            t1 = (-B[valid_mask] - sqrt_discriminant) / (2 * A[valid_mask])
            t2 = (-B[valid_mask] + sqrt_discriminant) / (2 * A[valid_mask])

            # Select the first valid intersection in the z-range
            valid_t = bd.where((t_in[valid_mask] <= t1) & (t1 <= t_out[valid_mask]), t1, 
                            bd.where((t_in[valid_mask] <= t2) & (t2 <= t_out[valid_mask]), t2, bd.nan))

            # Compute intersection coordinates for valid rays
            intersection_coords = ray_origins[valid_mask] + valid_t[:, None] * ray_dirs[valid_mask]

            # Store valid intersections in the output array
            intersections[valid_mask] = intersection_coords

        return intersections, valid_mask
            
    




def main():

    E1 = SpatialEllipse(bd.array([0, 0, 5]), 
                        bd.array([1, 0, 0]),
                        bd.array([0, 1, 0]), 
                        20, 20)
    
    E2 = SpatialEllipse(bd.array([0, 0, 15]), 
                        bd.array([1, 0, 0]), 
                        bd.array([0, 1, 0]), 
                        10, 10)

    testCB = ClearBoundary(E1, E2)

    testCB.DrawSurface()
    testCB.Trace(None)

    
    SetUnifScale(50)
    #AddXYZ()
    #RemoveBG()
    plt.show()


if __name__ == "__main__":
    main()