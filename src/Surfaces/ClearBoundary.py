

from enum import Enum
import matplotlib.pyplot as plt


from Util.Backend import backend as bd 
from Util.Backend import constant
from Util.MathFunctions import NewtonSolver
from Util.Misc import ArrayNormalized
from Util.SpatialEllipse import SpatialEllipse
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, SetUnifScale, RemoveBG, AddXYZ, DrawEllipse, DrawClearBoundary


class RayBehavior(Enum):
    """ While it's called Refractive, it simply means the surface behaves like a normal glass, with refraction and reflection both at play. """
    Refrative = 0

    """ Those marked as Kill will remove all the rays that intersect with them. """
    Kill = 1



class ClearBoundary():
    """
    A clear boundary is the cylinder shaped surface that connects 2 lens surfaces. 
    """

    def __init__(self, E1, E2):

        """2 rings of class SpatialEllipse that defines this clear boundary"""
        self.E1 = E1 
        self.E2 = E2


        """When the 2 ends are both on aixs circle with the same radius, this clear bounardy becomes a cylinder, which could simplify calculation significantly. """
        self.isCylindrical = False


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
        
        # points, _bool = self._RayCircularFrustumIntersection(tempO, tempD, 
        #                                           self.E1.ZCoord(), 
        #                                           self.E2.ZCoord(), 
        #                                           self.E1.SemiAxisMagnititude(), 
        #                                           self.E2.SemiAxisMagnititude())
        
        points, _bool = self._RayCylinderIntersection(tempO, tempD, 
                                                  self.E1.ZCoord(), 
                                                  self.E2.ZCoord(),
                                                  self.E1.SemiAxisMagnititude())
        
        print(points, "\n", _bool)

        DrawPoints(points)
        


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _RayFrustumIntersection(self, ray_origin, ray_dir, z1, z2, xc, yc, a1, b1, a2, b2):
        oz, dz = ray_origin[2], ray_dir[2]
        """
        Depreciated. 
        This is for truely elliptical frustum. While technically possible, this is a very rare case of clear boundary. This method is also incredibly costly due to the useage of Newton method. For these reasons, try not to ever invoke it. 
        """
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


    def _RayCircularFrustumIntersection(self, rayPos, rayDir, z1, z2, r1, r2, eps=1e-8):
        """
        Compute the intersection coordinates between rays and a conical frustum.
        
        :param rayPos: (N,3) array of ray origins.
        :param rayDir (ndarray): (N,3) array of ray directions.
        :param z1: (float) z-coordinates of the 1st circle.
        :param z2: (float) z-coordinates of the 2nd circle.
        :param r1: (float) Radii at z1.
        :param r2: (float) Radii at z2.
        :param eps: Tolerance for degenerate cases (e.g. |A| < eps).
        
        :return: an intersections (ndarray): (N,3) array of intersection coordinates and intersects (ndarray): (N,) boolean array.
        """
        N = rayPos.shape[0]

        # Compute t-values where the ray crosses z=z1 and z=z2.
        oz = rayPos[:, 2]
        dz = rayDir[:, 2]
        t_range_min = bd.where(bd.abs(dz) > eps, (z1 - oz) / dz, -bd.inf)
        t_range_max = bd.where(bd.abs(dz) > eps, (z2 - oz) / dz, bd.inf)
        # Swap if needed so that t_range_min <= t_range_max.
        swap_mask = t_range_min > t_range_max
        t_range_min[swap_mask], t_range_max[swap_mask] = t_range_max[swap_mask], t_range_min[swap_mask]
        
        # Compute the frustum slope.
        k = (r2 - r1) / (z2 - z1)
        # Let R(z) = r1 + k*(z - z1). Then the condition is:
        #    (x^2 + y^2) = (R(z))^2.
        # With z = oz + t * dz and (x,y) = (ox,oy) + t*(dx,dy), we get a quadratic:
        #   A t^2 + B t + C = 0,
        A = rayDir[:,0]**2 + rayDir[:,1]**2 - (k * rayDir[:,2])**2
        B = 2 * (rayPos[:,0] * rayDir[:,0] + rayPos[:,1] * rayDir[:,1]) \
            - 2 * k * rayDir[:,2] * (r1 + k*(oz - z1))
        C = rayPos[:,0]**2 + rayPos[:,1]**2 - (r1 + k*(oz - z1))**2

        # Initialize output arrays.
        intersections = bd.full((N, 3), bd.nan)
        intersects = bd.zeros(N, dtype=bool)
        
        # First, handle non-degenerate (quadratic) cases where |A| > eps.
        quad_mask = bd.abs(A) > eps
        if bd.any(quad_mask):
            A_quad = A[quad_mask]
            B_quad = B[quad_mask]
            C_quad = C[quad_mask]
            disc = B_quad**2 - 4*A_quad*C_quad
            
            # Only consider rays with a real solution.
            real_mask = disc >= 0
            # Compute candidate solutions where available.
            t1 = (-B_quad - bd.sqrt(disc)) / (2*A_quad)
            t2 = (-B_quad + bd.sqrt(disc)) / (2*A_quad)
            # For each ray, select the smallest t that is in [t_range_min, t_range_max] and non-negative.
            t_candidate = bd.where((t1 >= t_range_min[quad_mask]) & (t1 <= t_range_max[quad_mask]) & (t1 >= 0),
                                t1, bd.inf)
            t_candidate = bd.minimum(t_candidate,
                                    bd.where((t2 >= t_range_min[quad_mask]) & (t2 <= t_range_max[quad_mask]) & (t2 >= 0),
                                            t2, bd.inf))
            valid_quad = t_candidate != bd.inf
            indices_quad = bd.where(quad_mask)[0]
            # Fill in valid intersections.
            intersections[indices_quad[valid_quad]] = rayPos[indices_quad[valid_quad]] + \
                t_candidate[valid_quad, None] * rayDir[indices_quad[valid_quad]]
            intersects[indices_quad[valid_quad]] = True

        # Next, handle degenerate (linear) cases where |A| <= eps.
        linear_mask = ~quad_mask
        if bd.any(linear_mask):
            # For these rays, the equation is B*t + C = 0 (if B != 0).
            # Avoid division by zero:
            valid_linear = bd.abs(B[linear_mask]) > eps
            t_linear = bd.full(int(bd.sum(linear_mask)), bd.inf)
            t_linear[valid_linear] = -C[linear_mask][valid_linear] / B[linear_mask][valid_linear]
            # Accept only if within the valid t-range and non-negative.
            valid_lin = (t_linear >= t_range_min[linear_mask]) & (t_linear <= t_range_max[linear_mask]) & (t_linear >= 0)
            indices_linear = bd.where(linear_mask)[0]
            intersections[indices_linear[valid_lin]] = rayPos[indices_linear[valid_lin]] + \
                t_linear[valid_lin, None] * rayDir[indices_linear[valid_lin]]
            intersects[indices_linear[valid_lin]] = True

        # Additionally, force rays that don't intersect the z-range to be marked as no intersection.
        intersects &= (t_range_max >= 0)
        
        return intersections, intersects
    

    def _RayCircularFrustumIntersecCert(self, rayPos, rayDir, z1, z2, r1, r2):
        """
        Compute the intersection coordinates between rays and a conical frustum, assuming that every incident ray will intersect the frustum.
        
        :param rayPos: (N,3) array of ray origins.
        :param rayDir (ndarray): (N,3) array of ray directions.
        :param z1: (float) z-coordinates of the 1st circle.
        :param z2: (float) z-coordinates of the 2nd circle.
        :param r1: (float) Radii at z1.
        :param r2: (float) Radii at z2.
        
        :return: an intersections (ndarray): (N,3) array of intersection coordinates and intersects (ndarray): (N,) boolean array (all True).
        """

        # For each ray, compute the t-values where the ray crosses the z boundaries.
        oz = rayPos[:, 2]
        dz = rayDir[:, 2]
        # Compute t-values at z=z1 and z=z2.
        t_min = (z1 - oz) / dz
        t_max = (z2 - oz) / dz
        # Ensure t_min is the smaller value.
        swap = t_min > t_max
        t_min[swap], t_max[swap] = t_max[swap], t_min[swap]
        
        # Slope of the frustum's lateral surface.
        k = (r2 - r1) / (z2 - z1)
        
        # Formulate the quadratic equation for the ray's XY projection:
        #   A t^2 + B t + C = 0,
        A = rayDir[:, 0]**2 + rayDir[:, 1]**2 - (k * rayDir[:, 2])**2
        B = 2 * (rayPos[:, 0] * rayDir[:, 0] + rayPos[:, 1] * rayDir[:, 1]) - \
            2 * k * rayDir[:, 2] * (r1 + k * (oz - z1))
        C = rayPos[:, 0]**2 + rayPos[:, 1]**2 - (r1 + k * (oz - z1))**2
        
        # Compute the discriminant (assumed nonnegative)
        disc = bd.sqrt(B**2 - 4 * A * C)
        
        # Compute the two solutions t1 and t2.
        t1 = (-B - disc) / (2 * A)
        t2 = (-B + disc) / (2 * A)
        
        # Select the smallest positive t-value.
        t_candidate = bd.minimum(bd.where(t1 >= 0, t1, bd.inf),
                                bd.where(t2 >= 0, t2, bd.inf))
        
        # Compute intersection coordinates.
        intersections = rayPos + t_candidate[:, None] * rayDir
        
        # Since every ray is supposed to intersect, return an all-True boolean mask.
        return intersections, bd.ones(rayPos.shape[0], dtype=bool)


    def _RayCylinderIntersection(self, rayPos, rayDir, z1, z2, r):
        """
        This method assumes the clear boundary is a cylinder on the z axis with radius r. 
        """
        # Component decomposition
        ox, oy, oz = rayPos[:, 0], rayPos[:, 1], rayPos[:, 2]
        dx, dy, dz = rayDir[:, 0], rayDir[:, 1], rayDir[:, 2]
        
        # Initialize empty array for coordinates
        all_coords = bd.empty((0, 3), dtype=bd.float32)

        # Process non-vertical rays ------------------------------------------
        non_vert_mask = dz != 0
        if bd.any(non_vert_mask):
            # Batch compute quadratic solutions
            A = dx[non_vert_mask]**2 + dy[non_vert_mask]**2
            B = 2 * (ox[non_vert_mask]*dx[non_vert_mask] + oy[non_vert_mask]*dy[non_vert_mask])
            C = ox[non_vert_mask]**2 + oy[non_vert_mask]**2 - r**2
            
            discriminant = B**2 - 4*A*C
            intersectMask = discriminant >= 0
            
            sqrt_d = bd.sqrt(discriminant[intersectMask])
            t1 = (-B[intersectMask] - sqrt_d) / (2*A[intersectMask])
            t2 = (-B[intersectMask] + sqrt_d) / (2*A[intersectMask])

            # Calculate coordinates for both solutions
            coords_t1 = bd.stack([
                ox[non_vert_mask][intersectMask] + t1*dx[non_vert_mask][intersectMask],
                oy[non_vert_mask][intersectMask] + t1*dy[non_vert_mask][intersectMask],
                oz[non_vert_mask][intersectMask] + t1*dz[non_vert_mask][intersectMask]
            ], axis=1)

            coords_t2 = bd.stack([
                ox[non_vert_mask][intersectMask] + t2*dx[non_vert_mask][intersectMask],
                oy[non_vert_mask][intersectMask] + t2*dy[non_vert_mask][intersectMask],
                oz[non_vert_mask][intersectMask] + t2*dz[non_vert_mask][intersectMask]
            ], axis=1)

            # Filter valid z-ranges
            valid_z = (coords_t1[:, 2] >= z1) & (coords_t1[:, 2] <= z2) & (t1 >= 0)
            coords_t1 = coords_t1[valid_z]
            
            valid_z = (coords_t2[:, 2] >= z1) & (coords_t2[:, 2] <= z2) & (t2 >= 0)
            coords_t2 = coords_t2[valid_z]

            # Concatenate valid coordinates
            all_coords = bd.concatenate([all_coords, coords_t1, coords_t2], axis=0)

        # Process vertical rays ----------------------------------------------
        vert_mask = (dz == 0)
        if bd.any(vert_mask):
            # Find vertical rays inside cylinder
            in_z = (oz[vert_mask] >= z1) & (oz[vert_mask] <= z2)
            in_radius = (ox[vert_mask]**2 + oy[vert_mask]**2) <= r**2
            valid_vert = in_z & in_radius

            # Generate coordinates for valid vertical rays
            vert_coords = bd.stack([
                ox[vert_mask][valid_vert],
                oy[vert_mask][valid_vert],
                oz[vert_mask][valid_vert]
            ], axis=1)

            # Duplicate coordinates for both endpoints
            vert_coords = bd.repeat(vert_coords, 2, axis=0)
            vert_coords[1::2, 2] = z2  # Second point at upper Z
            
            all_coords = bd.concatenate([all_coords, vert_coords], axis=0)

        return all_coords, intersectMask
        


def main():

    E1 = SpatialEllipse(bd.array([0, 0, 5]), 
                        bd.array([1, 0, 0]),
                        bd.array([0, 1, 0]), 
                        15, 15)
    
    E2 = SpatialEllipse(bd.array([0, 0, 5]), 
                        bd.array([1, 0, 0]), 
                        bd.array([0, 1, 0]), 
                        30, 30)

    testCB = ClearBoundary(E1, E2)

    testCB.DrawSurface()
    #testCB.Trace(None)

    
    SetUnifScale(50)
    #AddXYZ()
    #RemoveBG()
    plt.show()


if __name__ == "__main__":
    main()