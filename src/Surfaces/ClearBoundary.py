

from enum import Enum
import matplotlib.pyplot as plt


from Material import Material
from Raytracing.RayBatch import RayBatch, GenerateBeam
from Raytracing.Reflection import Reflect, LambertianReflect
from Raytracing.Refraction import Refract
from Raytracing.Polarization import SenkrechtUndParallel, PolarizeRB, ResidueRB, FresnelReflectance, QuantitativePolarize
from Util.Backend import backend as bd 
from Util.Backend import constant
from Util.MathFunctions import NewtonSolver
from Util.Misc import ArrayNormalized, TransversalDistance
from Util.Globals import OBJ_FACING, Axis, RNG, PBR
from Util.SpatialEllipse import SpatialEllipse, SpatialCircle
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, SetUnifScale, RemoveBG, AddXYZ, DrawEllipse, DrawClearBoundary





class ClearBoundary():
    """
    A clear boundary is the cylinder shaped surface that connects 2 lens surfaces. 
    """

    def __init__(self, E1, E2, mat=PBR.GLASS):

        """2 rings of class SpatialEllipse that defines this clear boundary"""
        self.E1 = E1 
        self.E2 = E2


        """Type of material for the bounding surface. Fow those within the lens element, it would naturally be glass. For clear boundaries outside of the lens element, such as between 2 groups, it can be set to metal or plastic."""
        self.materialType = mat


        """Describes the material at the other side. When set to None, treat it as Air; it can also be set to a constant float that represents IOR; alternatively it could be a material class. Note that if self.materialType is set to metal, this will not be used at all due to metal's preservation of the incidence."""
        self.exteriorCoating = Material()


        """Weight of [0, 1] that controls total diffuse and total mirror reflection. When set to 0, surface reflects as lambertian, when set to 1, reflects like mirror"""
        self.specularReflection = 0.5
        # This should be altered very carefully, too high of a change may result in reflected rays going beyond the surface

        self.absorption = 0.5


    def DrawSurface(self):
        DrawClearBoundary(self.E1, self.E2)
        

    def isCylindrical(self):
        """
        When the 2 ends are both on aixs circle with the same radius, this clear bounardy becomes a cylinder, which could simplify calculation significantly. 
        """
        return self.E1.SemiAxisMagnititude() == self.E2.SemiAxisMagnititude()


    def Intersection(self, incidentRaybatch):
        """
        Calculate the intersection points of all the rays in a raybatch, if any. 

        :return: intersection points (n_, 3) and a boolean mask (n, ) indicating the rays that did intersect. 
        """

        origin = incidentRaybatch.Position() 
        direction = incidentRaybatch.Direction()
        
        # Extract the parameters for easier access 
        E1Z = self.E1.ZCoord()
        E2Z = self.E2.ZCoord()
        E1R = self.E1.SemiAxisMagnititude()
        E2R = self.E2.SemiAxisMagnititude()
        
        if(E1Z == E2Z):
            # When both circle have same Z, they are on the same plane, use the plane intersection.
            # This method includes the outer raidus check  
            points, _bool = self._PlaneIntersection(origin, direction, self.E1, self.E2)

        elif(E1R == E2R):
            # When both circle have same radius, they are a cylinder
            # print("From cylinder")
            points, _bool = self._RayCylinderIntersection(origin, direction, 
                                                  self.E1.ZCoord(), 
                                                  self.E2.ZCoord(),
                                                  self.E1.SemiAxisMagnititude())
        else:
            # If both check failed, then it's a circular frustum
            # print("from circular frustum")
            points, _bool = self._RayCircularFrustumIntersect(origin, direction,
                                                  self.E1.ZCoord(), 
                                                  self.E2.ZCoord(), 
                                                  self.E1.SemiAxisMagnititude(), 
                                                  self.E2.SemiAxisMagnititude())


        return points, _bool


    def Normal(self, intersections):
        """
        Given intersection points, calculate the normal vectors at those points, assuming the points are on the surface. 
        """

        E1Z = self.E1.ZCoord()
        E2Z = self.E2.ZCoord()
        E1R = self.E1.SemiAxisMagnititude()
        E2R = self.E2.SemiAxisMagnititude()


        if (E1Z == E2Z):
            # When the CB is in the plane perpendicular to the z axis.
            return bd.tile(OBJ_FACING, (intersections.shape[0], 1))
        elif (E1R == E2R):
            # When the CB is a cylinder
            pointingDir = ArrayNormalized(self.E1.center - intersections)
            pointingDir[:, Axis.Z.value] = 0
            return pointingDir
        else:
            if(E1Z > E2Z):
                E1Z, E2Z = E2Z, E1Z
            return self._ComputeFrustumNormals(intersections, E1R, E2R,  E2Z, E1Z)

        
    def Trace(self, incidentRaybatch, previousRI, inverted=False):
        """
        A clear boundary still calculates refraction, but it is only for Fresnel reflectance and not for ray propagation.

        :return: reflected raybatch and a boolean mask indicating the rays that did intersect.
        """

        intersections, _mask = self.Intersection(incidentRaybatch)

        if(intersections.shape[0] != _mask.sum()):
            print("True in sum ", _mask.sum())
            print("Accident")

        # Reflected RB is created here first with the mask applied, this means it should not contain any rays that are not intersecting with the surface. 
        reflectedRB = RayBatch(bd.copy(incidentRaybatch.value[_mask]))
        reflectedRB.SetPosition(intersections)

        # Becasue reflected 
        directions = reflectedRB.Direction()

        # Calculate the normal direction
        normals = self.Normal(intersections)
        # Accquire a desired normal vector direction as they should be pointing against the incident rays 
        desiredDirection = -bd.sign(reflectedRB.Direction()[:, 2])
        # Only flip the normals if the clear boundary is not a cylinder 
        if (self.E1.SemiAxisMagnititude() != self.E2.SemiAxisMagnititude()):
            normals[desiredDirection != bd.sign(normals[:, 2])] *= -1

        # Calculate both the mirror reflection and a Lambertian outgoing direction.
        # Unlike the previous implementation, this samples the diffuse lobe directly
        # instead of perturbing the normal and then performing a mirror reflection.
        mirrorReflected = Reflect(directions, normals)
        lambertReflected, lambertIntensity = LambertianReflect(normals, outputPer=1)

        # Interpolate between the two reflected directions.
        # specularReflection = 1 gives pure mirror reflection;
        # specularReflection = 0 gives pure Lambertian reflection.
        specularReflection = bd.clip(self.specularReflection, 0.0, 1.0)
        reflected = ArrayNormalized(
            mirrorReflected * specularReflection +
            lambertReflected * (1 - specularReflection)
        )
        reflectedRB.SetDirection(reflected)

        # Couple diffuse intensity to the sampled outgoing direction.
        # For the Lambertian branch, brighter contributions stay closer to the
        # surface normal; the specular branch preserves full strength.
        lambertCos = bd.sum(reflected * normals, axis=1)
        lambertCos = bd.clip(lambertCos, 0.0, 1.0)
        reflectionIntensity = (
            specularReflection +
            (1 - specularReflection) * lambertIntensity * lambertCos
        )
        reflectedRB.SetRadianceTerms(
            reflectedRB.RadianceTerms() * reflectionIntensity[:, None]
        )

        # Accquire the index of refractions (resp. wavelength)
        n1 = previousRI[_mask]
        n2 = self.exteriorCoating.RI(reflectedRB.Wavelength())
        # If the ray hits from the behind, RI needs to be swapped 
        if(inverted):
            n1, n2 = n2, n1 

        # Calculate the refraction for the polaried radiance ellipses 
        # Refracted rays themselves are not used since they no longer contribute to the imaging process. 
        refracted, TIR, _temp = Refract(directions, normals, n1, n2)

        # Accquire the reflectance ratio for the polarized radiance ellipses
        R_s, R_p = FresnelReflectance(normals[~TIR], directions[~TIR], refracted, n1[~TIR], n2[~TIR])
        # Accquire s and p directional vector 
        senkrecht, parallel = SenkrechtUndParallel(directions[~TIR], normals[~TIR])
        # nonTIRRB is a temporary RayBatch that only contains the non-TIR rays
        nonTIRRB = RayBatch(reflectedRB.value[~TIR])
        # Calculate the quantitative reflectance on the local s and p direction.
        senkrecht, parallel = QuantitativePolarize(
                nonTIRRB.PolarizationMat(),
                senkrecht[:, :2], 
                parallel[:, :2], 
                R_s, 
                R_p
            )
        # Modify the polarization ellipse of the nonTIR rays based on the quantitative reflectance just calculated. 
        nonTIRRB = ResidueRB(nonTIRRB, senkrecht, parallel)

        # The direction of the reflected, including TIR, are already set previously. 
        # So here only need to merge the nonTIR, whose raidance ellipse just got modified, with the TIR rays of the original raybatch.
        reflectedRB = nonTIRRB.Merge(RayBatch(reflectedRB.value[TIR]))

        absorption = min(max(self.absorption, 0.0), 1.0)
        reflectedRB.RandomDrop(absorption)

        return reflectedRB, _mask

        

    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _RandomInHemisphere(self, normals):
        """
        Generate random unit vectors that lie in the hemisphere defined by each normal vector.
        
        Parameters:
            normals (ndarray): Array of shape (N, 3) where each row is a unit normal vector.
        
        Returns:
            ndarray: Array of shape (N, 3) with random unit vectors in the hemisphere of the corresponding normal.
        """
        normals = bd.asarray(normals)
        N = normals.shape[0]
        
        # Generate N random vectors (not necessarily unit length) from a normal distribution
        v = RNG.randn(N, 3)
        
        # Normalize each vector
        v = v / bd.linalg.norm(v, axis=1, keepdims=True)
        
        # Compute dot products for each pair (row-wise)
        dots = bd.sum(v * normals, axis=1)
        
        # For each vector that is in the opposite hemisphere, flip its direction
        flip_mask = dots < 0
        v[flip_mask] = -v[flip_mask]
        
        return v


    def _ComputeFrustumNormals(self, points, r1, r2, z1, z2, epsilon=1e-6):
        """
        Compute normals for points on a circular frustum (GPU-accelerated).
        
        :param points: (N, 3) intersection points on the frustum. 
        :param r1
        :param r2: 
        :param z1:
        :param z2:
            
        :return: (N, 3) array of unit normals
        """

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r_actual = bd.sqrt(x**2 + y**2)
        normals = bd.zeros_like(points)
        
        # Bottom cap
        bottom_mask = bd.abs(z - z2) < epsilon
        bottom_mask &= (r_actual <= r1 + epsilon)
        normals[bottom_mask] = [0, 0, -1]
        
        # Top cap
        top_mask = bd.abs(z - z1) < epsilon
        top_mask &= (r_actual <= r2 + epsilon)
        normals[top_mask] = [0, 0, 1]
        
        # Lateral surface
        lateral_mask = ~(bottom_mask | top_mask)
        h = z1 - z2
        m = (r2 - r1) / h  # Slope of the lateral surface
        
        # Compute radius at each z-height
        z_rel = z[lateral_mask] - z2
        r_z = r1 + m * z_rel
        
        # Compute unnormalized normals (derived from surface gradient)
        x_lat, y_lat = x[lateral_mask], y[lateral_mask]
        N_unnorm = bd.column_stack([x_lat, y_lat, -m * r_z])
        
        # Normalize vectors
        norms = bd.linalg.norm(N_unnorm, axis=1, keepdims=True)
        normals[lateral_mask] = N_unnorm / norms
        
        return -normals


    def _RayFrustumIntersection(self, ray_origin, ray_dir, z1, z2, xc, yc, a1, b1, a2, b2):
        oz, dz = ray_origin[2], ray_dir[2]
        """
        Depreciated. 
        This is for truely elliptical frustum. While technically possible, this is a very rare case of clear boundary. This method is also incredibly costly due to the useage of Newton method. For these reasons, try not to ever invoke it. 
        """
        # What the fuck

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


    def _RayCircularFrustumIntersectionStrict(self, rayPos, rayDir, z1, z2, r1, r2, eps=1e-8):
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
    

    def _RayCircularFrustumIntersect(self, rayPos, rayDir, z1, z2, r1, r2):
        """
        Compute the intersection coordinates between rays and a conical frustum, assuming that every incident ray will intersect the frustum.
        
        :param rayPos: (N,3) array of ray origins.
        :param rayDir (ndarray): (N,3) array of ray directions.
        :param z1: (float) z-coordinates of the 1st circle.
        :param z2: (float) z-coordinates of the 2nd circle.
        :param r1: (float) Radius at z1.
        :param r2: (float) Radius at z2.
        
        :return: an intersections (ndarray): (N,3) array of intersection coordinates and intersects (ndarray): (N,) boolean array (all True).
        """

        # For each ray, compute the t-values where the ray crosses the z boundaries.
        oz = rayPos[:, 2]
        dz = rayDir[:, 2]
        
        # Compute t-values for z boundaries
        t_min = (z1 - oz) / dz
        t_max = (z2 - oz) / dz
        swap = t_min > t_max
        t_min[swap], t_max[swap] = t_max[swap], t_min[swap]
        
        k = (r2 - r1) / (z2 - z1)
        
        # Quadratic coefficients
        A = rayDir[:, 0]**2 + rayDir[:, 1]**2 - (k * rayDir[:, 2])**2
        B = 2 * (rayPos[:, 0] * rayDir[:, 0] + rayPos[:, 1] * rayDir[:, 1]) - 2 * k * rayDir[:, 2] * (r1 + k * (oz - z1))
        C = rayPos[:, 0]**2 + rayPos[:, 1]**2 - (r1 + k * (oz - z1))**2
        
        # Define masks
        epsilon = 1e-12
        linear_mask = bd.abs(A) < epsilon  # Mask for linear case (A ≈ 0)
        quad_mask = ~linear_mask            # Mask for quadratic case (A ≠ 0)
        
        # Initialize t_candidate with infinity
        t_candidate = bd.full(A.shape, bd.inf)
        
        # --- Solve for linear case (A ≈ 0) ---
        B_linear = B[linear_mask]
        C_linear = C[linear_mask]
        t_linear = -C_linear / B_linear  # Solve Bt + C = 0
        t_candidate[linear_mask] = t_linear
        
        # --- Solve for quadratic case (A ≠ 0) ---
        A_quad = A[quad_mask]
        B_quad = B[quad_mask]
        C_quad = C[quad_mask]
        
        disc = bd.sqrt(B_quad**2 - 4 * A_quad * C_quad)
        t1 = (-B_quad - disc) / (2 * A_quad)
        t2 = (-B_quad + disc) / (2 * A_quad)
        
        t1_valid = bd.where(t1 >= 0, t1, bd.inf)
        t2_valid = bd.where(t2 >= 0, t2, bd.inf)
        t_quad = bd.minimum(t1_valid, t2_valid)
        t_candidate[quad_mask] = t_quad
        
        # Compute intersection coordinates
        intersections = rayPos + t_candidate[:, None] * rayDir
        return intersections, bd.ones(rayPos.shape[0], dtype=bool)


    def _RayCylinderIntersection(self, rayPos, rayDir, z1, z2, r):
        """
        This method assumes the clear boundary is a cylinder on the longitudinal z axis direction with radius r.
        Each ray contributes at most one intersection along the positive ray direction.
        """
        ox, oy, oz = rayPos[:, 0], rayPos[:, 1], rayPos[:, 2]
        dx, dy, dz = rayDir[:, 0], rayDir[:, 1], rayDir[:, 2]

        zMin = bd.minimum(z1, z2)
        zMax = bd.maximum(z1, z2)
        eps = 1e-12

        A = dx ** 2 + dy ** 2
        B = 2 * (ox * dx + oy * dy)
        C = ox ** 2 + oy ** 2 - r ** 2

        discriminant = B ** 2 - 4 * A * C
        canHitSide = (A > eps) & (discriminant >= 0)

        sqrtD = bd.sqrt(bd.maximum(discriminant, 0))
        denom = 2 * bd.where(A > eps, A, 1)

        t1 = (-B - sqrtD) / denom
        t2 = (-B + sqrtD) / denom

        zAtT1 = oz + t1 * dz
        zAtT2 = oz + t2 * dz

        valid1 = canHitSide & (t1 >= 0) & (zAtT1 >= zMin) & (zAtT1 <= zMax)
        valid2 = canHitSide & (t2 >= 0) & (zAtT2 >= zMin) & (zAtT2 <= zMax)

        tCandidate = bd.where(valid1, t1, bd.inf)
        tCandidate = bd.minimum(tCandidate, bd.where(valid2, t2, bd.inf))

        valid = tCandidate != bd.inf
        intersections = rayPos + tCandidate[:, None] * rayDir

        return intersections[valid], valid


    def _PlaneIntersection(self, rayPos, rayDir, E1, E2, axis=OBJ_FACING):
        """
        This method is for when the clear boundary is in a plane perpendicular to the z axis.
        """

        frontVertex = E1.center if (E1.center[Axis.Z.value] < E2.center[Axis.Z.value]) else E2.center

        E1R = E1.SemiAxisMagnititude()
        E2R = E2.SemiAxisMagnititude()
        outerRadius = E1R if (E1R > E2R) else E2R

        # innerRadius = E1R if (E1R < E2R) else E2R

        denom = bd.dot(rayDir, axis)
        
        # Avoid division by zero for parallel vectors
        mainMask = bd.isclose(denom, 0)
        
        # Compute t for each vector
        t = bd.dot((frontVertex - rayPos), axis) / denom
        t[mainMask] = bd.nan  # Assign NaN for parallel vectors
        
        # Compute the intersection points
        intersections = rayPos + t[:, bd.newaxis] * rayDir

        # Calculate the bool mask for valid intersections within the clear bounday. 
        # Note that the inner raidus should already be used during propagation, no ray should be arriving at the inner radius region. 
        fsMask = (TransversalDistance(intersections) < outerRadius).reshape(-1)

        mainMask[~mainMask] = fsMask

        return intersections[fsMask], mainMask
    



def main():

    placeholderM = Material("BK1")

    temp = bd.array([0, 0, 0])

    testRB = GenerateBeam(bd.array([0, 0, 0]), bd.array([0, 2.5, 1]), size=4)
    testRB.Merge(GenerateBeam(bd.array([0, 19.5, 0]), bd.array([0, -0.4, 1]), size=4))
    testRB.Merge(GenerateBeam(bd.array([0, 0, 0]), bd.array([0, 1, 1]), size=4))
    testRB.Merge(GenerateBeam(bd.array([0, 0, 0]), bd.array([0, 0.55, 1]), size=4))
    testRB.Merge(GenerateBeam(bd.array([0, 0, 0]), bd.array([0, 0.25, 1]), size=4))

    E0 = SpatialCircle(0,   20)
    E1 = SpatialCircle(10,  20)
    E2 = SpatialCircle(20,  10)
    E3 = SpatialCircle(20,  3)

    # testCBLL = ClearBoundary(E0, E1) # cylinder 
    testCBL = ClearBoundary(E1, E2)  # Frustum 
    # testCBT = ClearBoundary(E2, E3)  # Plane 

    RI = placeholderM.RI(testRB.Wavelength())
    # pointsLL = testCBLL.Trace(testRB, RI)
    pointsL = testCBL.Trace(testRB, RI)
    # pointsT = testCBT.Trace(testRB, RI)

    # testCBLL.DrawSurface()
    testCBL.DrawSurface()
    # testCBT.DrawSurface()

    DrawRaybatch(testRB, lLength=20)

    
    SetUnifScale(50)
    #AddXYZ()
    #RemoveBG()
    plt.show()


if __name__ == "__main__":
    main()
