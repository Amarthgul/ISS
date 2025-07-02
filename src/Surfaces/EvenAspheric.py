
from scipy.optimize import minimize_scalar


from .Surface import *
from Util.Backend import backend as bd
from Util.Backend import constant, backend_name
from Util.PltPlot import DrawAspherical, DrawAsphericalProfile
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR



class EvenAspheric(Surface):
    """
    Even aspheric surface.
    """ 
    def __init__(self, r, t, sd, K, A):
        super().__init__(r, t, sd)

        self.K = bd.array(K)

        """Aspherical coefficients. Start from 2nd order, then 4th, then 6th, etc."""
        self.asphCoef = bd.array(A)

        self.boundingSurfaceF = None

        self.boundingSurfaceB = None

        self.cType = CurvatureType.EvenAspheric


    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the vertices. Foe even aspheric, this also computes the bounding proxy geometry.
        """
        cumulativeT = bd.array(cumulativeT)

        # The local optical axis remains the same as OBJ FACING
        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT + self.radius])
        self._radiusDirection = self.frontVertex - self.radiusCenter

        if(self.radius == INFINITY):
            # When r=inf, the cumulative thickness at the edge is the same as the cumulative thickness of the vertex.
            self.sdCumulative = cumulativeT
        else:
            self.sdCumulative = cumulativeT + self.radius + bd.sqrt(self.radius**TWO - self.clearSemiDiameter**TWO) * bd.sign(-self.radius)

        self.PreComputeProxy()


    def PreComputeProxy(self):
        var = self._FindEnvelopingSpheres(self.radius, self.K, self.asphCoef, self.clearSemiDiameter)

        self.boundingSurfaceF = Surface(var["sphere_at_min"]["radius"], 0, self.clearSemiDiameter)
        tempOffset = var["sphere_at_min"]["radius"] + var["sphere_at_min"]["vertex_z"]
        print("Offset for min: ", tempOffset)
        self.boundingSurfaceF.SetCumulative(self.cumulativeThickness + tempOffset)

        self.boundingSurfaceB = Surface(var["sphere_at_max"]["radius"], 0, self.clearSemiDiameter)
        tempOffset = var["sphere_at_max"]["radius"] + var["sphere_at_max"]["vertex_z"]
        print("Offset for max: ", tempOffset)
        self.boundingSurfaceB.SetCumulative(self.cumulativeThickness + tempOffset)

        print(var)


    def Intersection(self, incidentRaybatch):
        """
        Calculates the intersection of ray and the aspheric surface.
        """

        pass


    def DrawSurface(self, drawSag=True, drawProxy=False):
        DrawAspherical(
            radius=self.radius,
            k=self.K,
            A=self.asphCoef,
            clearSemiDiameter=self.clearSemiDiameter,
            cumulativeThickness=self.cumulativeThickness,
            surfaceColor=SURFACE_COLOR)

        if (drawSag):
            DrawAsphericalProfile(
                radius=self.radius,
                k=self.K,
                A=self.asphCoef,
                clearSemiDiameter=self.clearSemiDiameter,
                cumulativeThickness=self.cumulativeThickness)

        if (drawProxy):
            self.boundingSurfaceF.DrawSurface()
            self.boundingSurfaceB.DrawSurface()



    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _AsphericSag(self, r, R, k, A):
        """Compute sag of even aspheric surface at radius r.
        Supports infinite radius (flat base) and starts from the 2nd order aspheric term.
        """
        r2 = r ** 2

        if bd.isinf(R):
            base = bd.zeros_like(r2)
        else:
            sqrt_term = bd.sqrt(1 - (1 + k) * r2 / R ** 2)
            base = r2 / (R * (1 + sqrt_term))

        asphere = sum(A[i] * r2 ** (i + 1) for i in range(len(A)))  # starts from r^2
        return base + asphere


    def _AsphericSagDerivative(self, r, R, k, A):
        """Compute dz/dr for the aspheric surface."""
        r2 = r ** 2
        sqrt_term = bd.sqrt(1 - (1 + k) * r2 / R ** 2)
        denom = (1 + sqrt_term) * sqrt_term
        d_base = (2 * r / R) * (1 / (1 + sqrt_term) + (1 + k) * r2 / (R ** 2 * denom))
        d_asphere = sum((2 * (i + 2)) * A[i] * r ** (2 * i + 3) for i in range(len(A)))
        return d_base + d_asphere


    def _FindExtremePoints(self, R, k, A, C):
        """Find r that gives min and max sag within [0, C]."""

        if (backend_name == "cupy"):
            # The next several lines will need Scipy, however,
            # Scipy does not go well with cupy, actually numpy too
            R = R.item()
            k = bd.asnumpy(k).item()
            A = bd.asnumpy(A).tolist()
            C = bd.asnumpy(C).item()

        res_min = minimize_scalar(lambda r: float(self._AsphericSag(r, R, k, A)), bounds=(0, C), method='bounded')
        res_max = minimize_scalar(lambda r: float(-self._AsphericSag(r, R, k, A)), bounds=(0, C), method='bounded')
        return {
            'r_min': res_min.x,
            'z_min': res_min.fun,
            'dzdr_min': self._AsphericSagDerivative(res_min.x, R, k, A),
            'r_max': res_max.x,
            'z_max': -res_max.fun,
            'dzdr_max': self._AsphericSagDerivative(res_max.x, R, k, A)
        }


    def _AsphericSagSecondDerivative(self, r, R, k, A):
        """Compute d²z/dr² for curvature calculation."""
        r2 = r ** 2
        sqrt_term = bd.sqrt(1 - (1 + k) * r2 / R ** 2)
        denom = (1 + sqrt_term) * sqrt_term
        d2_base = (2 / R) * (1 / (1 + sqrt_term) +
                             3 * (1 + k) * r2 / (R ** 2 * denom ** 2) +
                             (1 + k) / (R ** 2 * denom))

        d2_asphere = sum((2 * (i + 2)) * (2 * i + 3) * A[i] * r ** (2 * i + 2) for i in range(len(A)))
        return d2_base + d2_asphere


    def _BestFitSphereThroughPointAndSlope(self, r, z, dzdr):
        d2zdr2 = self._AsphericSagSecondDerivative(r, self.radius, self.K, self.asphCoef)
        curvature = d2zdr2
        if curvature == 0:
            return bd.inf, z  # Flat
        R_sphere = (1 + dzdr ** 2) ** 1.5 / curvature
        z_vertex = z - R_sphere / bd.sqrt(1 + dzdr ** 2)
        return R_sphere, z_vertex


    def _FindEnvelopingSpheres(self, R, k, A, C):
        """Find two tangent spheres enveloping the asphere."""
        extremes = self._FindExtremePoints(R, k, A, C)

        R1, zv1 = self._BestFitSphereThroughPointAndSlope(extremes['r_min'], extremes['z_min'], extremes['dzdr_min'])
        R2, zv2 = self._BestFitSphereThroughPointAndSlope(extremes['r_max'], extremes['z_max'], extremes['dzdr_max'])

        return {
            'min_point': (extremes['r_min'], extremes['z_min'], extremes['dzdr_min']),
            'max_point': (extremes['r_max'], extremes['z_max'], extremes['dzdr_max']),
            'sphere_at_min': {'radius': R1, 'vertex_z': zv1},
            'sphere_at_max': {'radius': R2, 'vertex_z': zv2}
        }





