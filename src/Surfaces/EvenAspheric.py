
from scipy.optimize import minimize_scalar


from .Surface import * 
from Util.Backend import backend as bd
from Util.Backend import constant
from Util.PltPlot import DrawAspherical, DrawAsphericalProfile
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR



class EvenAspheric(Surface):
    """
    Even aspheric surface.
    """ 
    def __init__(self, r, t, sd, K, A):
        super().__init__(r, t, sd)

        self.K = K

        """Aspherical coefficients"""
        self.asphCoef = A

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


    def PreComputeProxy(selfself):
        pass


    def Intersection(self, incidentRaybatch):
        """
        Calculates the intersection of ray and the aspheric surface.
        """

        pass


    def DrawSurface(self, drawSag=True):
        DrawAspherical(
            radius=self.radius,
            k=self.K,
            A=[1e-5, -2e-7],
            clearSemiDiameter=15.0,
            cumulativeThickness=0,
            opacity=0.2)

        if (drawSag):
            DrawAsphericalProfile(
                radius=50.0,
                k=-1.0,
                A=[1e-5, -2e-7],
                clearSemiDiameter=15.0,
                cumulativeThickness=0,
                axis="x",  # or "y"
                lineWidth=1.5
            )


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================



    def _AsphericSag(self, r, R, k, A):
        """Compute sag of even aspheric surface at radius r."""
        r2 = r ** 2
        sqrt_term = bd.sqrt(1 - (1 + k) * r2 / R ** 2)
        base = r2 / (R * (1 + sqrt_term))
        asphere = sum(A[i] * r2 ** (i + 2) for i in range(len(A)))
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
        res_min = minimize_scalar(lambda r: self._AsphericSag(r, R, k, A), bounds=(0, C), method='bounded')
        res_max = minimize_scalar(lambda r: -self._AsphericSag(r, R, k, A), bounds=(0, C), method='bounded')
        return {
            'r_min': res_min.x,
            'z_min': res_min.fun,
            'dzdr_min': self._AsphericSagDerivative(res_min.x, R, k, A),
            'r_max': res_max.x,
            'z_max': -res_max.fun,
            'dzdr_max': self._AsphericSagDerivative(res_max.x, R, k, A)
        }


    def _BestFitSphereThroughPointAndSlope(self, r, z, dzdr):
        """Compute radius and vertex height of sphere tangent to point (r, z) with slope dz/dr."""
        theta = bd.arctan(dzdr)
        sin_theta = bd.sin(theta)
        cos_theta = bd.cos(theta)

        # Center coordinates in polar direction
        cx = r - sin_theta * bd.inf
        cz = z + cos_theta * bd.inf

        # Use geometric formula for radius and vertex height
        R_sphere = (1 + dzdr ** 2) ** 1.5 / 1e-6  # Use a small curvature value for approximation
        direction = bd.sign(dzdr) if dzdr != 0 else 1
        R_sphere = direction * abs(R_sphere)
        z_vertex = z - R_sphere * bd.sqrt(1 / (1 + dzdr ** 2))
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





