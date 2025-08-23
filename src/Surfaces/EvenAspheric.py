
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

        self.proxyEnvelopeThickness = .0

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
        """
        Build two equal-radius spherical proxies that tightly enclose the even asphere
        over r ∈ [0, clearSemiDiameter]. The two proxies share the *same signed radius*;
        they differ only by a vertex-z offset. Offsets are relative to the asphere vertex.
        """
        C = float(self.clearSemiDiameter)

        # Degenerate case: no aperture → flat proxies at the asphere vertex.
        if C <= 0:
            self.boundingSurfaceF = Surface(INFINITY, 0, self.clearSemiDiameter)
            self.boundingSurfaceF.SetCumulative(self.cumulativeThickness)
            self.boundingSurfaceB = Surface(INFINITY, 0, self.clearSemiDiameter)
            self.boundingSurfaceB.SetCumulative(self.cumulativeThickness)
            return

        # Find equal-radius envelope (GPU-friendly, deterministic)
        env = self.FindTightEqualRadiusEnvelope(
            C=self.clearSemiDiameter,
            nr=2048,  # radial samples (increase for tighter fit)
            nR=128,  # radius grid samples (log-spaced)
            margin=0.0  # optional safety inflation of the envelope
        )

        Rs = env['radius']  # shared signed radius
        z_v_lower = env['lower']['vertex_z']  # offset relative to asphere vertex
        z_v_upper = env['upper']['vertex_z']

        # Front (“lower”) proxy sphere
        self.boundingSurfaceF = Surface(Rs, 0, self.clearSemiDiameter)
        # Place the proxy vertex at the asphere’s cumulative vertex + computed offset
        self.boundingSurfaceF.SetCumulative(self.cumulativeThickness + z_v_lower)

        # Back (“upper”) proxy sphere
        self.boundingSurfaceB = Surface(Rs, 0, self.clearSemiDiameter)
        self.boundingSurfaceB.SetCumulative(self.cumulativeThickness + z_v_upper)

        # (Optional) keep the envelope thickness for diagnostics
        self.proxyEnvelopeThickness = env['thickness']

        print(self.GetInfo())


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


    def GetInfo(self, showBounding=True):

        info = "Aspheric surface " +\
            "\nRadius: " + str(self.radius) +\
            "\nK: " + str(self.K) +\
            "\nAsph coefficients:" +\
            "".join(f"\n  {(i+1)*2}: {v}" for i, v in enumerate(self.asphCoef))

        if showBounding:
            if self.boundingSurfaceF is not None:
                info += ("\nBounding surface front: " +
                         "\n  Radius: " + str(self.boundingSurfaceF.radius) +
                         "\n  Thickness: " + str(self.boundingSurfaceF.thickness) +
                         "\n  Semi diameter: " + str(self.boundingSurfaceF.clearSemiDiameter))
            if self.boundingSurfaceB is not None:
                info += ("\nBounding surface back: " +
                         "\n  Radius: " + str(self.boundingSurfaceB.radius) +
                         "\n  Thickness: " + str(self.boundingSurfaceB.thickness) +
                         "\n  Semi diameter: " + str(self.boundingSurfaceB.clearSemiDiameter))

        return info


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _sag_asphere_vec(self, r):
        """Vectorized even-asphere sag f(r) with R= self.radius, k=self.K, A=self.asphCoef."""
        r2 = r ** 2
        # base (conic)
        if bd.isinf(self.radius):
            base = bd.zeros_like(r2)
        else:
            sqrt_term = bd.sqrt(1 - (1 + self.K) * r2 / self.radius ** 2)
            base = r2 / (self.radius * (1 + sqrt_term))
        # even asphere terms starting at r^2 (A2, A4, ...)
        asph = bd.zeros_like(r2)
        for i, a in enumerate(self.asphCoef):
            asph += a * r2 ** (i + 1)
        return base + asph


    def _sphere_shape_g(self, r, Rs_abs, sign_rs):
        """g_R(r) = R - sign(R)*sqrt(R^2 - r^2) with |R|=Rs_abs and sign=sign_rs (±1)."""
        # ensure domain: Rs_abs >= max(r)
        return sign_rs * (Rs_abs - bd.sqrt(Rs_abs ** 2 - r ** 2))


    def _pick_R_range(self, C):
        """
        Sensible radius bracket [Rlo, Rhi] (absolute value).
        - Must satisfy R >= C (domain of sqrt).
        - Use local curvature at r=0 if available as a guide.
        """
        C = float(C)
        # curvature estimate from r=0 (if A2 present): kappa0 = d2z/dr2|0 = 2*A2
        Rs_hint = None
        if len(self.asphCoef) > 0:
            A2 = float(self.asphCoef[0])
            if A2 != 0.0:
                Rs_hint = abs(1.0 / (2.0 * A2))  # sphere matching vertex curvature
        if (not bd.isinf(self.radius)):
            R_base = abs(float(self.radius))
            Rs_hint = R_base if Rs_hint is None else min(Rs_hint, R_base)

        if Rs_hint is None:  # fallback
            Rs_hint = 10.0 * C

        Rlo = max(1.001 * C, 0.1 * Rs_hint)
        Rhi = max(Rlo * 1.1, 10.0 * Rs_hint)
        return Rlo, Rhi


    def FindTightEqualRadiusEnvelope(self, C, nr=2048, nR=128, margin=0.0):
        """
        Find two enclosing spherical surfaces with the SAME radius (signed),
        that minimize the max gap thickness over r∈[0,C].
        Returns:
            {
              'radius': signed_Rs,                      # shared radius (float)
              'lower': {'radius': signed_Rs, 'vertex_z': z_v_lower},
              'upper': {'radius': signed_Rs, 'vertex_z': z_v_upper},
              'thickness': Tmin,
              'sign': +1 or -1
            }
        Notes:
          - 'vertex_z' are offsets relative to the asphere vertex (z=0 at r=0).
          - To place these in world coords, add your cumulative thickness.
          - margin>0 can inflate the envelope to add safety (e.g., margin=1e-9).
        """
        # radial samples
        r = bd.linspace(0.0, float(C), int(nr))
        f = self._sag_asphere_vec(r)

        # candidate |R| range
        Rlo, Rhi = self._pick_R_range(C)

        # search |R| on a log grid (deterministic, GPU-friendly)
        Rs_grid = bd.exp(bd.linspace(bd.log(Rlo), bd.log(Rhi), int(nR)))

        best = None  # (Tmin, Rs_abs, sign, z_v_lower, z_v_upper)

        for sign_rs in (+1, -1):
            # Broadcast r:(nr,1), Rs_grid:(1,nR) -> g:(nr,nR)
            rr = r[:, None]
            GG = sign_rs * (Rs_grid[None, :] - bd.sqrt(Rs_grid[None, :] ** 2 - rr ** 2))

            H = f[:, None] - GG  # (nr,nR)
            hmax = H.max(axis=0)  # (nR,)
            hmin = H.min(axis=0)  # (nR,)
            T = hmax - hmin  # thickness per candidate radius

            j = int(bd.argmin(T)) if backend_name == "cupy" else int(T.argmin())
            Tmin = float(T[j])
            Rs_best = float(Rs_grid[j])
            z_v_lower = float(hmin[j] - margin)
            z_v_upper = float(hmax[j] + margin)

            cand = (Tmin, Rs_best, sign_rs, z_v_lower, z_v_upper)
            if (best is None) or (Tmin < best[0]):
                best = cand

        Tmin, Rs_abs, sign_rs, zlo, zhi = best
        signed_Rs = sign_rs * Rs_abs

        return {
            'radius': signed_Rs,
            'lower': {'radius': signed_Rs, 'vertex_z': zlo},
            'upper': {'radius': signed_Rs, 'vertex_z': zhi},
            'thickness': Tmin,
            'sign': sign_rs
        }





