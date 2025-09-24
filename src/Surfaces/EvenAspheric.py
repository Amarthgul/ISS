
from scipy.optimize import minimize_scalar


from .Surface import *
from Util.Backend import backend as bd
from Util.Backend import constant, backend_name
from Util.PltPlot import DrawAspherical, DrawAsphericalProfile, DrawSphericalProfile
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR



class EvenAspheric(Surface):
    """
    Even aspheric surface.
    """ 
    def __init__(self, r, t, sd, m, K, A):
        super().__init__(r, t, sd, m)

        """The number for conic section"""
        self.K = bd.array(K)

        """Aspherical coefficients. Start from 2nd order, then 4th, then 6th, etc."""
        self.asphCoef = bd.array(A)

        """The bounding surface at the front"""
        self.boundingSurfaceF = None

        """The bounding surface at the back"""
        self.boundingSurfaceB = None

        """Thickness/distance between the two bounding surfaces"""
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
        env = self._FindTightEqualRadiusEnvelope(
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
        Vectorized intersection of a ray batch with the even asphere.
        Uses two precomputed proxy spheres as a bracket, then runs
        a fixed-iteration safeguarded solve (array-only).
        Returns:
            points_on_surface (M,3)
            dummy_bool (M,)   # kept for compatibility; all False
            vignette_mask (N,)# True where ray did NOT produce a valid intersection
        """
        
        o = incidentRaybatch.Position()  # (N,3)
        d = incidentRaybatch.Direction()  # (N,3)
        N = o.shape[0]
        C = self.clearSemiDiameter

        # --- proxy sphere centers (z only) and shared signed radius
        zc_lo, zc_hi = self._ProxyCentersZ()  # floats
        Rs = float(self.boundingSurfaceF.radius)  # signed

        # --- intersect with both proxies to form a bracket [t_lo, t_hi]
        t1, v1 = self._IntersectRaySphereCenterz(o, d, zc_lo, abs(Rs))
        t2, v2 = self._IntersectRaySphereCenterz(o, d, zc_hi, abs(Rs))

        # valid bracket begins where both proxies intersect and at least one t>0
        valid = v1 & v2
        # build bracket
        t_lo = bd.minimum(t1, t2)
        t_hi = bd.maximum(t1, t2)

        # Any NaN or degenerate brackets are invalid
        bad = bd.isnan(t_lo) | bd.isnan(t_hi) | (t_hi <= t_lo)
        valid &= ~bad

        # If no valids at all, quick out (mirror Surface API)
        if not valid.any():
            vignette = bd.ones(N, dtype=bd.bool_)
            return bd.zeros((0, 3)), bd.zeros(0, dtype=bd.bool_), vignette

        # --- define F(t) and F'(t) on the fly (world coords)
        def F(t):
            # z_ray(t) - z_asphere_world(r(t))
            x = o[:, 0] + t * d[:, 0]
            y = o[:, 1] + t * d[:, 1]
            z = o[:, 2] + t * d[:, 2]
            r2 = x * x + y * y
            f = self._SagAsphere_r2(r2)  # local sag
            zsurf = self.cumulativeThickness + f
            return z - zsurf

        def Fp(t):
            # d/dt [z_ray - z_asphere] = d_z - (dz/dr)*(dr/dt)
            x = o[:, 0] + t * d[:, 0]
            y = o[:, 1] + t * d[:, 1]
            r2 = x * x + y * y
            r = bd.sqrt(bd.maximum(r2, 0.0))
            dzdr = self._AsphereDzDr(r)
            numer = x * d[:, 0] + y * d[:, 1]
            dzdt_asph = bd.where(r > 0, dzdr * (numer / r), bd.zeros_like(r))
            return d[:, 2] - dzdt_asph

        # Evaluate at bracket ends
        F_lo = F(t_lo)
        F_hi = F(t_hi)

        # If same sign, move the worse side to midpoint (deterministic, mask-safe)
        same = (F_lo * F_hi) > 0
        if same.any():
            t_mid = 0.5 * (t_lo + t_hi)
            F_mid = F(t_mid)
            # Replace the worse side by magnitude
            replace_lo = same & (bd.abs(F_lo) > bd.abs(F_hi))
            t_lo = bd.where(replace_lo, t_mid, t_lo)
            F_lo = bd.where(replace_lo, F_mid, F_lo)
            replace_hi = same & ~replace_lo
            t_hi = bd.where(replace_hi, t_mid, t_hi)
            F_hi = bd.where(replace_hi, F_mid, F_hi)

        # Keep a mask of actually solvable rays
        solvable = valid & ((F_lo * F_hi) <= 0)

        if not solvable.any():
            vignette = bd.ones(N, dtype=bd.bool_)
            return bd.zeros((0, 3)), bd.zeros(0, dtype=bd.bool_), vignette

        # === Safeguarded Newton (1–2 steps) + fixed bisection (4–6 steps) ===
        # Tune these two integers to trade accuracy vs. cost:
        NEWTON_STEPS = 1
        BISECT_STEPS = 6
        EPS = 1e-9

        t = 0.5 * (t_lo + t_hi)

        # Newton phase (safeguarded to bracket)
        for _ in range(NEWTON_STEPS):
            Ft = F(t)
            Fpt = Fp(t)
            good_deriv = bd.abs(Fpt) > EPS
            step = bd.where(good_deriv, Ft / Fpt, bd.zeros_like(Fpt))
            t_proposed = t - step
            escaped = (t_proposed < t_lo) | (t_proposed > t_hi) | (~good_deriv)
            t = bd.where(escaped, 0.5 * (t_lo + t_hi), t_proposed)
            # re-bracket
            Ft = F(t)
            go_left = (F_lo * Ft) <= 0
            t_hi = bd.where(go_left, t, t_hi)
            F_hi = bd.where(go_left, Ft, F_hi)
            t_lo = bd.where(go_left, t_lo, t)
            F_lo = bd.where(go_left, F_lo, Ft)

        # Bisection phase (fixed iterations)
        for _ in range(BISECT_STEPS):
            t_mid = 0.5 * (t_lo + t_hi)
            F_mid = F(t_mid)
            go_left = (F_lo * F_mid) <= 0
            t_hi = bd.where(go_left, t_mid, t_hi)
            F_hi = bd.where(go_left, F_mid, F_hi)
            t_lo = bd.where(go_left, t_lo, t_mid)
            F_lo = bd.where(go_left, F_lo, F_mid)

        t_star = 0.5 * (t_lo + t_hi)

        # Compute intersection points
        P = bd.stack([o[:, 0] + t_star * d[:, 0],
                      o[:, 1] + t_star * d[:, 1],
                      o[:, 2] + t_star * d[:, 2]], axis=1)

        # Field-stop mask on the asphere: r <= C
        r_ok = bd.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) < C
        t_ok = bd.isfinite(t_star) & (t_star > 0)

        inter_mask = solvable & r_ok & t_ok

        if not inter_mask.any():
            vignette = ~inter_mask
            return bd.zeros((0, 3)), bd.zeros(0, dtype=bd.bool_), vignette

        # Return like Surface._SphericalIntersection:
        return P[inter_mask], \
            bd.zeros(P[inter_mask].shape[0], dtype=bd.bool_), \
            ~inter_mask


    def Normal(self, intersections):
        """
        Compute unit surface normals at intersection points lying on this even-aspheric surface.
        intersections: (N,3) bd.array  [x, y, z_world]
        Returns: (N,3) bd.array of unit normals.

        """
        x = intersections[:, 0]
        y = intersections[:, 1]

        # r and dz/dr (vectorized)
        r2 = x * x + y * y
        r = bd.sqrt(bd.maximum(r2, 0.0))
        dzdr = self._AsphereDzDr(r)  # uses self.radius, self.K, self.asphCoef

        # avoid division by zero on axis
        eps = constant(1e-12)
        safe_r = bd.maximum(r, eps)

        # fx = ∂f/∂x, fy = ∂f/∂y
        fx = dzdr * (x / safe_r)
        fy = dzdr * (y / safe_r)

        # Un-normalized normal = (-fx, -fy, 1)
        nx = -fx
        ny = -fy
        nz = bd.ones_like(nx)

        invlen = 1.0 / bd.sqrt(nx * nx + ny * ny + nz * nz)
        nx *= invlen
        ny *= invlen
        nz *= invlen

        # Orientation points +z direction

        return bd.stack([nx, ny, nz], axis=1)


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

            # This is kinda pathetic but hey how else should this if be placed?
            if(drawSag):
                DrawSphericalProfile(self.boundingSurfaceF.radius,
                                     clearSemiDiameter=self.boundingSurfaceF.clearSemiDiameter,
                                     cumulativeThickness=self.boundingSurfaceF.cumulativeThickness,)
                DrawSphericalProfile(self.boundingSurfaceB.radius,
                                     clearSemiDiameter=self.boundingSurfaceB.clearSemiDiameter,
                                     cumulativeThickness=self.boundingSurfaceB.cumulativeThickness,)

        if (self.clearBoundaryL is not None):
            self.clearBoundaryL.DrawSurface()

        if (self.clearBoundaryT is not None):
            self.clearBoundaryT.DrawSurface()


    def GetInfo(self, showBounding=True):
        """
        Get the information of this aspherical surface, include its own parameters and its proxy surfaces, should they exist. Result returns as a big chunk of string.
        """

        info = "Aspheric surface " +\
            "\nRadius: " + str(self.radius) + \
            "\nThickness: " + str(self.thickness) + \
            "\nSemi-diameter: " + str(self.clearSemiDiameter)+\
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

    def _SagAsphereVec(self, r):
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


    def _SphereShapeG(self, r, Rs_abs, sign_rs):
        """g_R(r) = R - sign(R)*sqrt(R^2 - r^2) with |R|=Rs_abs and sign=sign_rs (±1)."""
        # ensure domain: Rs_abs >= max(r)
        return sign_rs * (Rs_abs - bd.sqrt(Rs_abs ** 2 - r ** 2))


    def _PickRRange(self, C):
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


    def _SagAsphere_r2(self, r2):
        """Asphere sag f(r) from r^2 (vectorized). Adds only the local sag (no cumulative)."""
        # base conic
        if bd.isinf(self.radius):
            base = bd.zeros_like(r2)
        else:
            sqrt_term = bd.sqrt(1 - (1 + self.K) * r2 / self.radius ** 2)
            base = r2 / (self.radius * (1 + sqrt_term))
        # even asphere terms: A2, A4, ...  (Horner over r2)
        asph = bd.zeros_like(r2)
        p = r2
        for a in self.asphCoef:
            asph = asph + a * p
            p = p * r2
        return base + asph


    def _AsphereDzDr(self, r):
        """dz/dr for the asphere (vectorized), safe at r=0."""

        r2 = r ** 2

        # base derivative
        if bd.isinf(self.radius):
            d_base = bd.zeros_like(r)
        else:
            sqrt_term = bd.sqrt(1 - (1 + self.K) * r2 / self.radius ** 2)
            denom = (1 + sqrt_term) * sqrt_term
            d_base = (2 * r / self.radius) * (1 / (1 + sqrt_term) + (1 + self.K) * r2 / (self.radius ** 2 * denom))

        # asphere derivative: sum A[i] * 2(i+1) * r^(2(i+1)-1)
        d_asph = bd.zeros_like(r)
        if len(self.asphCoef) > 0:
            p = r  # r^(2(0+1)-1) = r
            for i, a in enumerate(self.asphCoef):
                d_asph = d_asph + (2 * (i + 1)) * a * p
                p = p * r2

        return d_base + d_asph


    def _ProxyCentersZ(self):
        """Return (zc_lower, zc_upper) of the two proxy spheres."""
        zc_lo = float(self.boundingSurfaceF.radiusCenter[2])
        zc_hi = float(self.boundingSurfaceB.radiusCenter[2])
        return zc_lo, zc_hi


    def _IntersectRaySphereCenterz(self, o, d, zc, Rs_abs):
        """
        Intersection t with a sphere centered at (0,0,zc), radius |Rs_abs|.
        Returns t (nearest side by sign rule) and a validity mask (discriminant>=0).
        """
        ocx = o[:, 0]
        ocy = o[:, 1]
        ocz = o[:, 2] - zc
        dx = d[:, 0]
        dy = d[:, 1]
        dz = d[:, 2]

        a = dx * dx + dy * dy + dz * dz
        b = 2.0 * (ocx * dx + ocy * dy + ocz * dz)
        c = (ocx * ocx + ocy * ocy + ocz * ocz) - (Rs_abs * Rs_abs)

        disc = b * b - 4 * a * c
        valid = disc >= 0

        t = bd.full_like(a, bd.nan)

        # Make the “if” scalar-safe on CuPy:
        if bool(valid.any()):
            sd = bd.sqrt(disc[valid])
            t0 = (-b[valid] - sd) / (2 * a[valid])
            t1 = (-b[valid] + sd) / (2 * a[valid])

            # For proxy bracketing, the most robust choice is:
            # pick the nearest positive root (independent of dz and sign(Rs))
            # This avoids side-dependent surprises for “backwards” rays.
            t_near = bd.where(t0 > 0, t0, t1)
            t_far = bd.where(t0 > 0, t1, t0)

            # Alternatively, if near/far rule tied to dz & sign(Rs) is wanted, compute mask_far on the valid slice:
            # mask_far = (bd.sign(self.boundingSurfaceF.radius) != bd.sign(dz[valid]))
            # t_sel = bd.where(mask_far, t_far, t_near)
            # Otherwise, just take the nearest positive:
            t_sel = t_near

            # If both roots are negative, keep NaN:
            has_pos = (t_near > 0) | (t_far > 0)
            t_sel = bd.where(has_pos, t_sel, bd.full_like(t_sel, bd.nan))

            # masked assignment writes M values into N-slot vector
            t = t.copy()
            t[valid] = t_sel

        return t, valid


    def _FindTightEqualRadiusEnvelope(self, C, nr=2048, nR=128, margin=0.0):
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
        f = self._SagAsphereVec(r)

        # candidate |R| range
        Rlo, Rhi = self._PickRRange(C)

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





