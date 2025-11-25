

from Util.Backend import backend as bd
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR
from Raytracing.Reflection import Reflect
from Raytracing.Polarization import SenkrechtUndParallel, PolarizeRB, ResidueRB, FresnelReflectance, QuantitativePolarize
from Raytracing.RayBatch import RayBatch
from Material import Material

from .Surface import Surface





class MLA(Surface):
    def __init__(self, r, sd, m, t=0):
        """
        :param r: radius of each micro lens.
        :param sd: clear aperture, or clear semi-diameter.
        :param m: material of the micro lenses.
        :param t: surface class argument, not used here.
        """
        super().__init__(r, t, sd, m)

        self.horizontalCount = 6000
        self.verticalCount = 4000
        self.pixelPitch = 0.006 # unit in mm
        self.offsets = None     # No offset by default

        self.width = 36
        self.height = 24

        self.centers = None


    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the vertices. This is for when the surface share the same optical axis with the lens.
        """
        cumulativeT = bd.array(cumulativeT)

        # The local optical axis remains the same as OBJ FACING
        self.cumulativeThickness = cumulativeT

        # The sdCumulative does not really exist for MLA
        self.sdCumulative = cumulativeT


    def SampleFromClearAperture(self, sampleCount=32):
        pass


    def SetShape(self, w:int, h:int, p:float, o:bd.ndarray=None):
        """
        Manually set the shape parameters of the MLA, including lens count and pixel pitch.

        :param w: number of micro lenses on the width/horizontal axis.
        :param h: number of micro lenses on the height/vertical axis.
        :param p: pixel pitch for each micro lenses in mm. This should be smaller than the radius.
        :param o: array recording the offset of each micro lenses in mm, by default None.
        """
        self.horizontalCount = w
        self.verticalCount = h
        self.pixelPitch = p
        self.offsets = o

        self.centers = None

        # Update the physical dimension of the MLA
        self.width = self.pixelPitch * w
        self.height = self.pixelPitch * h


    def Intersection(self, incidentRaybatch):
        pass


    def Normal(self, intersections):
        """
        Given an array of intersections, calculate the normal direction on these intersection points.

        :param intersections: points on the surface.

        :return: Normalized normals of the intersection points on this surface.
        """

        # The intersections are regarded to already on the surface and no check is needed.

        pass


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False, useClearBoundary=False):
        """
        Trace through the MLA and get the different types of rays and flags.

        :param incidentRaybatch: incident RayBatch object.
        :param previousRI: IOR of the medium prior to MLA. An array the same size as incidentRaybatch.
        :param inverted: Not used for MLA.
        :param reflection: Not used for MLA.
        :param useClearBoundary: Not used for MLA.

        :return:
            refractedRB: RayBatch of rays after MLA (including seam rays),
            TIR:         bool array for incident rays that undergo total internal reflection,
            boolVig:     bool array for rays that are vignetted (outside MLA active area),
            reflectedRB: None (no explicit reflected rays modeled for MLA).
        """

        # ----------------- 0. Extract ray data from RayBatch -----------------
        o = incidentRaybatch.Position()     # shape (N, 3)
        d = incidentRaybatch.Direction()    # shape (N, 3)
        lam = incidentRaybatch.Wavelength()    # shape (N,)
        L   = incidentRaybatch.RadianceTerms()    # shape (N,)

        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

        N = ox.shape[0]

        # previousRI: n_up for each ray (same size as lam)
        n_up = previousRI           # shape (N,)

        # material RIs
        # If Material has a method RI(lambda) that is vectorized, use that:
        n_ml   = self.material.RI(lam)    # microlens index
        n_down = self.material.RI(lam)    # or another medium below if different

        # ----------------- 1. Which microlens cell? -----------------
        i, j, inside_tile = self._microlens_index(ox, oy)

        # boolVig: rays completely outside MLA active area
        boolVig = ~inside_tile

        # start with all rays marked as non-TIR
        TIR = bd.zeros(N, dtype=bd.bool_)

        # ----------------- 2. Compute microlens centers with offsets -----------------
        pitch = self.pixelPitch

        # nominal centers
        xc_nom = (i + 0.5) * pitch
        yc_nom = (j + 0.5) * pitch

        # offsets: (verticalCount, horizontalCount, 2) or None
        if self.offsets is not None:
            # gather offsets per ray
            # note: backend may have different advanced indexing rules; adjust if needed.
            dx_off = bd.zeros_like(ox)
            dy_off = bd.zeros_like(oy)

            valid = inside_tile
            dx_off[valid] = self.offsets[j[valid], i[valid], 0]
            dy_off[valid] = self.offsets[j[valid], i[valid], 1]
        else:
            dx_off = bd.zeros_like(ox)
            dy_off = bd.zeros_like(oy)

        xc = xc_nom + dx_off
        yc = yc_nom + dy_off

        # sphere center z-coordinate is global constant
        # z_ml_bot: bottom plane of microlens
        # h_max: sag at vertex; for a spherical cap it's r - sqrt(r^2 - s_d^2)
        z_ml_bot = 0.0  # <-- set this to the actual bottom-plane z of the MLA
        h_max = self.radius - bd.sqrt(self.radius * self.radius - self.clearSemiDiameter * self.clearSemiDiameter)
        zc = z_ml_bot + h_max - self.radius

        cz = bd.full_like(ox, zc)

        # ----------------- 3. Local coords & clear-aperture mask -----------------
        x_l = ox - xc
        y_l = oy - yc

        # within clear semi-diameter
        inside_aperture = (x_l * x_l + y_l * y_l) <= (self.clearSemiDiameter * self.clearSemiDiameter)

        # Rays that are outside MLA tile are already vignetted; they won't be refracted.
        # Effective "inside MLA aperture" mask:
        mask_lens = inside_tile & inside_aperture

        # Rays that are inside tile but outside aperture are "seam rays":
        mask_seam = inside_tile & (~inside_aperture)

        # ----------------- 4. Allocate outputs -----------------
        # Initialize outputs with "straight-through" as default
        qx = bd.array(ox, copy=True)
        qy = bd.array(oy, copy=True)
        qz = bd.array(oz, copy=True)
        dx_out = bd.array(dx, copy=True)
        dy_out = bd.array(dy, copy=True)
        dz_out = bd.array(dz, copy=True)

        # Seam rays: keep direction unchanged; they don't get TIR.
        # Vignetted rays: we will keep them in RB but mark boolVig=True
        # (You can also drop them from the RayBatch if that’s preferred.)

        # ----------------- 5. Process rays that actually go through a microlens -----------------
        if bd.any(mask_lens):
            idx = bd.where(mask_lens)[0]

            # Gather subset
            ox_l = ox[idx]
            oy_l = oy[idx]
            oz_l = oz[idx]
            dx_l = dx[idx]
            dy_l = dy[idx]
            dz_l = dz[idx]
            xc_l = xc[idx]
            yc_l = yc[idx]
            cz_l = cz[idx]
            n_up_l = n_up[idx]
            n_ml_l = n_ml[idx]
            n_dn_l = n_down[idx]

            # ---- 5.1 Sphere intersection at top ----
            t_sphere, hit_sphere = self._sphere_intersection(
                ox_l, oy_l, oz_l,
                dx_l, dy_l, dz_l,
                xc_l, yc_l, cz_l,
                self.radius
            )

            # update TIR / validity: rays that fail to hit the sphere are effectively seams
            valid_lens = hit_sphere

            # Hit positions
            px = ox_l + t_sphere * dx_l
            py = oy_l + t_sphere * dy_l
            pz = oz_l + t_sphere * dz_l

            # Surface normals (outward from sphere)
            nx = (px - xc_l) / self.radius
            ny = (py - yc_l) / self.radius
            nz = (pz - cz_l) / self.radius

            # ---- 5.2 Refraction at top: n_up -> n_ml ----
            dx1, dy1, dz1, tir_top = self._refract(dx_l, dy_l, dz_l, nx, ny, nz, n_up_l, n_ml_l)

            # Combined validity: must hit sphere and not TIR at top
            valid_lens = valid_lens & (~tir_top)

            # ---- 5.3 Intersect bottom plane z = z_ml_bot ----
            # t_bot = (z_ml_bot - pz) / dz1
            t_bot = (z_ml_bot - pz) / dz1
            valid_lens = valid_lens & (t_bot > 0)

            qx_l = px + t_bot * dx1
            qy_l = py + t_bot * dy1
            qz_l = bd.full_like(qx_l, z_ml_bot)

            # ---- 5.4 Refraction at bottom: n_ml -> n_down ----
            # bottom normal points +z (out of lens toward sensor)
            nx_b = bd.zeros_like(qx_l)
            ny_b = bd.zeros_like(qy_l)
            nz_b = bd.ones_like(qz_l)

            dx2, dy2, dz2, tir_bot = self._refract(dx1, dy1, dz1, nx_b, ny_b, nz_b, n_ml_l, n_dn_l)

            # total TIR for these rays
            tir_l = tir_top | tir_bot

            # update global TIR flags
            TIR[idx] = tir_l

            # for rays that experienced TIR, we can choose to:
            # - mark them TIR and leave position/direction as original (or NaN)
            # - or treat them as lost; here we treat them as lost (no update).
            # valid ones get the new position/direction
            final_valid = valid_lens & (~tir_l)

            if bd.any(final_valid):
                idx_valid = idx[final_valid]

                qx[idx_valid] = qx_l[final_valid]
                qy[idx_valid] = qy_l[final_valid]
                qz[idx_valid] = qz_l[final_valid]
                dx_out[idx_valid] = dx2[final_valid]
                dy_out[idx_valid] = dy2[final_valid]
                dz_out[idx_valid] = dz2[final_valid]

        # ----------------- 6. Build output RayBatch -----------------
        # Again, map this to your actual RayBatch constructor/fields.
        refractedRB = RayBatch()
        refractedRB.o = bd.stack([qx, qy, qz], axis=-1)
        refractedRB.d = bd.stack([dx_out, dy_out, dz_out], axis=-1)
        refractedRB.w = lam
        refractedRB.SetRadianceTerms(L)

        # No explicit reflections modeled here
        reflectedRB = None

        return refractedRB, TIR, boolVig, reflectedRB


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================



    def _microlens_index(self, x, y):
        """
        Compute integer microlens indices (i, j) from global x,y.

        i: horizontal index [0, horizontalCount)
        j: vertical   index [0, verticalCount)

        x, y are 1D backend arrays.
        """
        pitch = self.pixelPitch

        # floor division; cast to int
        i = bd.floor(x / pitch).astype(bd.int32)
        j = bd.floor(y / pitch).astype(bd.int32)

        # mark out-of-range as -1
        inside_i = (i >= 0) & (i < self.horizontalCount)
        inside_j = (j >= 0) & (j < self.verticalCount)
        inside = inside_i & inside_j

        # anything outside the MLA active area gets index -1
        i = bd.where(inside, i, -bd.ones_like(i))
        j = bd.where(inside, j, -bd.ones_like(j))

        return i, j, inside


    def _sphere_intersection(self, ox, oy, oz, dx, dy, dz, cx, cy, cz, radius):
        """
        Intersection t with sphere centered at (cx,cy,cz) with radius.

        (ox,oy,oz): ray origins
        (dx,dy,dz): ray directions (unit)
        returns t (backend array), and a bool mask 'hit'.
        """
        # shift origin relative to sphere center
        oxp = ox - cx
        oyp = oy - cy
        ozp = oz - cz

        # quadratic coefficients with |d|=1
        b = oxp * dx + oyp * dy + ozp * dz
        c2 = oxp * oxp + oyp * oyp + ozp * ozp - radius * radius

        disc = b * b - c2
        hit = disc >= 0

        sqrt_disc = bd.zeros_like(disc)
        sqrt_disc[hit] = bd.sqrt(disc[hit])

        # take nearer positive root
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        # valid if > 0
        t = bd.where(t1 > 0, t1, t2)
        hit = hit & (t > 0)

        return t, hit


    def _refract(self, dx, dy, dz, nx, ny, nz, n1, n2):
        """
        Vector Snell refraction with possible TIR.

        (dx,dy,dz): incident directions (unit)
        (nx,ny,nz): surface normals (unit) pointing 'outward' of medium 2
        n1: refractive index of incident medium
        n2: refractive index of transmitted medium

        returns (dx_out, dy_out, dz_out, tir_mask)
        """
        # cos of incident angle: c = -n·d  (assuming d points towards interface)
        c = -(nx * dx + ny * dy + nz * dz)

        eta = n1 / n2
        k = 1.0 - eta * eta * (1.0 - c * c)

        tir = k < 0.0

        sqrtk = bd.zeros_like(k)
        sqrtk[~tir] = bd.sqrt(k[~tir])

        # transmitted direction
        factor = eta * c - sqrtk
        dx_out = eta * dx + factor * nx
        dy_out = eta * dy + factor * ny
        dz_out = eta * dz + factor * nz

        # normalize for safety
        mag = bd.sqrt(dx_out*dx_out + dy_out*dy_out + dz_out*dz_out)
        dx_out = dx_out / mag
        dy_out = dy_out / mag
        dz_out = dz_out / mag

        return dx_out, dy_out, dz_out, tir



