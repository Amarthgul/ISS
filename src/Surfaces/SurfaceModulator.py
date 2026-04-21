
from Util.Backend import backend as bd
from Util.Globals import RNG, PRECISION_TYPE, RefreshRNG


class SurfaceModulator:

    def __init__(self):
        self.frontVertex = None
        self.semiDiameter = None

        """Resolution of the maps if any is used"""
        self.mapRes=2048


    def Generate(self):
        # Create the pre-computed resources needed to perform modulation during runtime.
        pass


    def Modulate(self, exitingRB):
        # Given a raybatch object, modulate it and return the modulated one
        pass





class Dust(SurfaceModulator):

    def __init__(self, dustCount=2):
        super().__init__()

        """Number of dusts. It is recommended to keep it below 10. more than 10 specks of visible dusts then your lens is screwed and you should just use scattering attribute of the surfaces instead."""
        self.dustCount = dustCount

        """Dust data structured as:
            [x_position, y_position, base_size, opacity, 1st_dark_ring_size, max_reach]
        The 1st_dark_ring_size means the 1st dark ring of an airy disk.
        Positions and sizes are generated in normalized element space and mapped to runtime space when needed.
        """
        self.dustData = None  # No per-dust map needed

        self._maxSize = 0.02
        self._maxOpacity = 1.0
        self._opacityFadePower = 2
        self._1stRingMaxRatio = 2.5  # Ratio of the 1st dark ring size to the size of the dust itself

        self._airyNormal = None

        self._normalBlend = 0.25      # Blend ratio between airy disk normal and original normal
        self._normalStrength = 0.12   # Scales the xy perturbation stored in the Airy lookup
        self._ringSigma = 0.25        # Softness of the dark ring in the lookup profile
        self._minTransmission = 1e-6
        self._eps = 1e-12


    def Generate(self):

        # Randomly create dustCount number of dusts, assume they exist in a [0, 1] circular area, circle center at (.5, .5). This circular area will later be mapped into the self.semiDiameter during runtime. However, if self.semiDiameter is None, then treat the element as a square and place the dust in the entire [0, 1] square area

        # Randomly generate base size, opacity, and the size of the 1st dark ring. Then use the size of the 1st dark ring to calculate the max reach size, i.e., the furthest distance the dust could affect the ray direction.

        count = int(self.dustCount)
        if count <= 0:
            self.dustData = bd.zeros((0, 6), dtype=PRECISION_TYPE)
            self.mapRes = 512
            self._airyNormal = self._generate_airy_lookup()
            return self

        if self.semiDiameter is None:
            dust_x = RNG.rand(count)
            dust_y = RNG.rand(count)
        else:
            dust_x, dust_y = self._sample_unit_disk(count)

        base_size = (0.2 + 0.8 * RNG.rand(count)) * self._maxSize
        opacity = (0.35 + 0.65 * RNG.rand(count)) * self._maxOpacity
        ring_ratio = 1.0 + RNG.rand(count) * max(self._1stRingMaxRatio - 1.0, 0.0)
        first_ring = base_size * ring_ratio
        max_reach = bd.copy(first_ring)

        self.dustData = bd.stack(
            (dust_x, dust_y, base_size, opacity, first_ring, max_reach), axis=1
        ).astype(PRECISION_TYPE)

        self.mapRes = 512
        # Generate self._airyNormal, a normal map of an Airy disk at mapRes. This map will be used as the directional perturbation "lookup map" for all dusts.
        self._airyNormal = self._generate_airy_lookup()

        RefreshRNG()

        return self


    def Modulate(self, exitingRB):

        # The exitingRB is when the primary tracing is already finished and the rays are leaving the surface. As such, their position represents the intersection on the surface.

        # Since there are only a handful of dusts, calculate position difference between the rays and the dusts. If the ray sits further than the max_reach, they can be ignored.

        # For radiance change, use distance between dust and intersection to get a size ratio, opacity = 1 - ratio^opacityFadePower.
        # If the intersection is right at the dust, ratio = 0, opacity = 1, full blockage. If intersection is right at the edge of the dust size, ratio = 1, opacity = 0, full pass.
        # Change the radiance

        # For directional change, transfer the relative position of the intersection within the 1st_dark_ring_size to the image space of the _airyNormal, then blend the normal map's direction with the original direction.

        if self.dustCount == 0: return exitingRB

        pos_xy = bd.asarray(exitingRB.Position()[:, :2], dtype=PRECISION_TYPE)
        dir_xyz = bd.asarray(exitingRB.Direction(), dtype=PRECISION_TYPE)

        world_dust = self._dust_world_params()
        if world_dust is None or world_dust.shape[0] == 0:
            return exitingRB

        transmission = bd.ones(pos_xy.shape[0], dtype=PRECISION_TYPE)
        perturb_xy = bd.zeros((pos_xy.shape[0], 2), dtype=PRECISION_TYPE)
        perturb_w = bd.zeros(pos_xy.shape[0], dtype=PRECISION_TYPE)

        for i in range(world_dust.shape[0]):
            cx, cy, base_size, opacity, ring_size, max_reach = world_dust[i]

            delta = pos_xy - bd.array([cx, cy], dtype=PRECISION_TYPE)[None, :]
            dist = bd.linalg.norm(delta, axis=1)

            # Radiance attenuation inside the physical dust core.
            in_core = dist <= base_size
            if bd.any(in_core):
                ratio = bd.clip(dist / bd.maximum(base_size, self._eps), 0.0, 1.0)
                local_block = opacity * (1.0 - ratio ** self._opacityFadePower)
                local_trans = 1.0 - local_block
                transmission = bd.where(in_core, transmission * bd.maximum(local_trans, self._minTransmission), transmission)

            # Directional change inside the first dark ring support.
            in_ring = dist <= max_reach
            if bd.any(in_ring):
                ring_safe = bd.maximum(ring_size, self._eps)
                rel = delta[in_ring] / ring_safe
                uv = (rel + 1.0) * 0.5
                sampled = self._bilinear_lookup(uv)

                # The lookup normal lives in the local dust frame. Use its xy as a tangent-space perturbation.
                local_xy = sampled[:, :2]
                ring_ratio = bd.clip(1.0 - dist[in_ring] / bd.maximum(max_reach, self._eps), 0.0, 1.0)
                weight = opacity * ring_ratio

                perturb_xy[in_ring] += local_xy * weight[:, None]
                perturb_w[in_ring] += weight

        transmission = bd.maximum(transmission, self._minTransmission)
        exitingRB.RadianceChange(transmission)

        affected = perturb_w > 0.0
        if bd.any(affected):
            avg_xy = bd.zeros_like(perturb_xy)
            avg_xy[affected] = perturb_xy[affected] / bd.maximum(perturb_w[affected, None], self._eps)

            perturb_dir = bd.copy(dir_xyz)
            perturb_dir[affected, 0:2] += avg_xy[affected]
            perturb_dir = self._normalize_rows(perturb_dir)

            new_dir = bd.copy(dir_xyz)
            new_dir[affected] = dir_xyz[affected] * (1.0 - self._normalBlend) + perturb_dir[affected] * self._normalBlend
            new_dir = self._normalize_rows(new_dir)
            exitingRB.SetDirection(new_dir)

        return exitingRB


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _normalize_rows(self, arr):
        norm = bd.linalg.norm(arr, axis=1, keepdims=True)
        norm = bd.maximum(norm, self._eps)
        return arr / norm


    def _sample_unit_disk(self, count):
        """Sample points inside a unit disk centered at (0.5, 0.5)."""
        theta = RNG.rand(count) * (2.0 * bd.pi)
        r = bd.sqrt(RNG.rand(count)) * 0.5
        x = 0.5 + r * bd.cos(theta)
        y = 0.5 + r * bd.sin(theta)
        return x, y


    def _generate_airy_lookup(self):
        """
        Create a small universal lookup map whose normals roughly follow
        the derivative of the Airy disk up to the first dark ring.

        The radial profile is intentionally compact and smooth:
            - a bright central lobe
            - a darker annulus near the first dark ring
            - no oscillation beyond the first dark ring
        """
        res = int(self.mapRes)
        lin = bd.linspace(-1.0, 1.0, res, dtype=PRECISION_TYPE)
        xx, yy = bd.meshgrid(lin, lin, indexing="xy")
        rr = bd.sqrt(xx * xx + yy * yy)

        # Compact Airy-inspired height field.
        # Positive center, shallow negative ring near r=1, zero beyond.
        center_lobe = bd.where(rr <= 1.0, (1.0 - rr * rr) ** 2, 0.0)
        ring = -0.18 * bd.exp(-((rr - 1.0) ** 2) / (2.0 * self._ringSigma * self._ringSigma))
        height = bd.where(rr <= 1.15, center_lobe + ring, 0.0)

        # Central differences via roll. Edge values are masked anyway.
        dx = 0.5 * (bd.roll(height, -1, axis=1) - bd.roll(height, 1, axis=1))
        dy = 0.5 * (bd.roll(height, -1, axis=0) - bd.roll(height, 1, axis=0))

        nx = -dx * self._normalStrength
        ny = -dy * self._normalStrength
        nz = bd.ones_like(nx)

        normal = bd.stack((nx, ny, nz), axis=2)
        normal = normal / bd.maximum(bd.linalg.norm(normal, axis=2, keepdims=True), self._eps)

        # Outside the support, keep a neutral normal.
        neutral = bd.zeros_like(normal)
        neutral[:, :, 2] = 1.0
        valid = rr <= 1.0
        normal = bd.where(valid[:, :, None], normal, neutral)

        return normal.astype(PRECISION_TYPE)


    def _map_scale(self):
        if self.semiDiameter is None:
            return bd.array(1.0, dtype=PRECISION_TYPE)
        return bd.array(2.0 * self.semiDiameter, dtype=PRECISION_TYPE)


    def _center_xy(self):
        if self.frontVertex is None:
            return bd.array([0.0, 0.0], dtype=PRECISION_TYPE)
        fv = bd.asarray(self.frontVertex)
        return fv[:2].astype(PRECISION_TYPE)


    def _dust_world_params(self):
        """Map normalized dust positions / sizes into runtime surface space."""
        if self.dustData is None:
            return None

        data = bd.asarray(self.dustData, dtype=PRECISION_TYPE)
        centers = bd.copy(data[:, :2])
        sizes = bd.copy(data[:, 2:])

        if self.semiDiameter is not None:
            scale = self._map_scale()
            centers = self._center_xy()[None, :] + (centers - 0.5) * scale
            sizes = sizes * scale

        return bd.concatenate((centers, sizes), axis=1)


    def _bilinear_lookup(self, uv):
        """Sample self._airyNormal at uv in [0, 1]. Returns (N, 3)."""
        tex = self._airyNormal
        h = tex.shape[0]
        w = tex.shape[1]

        uv = bd.clip(uv, 0.0, 1.0)
        x = uv[:, 0] * (w - 1)
        y = uv[:, 1] * (h - 1)

        x0 = bd.floor(x).astype(bd.int32)
        y0 = bd.floor(y).astype(bd.int32)
        x1 = bd.minimum(x0 + 1, w - 1)
        y1 = bd.minimum(y0 + 1, h - 1)

        wx = (x - x0).reshape(-1, 1)
        wy = (y - y0).reshape(-1, 1)

        n00 = tex[y0, x0]
        n10 = tex[y0, x1]
        n01 = tex[y1, x0]
        n11 = tex[y1, x1]

        n0 = n00 * (1.0 - wx) + n10 * wx
        n1 = n01 * (1.0 - wx) + n11 * wx
        out = n0 * (1.0 - wy) + n1 * wy
        return self._normalize_rows(out)

