

from .SurfaceModulator import SurfaceModulator
from Util.Backend import backend as bd
from Util.Globals import RNG, PRECISION_TYPE


class OnionRing(SurfaceModulator):

    def __init__(self, ringCount=38, disturbance=1):
        super().__init__()

        """Total count of ring layers."""
        self.ringCount = ringCount

        """Max amount of positional offset that could happen to a ring. 0 being no offset, 1 could make it become tangent with the border of the neighboring ring at some point. """
        self.disturbance = disturbance

        self.normalMap = None
        self.heightMap = None
        self.perlinMask = None

        self._normalBlend = 0.0001 # 1 being replacement, 0 being no modification

        self._normalSinSag = 0.5 # Blend each ring profile between a sine shape and a spherical shape. 1 for pure sine, 0 for pure sphere/circular.

        self._ringHeight = 0.5 # Height of the profile. When set to 1, the profile has a full height equal to half of the ideal per-ring width, calculated as 1/(ringCount*2). When set to 0, then there is nothing and no perturbation will happen.

        self._normalStrength = 2.0
        self._disturbanceHarmonics = 3
        self._disturbanceSamples = None
        self._eps = 1e-12

        self._centerFade = 0.5  # When set to 1, center will be replaced with flat normal. When set to 0, skip this center fade entirely.
        self._centerFadeRadius = 0.35 # Ratio of the radius at which point the fade stops. 1 for the entire map radius range, 0 for none.

        # Perlin-like mask controls.
        self._perlinScale = 6.0          # Base cell count across the full diameter.
        self._perlinOctaves = 3          # Number of self-added fractal iterations.
        self._perlinPersistence = 0.5    # Amplitude decay per octave.
        self._perlinLacunarity = 2.0     # Frequency growth per octave.
        self._perlinStrength = 0.35      # 0 disables, 1 allows full local height suppression.


    def Generate(self):

        if self._ringHeight <= 0:
            self.heightMap = bd.zeros((self.mapRes, self.mapRes), dtype=PRECISION_TYPE)
            self.perlinMask = bd.ones((self.mapRes, self.mapRes), dtype=PRECISION_TYPE)
            self.normalMap = self._NeutralNormalMap(self.mapRes)
            return self

        lin = bd.linspace(-1.0, 1.0, self.mapRes, dtype=PRECISION_TYPE)
        xx, yy = bd.meshgrid(lin, lin, indexing='xy')
        rr = bd.sqrt(xx * xx + yy * yy)

        # Build height profile ring by ring. Each ring gets its own disturbed center
        # and its own angular radius wobble, but both share a single combined
        # disturbance budget so the total offset allowance is not exceeded.
        height = bd.zeros_like(rr)
        ringWidth = 1.0 / max(self.ringCount, 1)
        amp = self._ringHeight * (ringWidth * 0.5)

        signs = self._RingSign()
        center_offsets, radius_weights = self._PerRingDisturbanceParams()

        for idx in range(self.ringCount):
            inner = idx * ringWidth
            outer = (idx + 1) * ringWidth

            # Each ring has its own slightly shifted center.
            cx = center_offsets[idx, 0]
            cy = center_offsets[idx, 1]
            ring_rr = bd.sqrt((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy))
            ring_theta = bd.arctan2(yy - cy, xx - cx)

            # Optional angular radius disturbance around that shifted center.
            ring_rr = self._DisturbedRadius(
                ring_rr, ring_theta, self.ringCount,
                amplitudeScale=radius_weights[idx],
                ringIndex=idx
            )

            local = (ring_rr - inner) / max(outer - inner, self._eps)
            in_ring = (local >= 0.0) & (local <= 1.0)

            ring_profile = self._RingProfile(local)
            sign = signs[idx]
            height = bd.where(in_ring, height + sign * amp * ring_profile, height)

        # Clamp outside unit disk.
        height = bd.where(rr <= 1.0, height, 0.0)

        # Apply Perlin-like mask before the normal map is generated.
        self.perlinMask = self._GeneratePerlinMask(xx, yy, rr)
        height = height * self.perlinMask

        self.heightMap = height

        # Use non-wrapping finite differences so the visualization is not contaminated by roll-over seams.
        pixel_size = 2.0 / max(self.mapRes - 1, 1)
        dx = bd.zeros_like(height)
        dy = bd.zeros_like(height)

        dx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / (2.0 * pixel_size)
        dy[1:-1, :] = (height[2:, :] - height[:-2, :]) / (2.0 * pixel_size)

        dx[:, 0] = (height[:, 1] - height[:, 0]) / pixel_size
        dx[:, -1] = (height[:, -1] - height[:, -2]) / pixel_size
        dy[0, :] = (height[1, :] - height[0, :]) / pixel_size
        dy[-1, :] = (height[-1, :] - height[-2, :]) / pixel_size

        # Center fade reduces the perturbation strength near the optical center.
        center_weight = self._CenterFadeWeight(rr)

        # Increase normal strength impact.
        nx = -dx * self._normalStrength * center_weight
        ny = -dy * self._normalStrength * center_weight
        nz = bd.ones_like(nx)

        normal = bd.stack((nx, ny, nz), axis=2)
        normal = normal / bd.maximum(bd.linalg.norm(normal, axis=2, keepdims=True), self._eps)

        neutral = self._NeutralNormalMap(self.mapRes)
        self.normalMap = bd.where((rr <= 1.0)[:, :, None], normal, neutral)
        return self


    def Modulate(self, exitingRB):

        # Transform the raybatch position into UV coordinate of the normalMap, then blend the direction
        if exitingRB is None or getattr(exitingRB, 'value', None) is None:
            return exitingRB
        if exitingRB.value.shape[0] == 0:
            return exitingRB
        if self.normalMap is None:
            self.Generate()

        blend = float(self._normalBlend)
        if blend <= 0.0:
            return exitingRB
        blend = min(max(blend, 0.0), 1.0)

        pos = bd.asarray(exitingRB.Position()[:, :2], dtype=PRECISION_TYPE)
        direction = bd.asarray(exitingRB.Direction(), dtype=PRECISION_TYPE)

        uv, valid = self._PositionToUV(pos)
        if not bool(bd.any(valid)):
            return exitingRB

        sampled = self._BilinearLookup(uv)

        # Treat the sampled map as a perturbation normal field, so only the XY departure from neutral matters.
        delta_xy = sampled[:, :2]
        perturbed = bd.copy(direction)
        perturbed[:, 0:2] += delta_xy
        perturbed = self._NormalizeRows(perturbed)

        out_dir = bd.copy(direction)
        out_dir[valid] = direction[valid] * (1.0 - blend) + perturbed[valid] * blend
        out_dir = self._NormalizeRows(out_dir)
        exitingRB.SetDirection(out_dir)

        return exitingRB


    def NormalBlend(self, normals, intersections):

        # Given many normals and their position/intersection, blend it with the normal map
        if self.normalMap is None:
            self.Generate()

        if normals is None or normals.shape[0] == 0:
            return normals

        blend = float(self._normalBlend)
        if blend <= 0.0:
            return normals
        blend = min(max(blend, 0.0), 1.0)

        # Extract XY positions and cast to backend array
        pos = bd.asarray(intersections[:, :2], dtype=PRECISION_TYPE)
        base_normals = bd.asarray(normals, dtype=PRECISION_TYPE)

        # Map intersection coordinates to the UV space of the normal map
        uv, valid = self._PositionToUV(pos)
        if not bool(bd.any(valid)):
            return base_normals

        # Sample the map
        sampled = self._BilinearLookup(uv)

        # Treat the sampled map as a perturbation normal field, extracting the XY departure
        delta_xy = sampled[:, :2]

        # Apply the perturbation to the base normals
        perturbed = bd.copy(base_normals)
        perturbed[:, 0:2] += delta_xy
        perturbed = self._NormalizeRows(perturbed)

        # Lerp between the original normals and the perturbed normals based on the blend weight
        out_normals = bd.copy(base_normals)
        out_normals[valid] = base_normals[valid] * (1.0 - blend) + perturbed[valid] * blend
        out_normals = self._NormalizeRows(out_normals)

        return out_normals


    def ShowNormalMap(self, exaggeration=24.0, showHeight=True):
        if self.normalMap is None:
            self.Generate()

        arr = self.normalMap
        height = self.heightMap
        if hasattr(arr, 'get'):
            arr = arr.get()
        if hasattr(height, 'get'):
            height = height.get()

        try:
            import numpy as np
            import matplotlib.pyplot as plt

            viz = np.array(arr, copy=True)
            viz[:, :, 0] *= exaggeration
            viz[:, :, 1] *= exaggeration
            norm = np.linalg.norm(viz, axis=2, keepdims=True)
            norm = np.maximum(norm, self._eps)
            viz = viz / norm
            img = np.clip((viz + 1.0) * 0.5, 0.0, 1.0)

            if showHeight and height is not None:
                vmax = float(np.max(np.abs(height)))
                vmax = max(vmax, self._eps)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(img)
                axes[0].set_title(f'Onion Ring Normal Map (XY exaggerated x{exaggeration:g})')
                axes[0].axis('off')

                hm = axes[1].imshow(height, cmap='coolwarm', vmin=-vmax, vmax=vmax)
                axes[1].set_title('Underlying Height Map')
                axes[1].axis('off')
                fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.title(f'Onion Ring Normal Map (XY exaggerated x{exaggeration:g})')
                plt.axis('off')
                plt.show()
        except Exception:
            return arr


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _RingSign(self, unifiedSign=0):

        # Use this method to determine the sign of each ring, i.e., if they are convex or concave.

        # If unifiedSign is 0, then use alternating signs, otherwise, use whatever sign that argument has
        count = int(max(1, self.ringCount))

        if unifiedSign == 0:
            signs = bd.ones(count, dtype=PRECISION_TYPE)
            signs[1::2] = -1.0
            return signs

        val = 1.0 if unifiedSign > 0 else -1.0
        return bd.full(count, val, dtype=PRECISION_TYPE)

    def _NormalizeRows(self, arr):
        norm = bd.linalg.norm(arr, axis=1, keepdims=True)
        norm = bd.maximum(norm, self._eps)
        return arr / norm

    def _NeutralNormalMap(self, res):
        normal = bd.zeros((res, res, 3), dtype=PRECISION_TYPE)
        normal[:, :, 2] = 1.0
        return normal

    def _RingProfile(self, local):
        """
        local in [0, 1] within a single ring.
        Blend between a half-sine sag and a circular-cap sag.
        Both profiles are zero at the boundaries and peak at the middle.
        """
        l = bd.clip(local, 0.0, 1.0)

        sine_prof = bd.sin(bd.pi * l)

        x = 2.0 * l - 1.0
        circ_prof = bd.sqrt(bd.maximum(0.0, 1.0 - x * x))

        w = float(min(max(self._normalSinSag, 0.0), 1.0))
        return (w * sine_prof) + ((1.0 - w) * circ_prof)

    def _CenterFadeWeight(self, rr):
        """
        Return multiplicative weight for XY normal strength.

        At the center, the weight becomes (1 - _centerFade).
        At and beyond _centerFadeRadius, the weight becomes 1.
        """
        fade = float(min(max(self._centerFade, 0.0), 1.0))
        radius = float(min(max(self._centerFadeRadius, 0.0), 1.0))

        if fade <= 0.0 or radius <= 0.0:
            return bd.ones_like(rr)

        t = bd.clip(rr / max(radius, self._eps), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        return (1.0 - fade) + fade * smooth

    def _DisturbedRadius(self, rr, theta, ringCount, amplitudeScale=1.0, ringIndex=0):
        """
        Apply a small angular modulation to the radial coordinate.

        amplitudeScale is a per-ring [0, 1] weight that shares the same total
        disturbance allowance with the center offset. When it is 0, this method
        leaves the radius unchanged for that ring.
        """
        if self.disturbance <= 0:
            return rr

        ringWidth = 1.0 / max(ringCount, 1)
        maxShift = 0.5 * ringWidth * float(self.disturbance) * float(amplitudeScale)
        if maxShift <= 0.0:
            return rr

        samples = (RNG.rand(self._disturbanceHarmonics) * 2.0 - 1.0).astype(PRECISION_TYPE)

        field = bd.zeros_like(theta)
        for i in range(self._disturbanceHarmonics):
            freq = i + 1
            phase = (2.0 * bd.pi) * samples[i]
            amp = samples[i] / freq
            field = field + amp * bd.sin(freq * theta + phase)

        denom = bd.max(bd.abs(field))
        denom = bd.maximum(denom, self._eps)
        field = field / denom

        return rr + (maxShift * field)

    def _PerRingDisturbanceParams(self):
        """
        Create per-ring disturbance parameters.

        The total allowance is half a ring width at disturbance=1.
        For each ring, split that allowance between:
            - center offset magnitude
            - angular radius wobble amplitude

        so that:
            center_offset + radius_wobble <= total_allowance
        """
        count = int(max(1, self.ringCount))
        ringWidth = 1.0 / max(count, 1)
        total_allowance = 0.5 * ringWidth * float(max(self.disturbance, 0.0))

        if total_allowance <= 0.0:
            center_offsets = bd.zeros((count, 2), dtype=PRECISION_TYPE)
            radius_weights = bd.zeros(count, dtype=PRECISION_TYPE)
            return center_offsets, radius_weights

        center_offsets = bd.zeros((count, 2), dtype=PRECISION_TYPE)
        radius_weights = bd.zeros(count, dtype=PRECISION_TYPE)

        split = RNG.rand(count).astype(PRECISION_TYPE)

        angles = (2.0 * bd.pi) * RNG.rand(count).astype(PRECISION_TYPE)
        center_mag = total_allowance * split
        center_offsets[:, 0] = center_mag * bd.cos(angles)
        center_offsets[:, 1] = center_mag * bd.sin(angles)

        radius_weights[:] = (1.0 - split)

        return center_offsets, radius_weights

    def _GeneratePerlinMask(self, xx, yy, rr):
        """
        Fractal Perlin-like mask in [1 - strength, 1].
        It multiplies the height map, so brighter values preserve more height.
        """
        strength = float(min(max(self._perlinStrength, 0.0), 1.0))
        octaves = int(max(1, self._perlinOctaves))
        scale = float(max(self._perlinScale, 1.0))
        persistence = float(max(self._perlinPersistence, 0.0))
        lacunarity = float(max(self._perlinLacunarity, 1.0))

        if strength <= 0.0:
            return bd.where(rr <= 1.0, bd.ones_like(rr), bd.ones_like(rr))

        fbm = bd.zeros_like(xx)
        amp = 1.0
        amp_sum = 0.0
        freq = scale

        for _ in range(octaves):
            cells = int(max(1, round(freq)))
            octave_noise = self._Perlin2D(xx, yy, cells)
            fbm = fbm + amp * octave_noise
            amp_sum += amp
            amp *= persistence
            freq *= lacunarity

        fbm = fbm / max(amp_sum, self._eps)
        fbm = bd.clip(fbm, 0.0, 1.0)

        mask = 1.0 - strength * fbm
        return bd.where(rr <= 1.0, mask, bd.ones_like(mask))

    def _Perlin2D(self, xx, yy, cells):
        """
        2D gradient-noise field over the unit disk domain using a square lattice
        spanning the full [-1, 1] x [-1, 1] map.
        Returns noise in [0, 1].
        """
        cells = int(max(1, cells))

        px = (xx + 1.0) * 0.5 * cells
        py = (yy + 1.0) * 0.5 * cells

        x0 = bd.floor(px).astype(bd.int32)
        y0 = bd.floor(py).astype(bd.int32)

        x0 = bd.clip(x0, 0, cells - 1)
        y0 = bd.clip(y0, 0, cells - 1)

        x1 = x0 + 1
        y1 = y0 + 1

        xf = px - x0
        yf = py - y0

        angles = (2.0 * bd.pi) * RNG.rand(cells + 1, cells + 1).astype(PRECISION_TYPE)
        grads = bd.stack((bd.cos(angles), bd.sin(angles)), axis=2)

        g00 = grads[y0, x0]
        g10 = grads[y0, x1]
        g01 = grads[y1, x0]
        g11 = grads[y1, x1]

        d00x = xf
        d00y = yf
        d10x = xf - 1.0
        d10y = yf
        d01x = xf
        d01y = yf - 1.0
        d11x = xf - 1.0
        d11y = yf - 1.0

        n00 = g00[:, :, 0] * d00x + g00[:, :, 1] * d00y
        n10 = g10[:, :, 0] * d10x + g10[:, :, 1] * d10y
        n01 = g01[:, :, 0] * d01x + g01[:, :, 1] * d01y
        n11 = g11[:, :, 0] * d11x + g11[:, :, 1] * d11y

        u = self._Fade(xf)
        v = self._Fade(yf)

        nx0 = n00 * (1.0 - u) + n10 * u
        nx1 = n01 * (1.0 - u) + n11 * u
        out = nx0 * (1.0 - v) + nx1 * v

        out_min = bd.min(out)
        out_max = bd.max(out)
        denom = bd.maximum(out_max - out_min, self._eps)
        return (out - out_min) / denom

    def _Fade(self, t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _PositionToUV(self, posXY):
        if self.semiDiameter is None:
            uv = bd.copy(posXY)
            valid = (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
            return uv, valid

        if self.frontVertex is None:
            center = bd.array([0.0, 0.0], dtype=PRECISION_TYPE)
        else:
            fv = bd.asarray(self.frontVertex, dtype=PRECISION_TYPE)
            center = fv[:2]

        sd = bd.asarray(self.semiDiameter, dtype=PRECISION_TYPE)
        local = (posXY - center[None, :]) / bd.maximum(sd, self._eps)
        uv = (local + 1.0) * 0.5
        radial2 = local[:, 0] * local[:, 0] + local[:, 1] * local[:, 1]
        valid = radial2 <= 1.0
        return uv, valid

    def _BilinearLookup(self, uv):
        tex = self.normalMap
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
        norm = bd.linalg.norm(out, axis=1, keepdims=True)
        return out / bd.maximum(norm, self._eps)


