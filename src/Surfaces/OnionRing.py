from .SurfaceModulator import SurfaceModulator
from Util.Backend import backend as bd
from Util.Globals import RNG, PRECISION_TYPE


class OnionRing(SurfaceModulator):

    def __init__(self, ringCount=38, disturbance=0.5):
        super().__init__()

        """Total count of ring layers."""
        self.ringCount = ringCount

        """Max amount of positional offset that could happen to a ring. 0 being no offset, 1 could make it become tangent with the border of the neighboring ring at some point. """
        self.disturbance = disturbance

        self.normalMap = None
        self.heightMap = None

        self._normalBlend = 0.0001 # 1 being replacement, 0 being no modification

        self._normalSinSag = 0.5 # Blend each ring profile between a sine shape and a spherical shape. 1 for pure sine, 0 for pure sphere/circular.

        self._ringHeight = 0.5 # Height of the profile. When set to 1, the profile has a full height equal to half of the ideal per-ring width, calculated as 1/(ringCount*2). When set to 0, then there is nothing and no perturbation will happen.

        self._normalStrength = 2.0
        self._disturbanceHarmonics = 3
        self._disturbanceSamples = None
        self._eps = 1e-12


    def Generate(self):

        if self._ringHeight <= 0:
            self.heightMap = bd.zeros((self.mapRes, self.mapRes), dtype=PRECISION_TYPE)
            self.normalMap = self._NeutralNormalMap(self.mapRes)
            return self

        lin = bd.linspace(-1.0, 1.0, self.mapRes, dtype=PRECISION_TYPE)
        xx, yy = bd.meshgrid(lin, lin, indexing='xy')
        rr = bd.sqrt(xx * xx + yy * yy)
        theta = bd.arctan2(yy, xx)

        # Disturb the effective radius slightly per angle so each ring is not perfectly circular.
        disturbed_r = self._DisturbedRadius(rr, theta, self.ringCount)

        # Build height profile ring by ring.
        height = bd.zeros_like(disturbed_r)
        ringWidth = 1.0 / max(self.ringCount, 1)
        amp = self._ringHeight * (ringWidth * 0.5)

        signs = self._RingSign()
        for idx in range(self.ringCount):
            inner = idx * ringWidth
            outer = (idx + 1) * ringWidth
            local = (disturbed_r - inner) / max(outer - inner, self._eps)
            in_ring = (local >= 0.0) & (local <= 1.0)

            ring_profile = self._RingProfile(local)
            sign = signs[idx]
            height = bd.where(in_ring, height + sign * amp * ring_profile, height)

        # Clamp outside unit disk to neutral.
        height = bd.where(rr <= 1.0, height, 0.0)
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

        # Increase normal strength impact.
        nx = -dx * self._normalStrength
        ny = -dy * self._normalStrength
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

            # Ordinary tangent-space display heavily compresses subtle XY departures.
            # Exaggerate only XY for visualization while preserving sign.
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

        # Symmetric half-sine bump.
        sine_prof = bd.sin(bd.pi * l)

        # Circular-cap-like profile using a normalized semicircle over [-1, 1].
        x = 2.0 * l - 1.0
        circ_prof = bd.sqrt(bd.maximum(0.0, 1.0 - x * x))

        w = float(min(max(self._normalSinSag, 0.0), 1.0))
        return (w * sine_prof) + ((1.0 - w) * circ_prof)


    def _DisturbedRadius(self, rr, theta, ringCount):
        """
        Apply a small angular modulation to the radial coordinate.
        disturbance=1 means the local boundary can swing by up to half a ring width.
        """
        if self.disturbance <= 0:
            return rr

        ringWidth = 1.0 / max(ringCount, 1)
        maxShift = 0.5 * ringWidth * float(self.disturbance)

        # Low-frequency deterministic random field over angle.
        if self._disturbanceSamples is None or len(self._disturbanceSamples) != self._disturbanceHarmonics:
            self._disturbanceSamples = (RNG.rand(self._disturbanceHarmonics) * 2.0 - 1.0).astype(PRECISION_TYPE)

        field = bd.zeros_like(theta)
        for i in range(self._disturbanceHarmonics):
            freq = i + 1
            phase = (2.0 * bd.pi) * self._disturbanceSamples[i]
            amp = self._disturbanceSamples[i] / freq
            field = field + amp * bd.sin(freq * theta + phase)

        denom = bd.max(bd.abs(field))
        denom = bd.maximum(denom, self._eps)
        field = field / denom

        return rr + (maxShift * field)


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
