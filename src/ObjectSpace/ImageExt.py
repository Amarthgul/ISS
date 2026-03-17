





import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR, Imath
import numpy as np

from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import (
    ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE,
    KNOB_DISTANCE, PRECISION_TYPE, UP_DIR, Axis, ORIGIN, RNG
)
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor, RemoveBG
from Util.Misc import Magnitude, ArrayRotate, PolarToCartesian, RectPath
from Raytracing.RayBatch import RayBatch

from .Points import PointsSource
from .Images import Image2D, Image2DFlat
from .ImageVariDepth import  Image2DVariDepth



class Image2DVariHighlightExtension(Image2DVariDepth):
    def __init__(self):
        super().__init__()


        """Threshold for highlight clipping value, with the base being in [0, 1]"""
        self.highlightClipping = 0.98

        """Maximum brightness value the extension algorithm could reach, with the base being in [0, 1]"""
        self.maxBrightness = 64

        """Minimum number of connected clipped pixels before a region is treated as a highlight candidate"""
        self.highlightMinArea = 2

        """Additional padding around a clipped blob, measured in estimated blob radii, used for shoulder fitting"""
        self.highlightFitExpand = 6.0

        """Lower and upper multiplier around the equivalent clipped radius used during profile width search"""
        self.highlightSigmaBounds = (0.75, 8.0)

        """Shape parameters tested for the generalized Gaussian highlight profile"""
        self.highlightBetaCandidates = [1.15, 1.4, 1.75, 2.1, 2.6, 3.2]

        """Power of the monotonic size-to-brightness prior; larger values make large clipped regions rise faster"""
        self.highlightSizePower = 0.85

        """Strength of the size-to-brightness prior, expressed as extra headroom above clipping"""
        self.highlightSizeStrength = 0.55

        """Maximum size driven boost allowed before hard clamping to self.maxBrightness"""
        self.highlightSizeMaxBoost = 24.0

        """Number of candidate widths sampled when fitting the generalized Gaussian profile"""
        self.highlightSigmaSamples = 19

        """Soft shoulder weighting exponent; >1 biases the fit toward the brighter inner shoulder"""
        self.highlightShoulderWeightGamma = 1.25

        """Width, in equivalent clipped radii, over which a soft blend is applied around the clipped core"""
        self.highlightBlendWidth = 0.9

        """Small numerical floor used by the reconstruction to avoid division and log singularities"""
        self.highlightEps = 1e-8




    def LoadFrom8bitRGB(self, rgbImgPath):
        """
        Override the parent's 8-bit RGB loader so PNG files can preserve opacity.
        Non-PNG formats fall back to the parent implementation.
        """
        import os
        ext = os.path.splitext(str(rgbImgPath))[1].lower()
        if ext == ".png":
            self.LoadFrom8BitPNG(rgbImgPath)
        else:
            super().LoadFrom8bitRGB(rgbImgPath)


    def LoadFrom8BitPNG(self, imgPath, premultiplyAlphaForHighlight=True):
        """
        Load an 8-bit PNG while preserving alpha in Image2DVariDepth's infrastructure.

        RGB is stored in self.rgbArray in [0,1]. Opacity is stored in self.alphaArray in [0,1].
        When premultiplyAlphaForHighlight is True, transparent bright payloads inside PNG holes
        will not be interpreted as clipped highlights by ReconstructHighlight().
        """
        imgPath = RectPath(imgPath)
        rgba_master = PIL.Image.open(imgPath).convert("RGBA")
        self._fileMaster = rgba_master

        if self.imageDimensionOverride is not None:
            new_h = int(rgba_master.height * (self.imageDimensionOverride / rgba_master.width))
            rgba_img = rgba_master.resize((self.imageDimensionOverride, new_h))
        else:
            rgba_img = rgba_master

        rgba_np = np.array(rgba_img, dtype=np.float64)
        rgb_np = rgba_np[..., :3] / 255.0
        alpha_np = rgba_np[..., 3] / 255.0

        if premultiplyAlphaForHighlight:
            rgb_np = rgb_np * alpha_np[..., None]

        self.rgbArray = self._ToBackend(rgb_np)
        self.alphaArray = self._ToBackend(alpha_np)
        self._usingEXRDirectDepth = False


    def LoadFrom8bitZ(self, zImgPath):
        """
        Load an 8-bit depth image where black is near and white is far.

        The stored zArray stays normalized in [0,1], then _UpdateDepthRange maps it into
        self.zDepthMappingRange and writes the signed distance into self.zDistance.

        If the depth image contains opacity, it is merged into self.alphaArray so both RGB and Z
        validity can cull transparent pixels later.
        """
        zImgPath = RectPath(zImgPath)
        z_master = PIL.Image.open(zImgPath)
        self._fileZ = z_master

        has_alpha = (
            z_master.mode in ("RGBA", "LA")
            or (z_master.mode == "P" and "transparency" in z_master.info)
        )

        if has_alpha:
            z_rgba = z_master.convert("RGBA")
            if self.imageDimensionOverride is not None:
                new_h = int(z_rgba.height * (self.imageDimensionOverride / z_rgba.width))
                z_rgba = z_rgba.resize((self.imageDimensionOverride, new_h))

            z_rgba_np = np.array(z_rgba, dtype=np.float64)
            z_gray = z_rgba_np[..., :3].mean(axis=2) / 255.0
            z_alpha = z_rgba_np[..., 3] / 255.0
        else:
            z_l = z_master.convert("L")
            if self.imageDimensionOverride is not None:
                new_h = int(z_l.height * (self.imageDimensionOverride / z_l.width))
                z_l = z_l.resize((self.imageDimensionOverride, new_h))
            z_gray = np.array(z_l, dtype=np.float64) / 255.0
            z_alpha = None

        self.zArray = self._ToBackend(z_gray)
        self._usingEXRDirectDepth = False

        if z_alpha is not None:
            z_alpha_bd = self._ToBackend(z_alpha)
            if self.alphaArray is None:
                self.alphaArray = z_alpha_bd
            else:
                # Keep a pixel only if both RGB and depth masks consider it visible.
                self.alphaArray = self.alphaArray * z_alpha_bd

        self._UpdateDepthRange()


    def UpdatePointSources(self):

        self._GeneratePolarPointSources()


    def ReconstructHighlight(self, normalize8Bit=True):


        if self.rgbArray is None:
            return

        arr_np = self._ToNumpy(self.rgbArray).astype(np.float64, copy=True)
        if arr_np.ndim != 3 or arr_np.shape[2] < 3:
            raise ValueError('self.rgbArray must have shape (height, width, 3/4).')

        if normalize8Bit:
            # Convert [0, 255] 8 bit image into [0, 1]
            if np.max(arr_np[..., :3]) > 1.0 + self.highlightEps:
                arr_np[..., :3] /= 255.0
                if arr_np.shape[2] > 3 and np.max(arr_np[..., 3]) > 1.0 + self.highlightEps:
                    arr_np[..., 3] /= 255.0

        rgb = arr_np[..., :3]
        alpha = arr_np[..., 3:] if arr_np.shape[2] > 3 else None

        # Detect near-white clipped highlights conservatively; hand drawn plates often encode clipped energy as neutral white.
        sat_mask = np.min(rgb, axis=2) >= self.highlightClipping
        components = self._ConnectedComponents(sat_mask)

        if len(components) == 0:
            self.rgbArray = self._ToBackend(arr_np)
            return

        y_img = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        out_rgb = rgb.copy()

        h_full, w_full = y_img.shape

        for comp in components:
            if comp.shape[0] < self.highlightMinArea:
                continue

            ys = comp[:, 0]
            xs = comp[:, 1]
            yc, xc, vals, vecs = self._EstimateEllipse(ys, xs)
            eq_radius = max(np.sqrt(comp.shape[0] / np.pi), 1.0)
            pad = int(np.ceil(eq_radius * self.highlightFitExpand + 4.0))

            y0 = max(int(np.min(ys)) - pad, 0)
            y1 = min(int(np.max(ys)) + pad + 1, h_full)
            x0 = max(int(np.min(xs)) - pad, 0)
            x1 = min(int(np.max(xs)) + pad + 1, w_full)

            local_rgb = out_rgb[y0:y1, x0:x1, :]
            local_y = y_img[y0:y1, x0:x1]
            local_core = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            local_core[ys - y0, xs - x0] = True

            r_map, s0, s1 = self._EllipticalRadiusMap(y1 - y0, x1 - x0, yc - y0, xc - x0, vals, vecs)
            fit = self._FitGeneralizedGaussian(local_y, local_core, r_map, eq_radius)
            if fit is None:
                continue

            model = fit['baseline'] + fit['amp'] * np.exp(-np.power(np.maximum(r_map * eq_radius, self.highlightEps), fit['beta']) / (fit['sigma'] ** fit['beta'] + self.highlightEps))
            model = np.minimum(model, self.maxBrightness)

            blend_outer = 1.0 + self.highlightBlendWidth
            blend_mask = (r_map >= 1.0) & (r_map <= blend_outer) & (~local_core)
            blend_w = self._SmoothStep((blend_outer - r_map) / max(self.highlightBlendWidth, self.highlightEps))
            write_mask = local_core | blend_mask

            # Only lift the highlight upward; never darken existing valid values.
            new_y = local_y.copy()
            new_y[local_core] = np.maximum(new_y[local_core], model[local_core])
            new_y[blend_mask] = np.maximum(new_y[blend_mask], local_y[blend_mask] * (1.0 - blend_w[blend_mask]) + model[blend_mask] * blend_w[blend_mask])

            scale = new_y / np.maximum(local_y, self.highlightEps)
            lifted_rgb = np.minimum(local_rgb * scale[..., None], self.maxBrightness)

            # Preserve non-highlight surroundings entirely.
            local_rgb[write_mask] = lifted_rgb[write_mask]
            out_rgb[y0:y1, x0:x1, :] = local_rgb
            y_img[y0:y1, x0:x1] = new_y

        if alpha is not None:
            out = np.concatenate([out_rgb, alpha], axis=2)
        else:
            out = out_rgb

        self.rgbArray = self._ToBackend(out)


    def VisualizeLuminosityHotMap(self, rgb=None, showColorbar=True):
        """
        Visualize luminance as a hot-map using the direct [0, max] range.

        This intentionally avoids log compression or percentile normalization so reconstructed highlights above 1.0 are clearly visible.

        :param rgb: RGB image to visualize. If None, self.rgbArray is used.
        :param showColorbar: Whether to show the colorbar on the image.

        :return: The computed luminance map.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        if rgb is None:
            rgb = self.rgbArray

        # convert backend array if necessary
        if backend_name == "cupy":
            rgb = bd.asnumpy(rgb)

        rgb = rgb.astype(np.float64)

        # compute luminance (Rec.709)
        lum = (
                0.2126 * rgb[..., 0] +
                0.7152 * rgb[..., 1] +
                0.0722 * rgb[..., 2]
        )

        vmax = lum.max()

        plt.figure(figsize=(6, 6))
        im = plt.imshow(lum, cmap="hot", vmin=0.0, vmax=vmax)

        if showColorbar:
            plt.colorbar(im)

        plt.title(f"Luminance Hot Map (max = {vmax:.3f})")
        plt.axis("off")
        plt.show()

        return lum


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _GeneratePolarPointSources(self, appendAOV=False):
        """
        Highlight-extension override of polar point generation.

        Difference from Image2DVariDepth:
        - transparent pixels are ignored immediately and never become point sources
        - simplified for 8-bit RGB / PNG workflow only
        - no EXR-specific AOV packing is performed
        """
        if self.rgbArray is None:
            self.pointSource = None
            self.jitterPerPoint = None
            return

        sampleY, sampleX, _ = self.rgbArray.shape

        # Field of view in radians
        horizontalAoV_rad = bd.deg2rad(self.horizontalAoV)
        half_horizontal = horizontalAoV_rad / 2.0

        verticalAoV_rad = horizontalAoV_rad * (sampleY / sampleX)
        half_vertical = verticalAoV_rad / 2.0

        # Pixel-center coordinates in [0,1]
        x_idx = bd.arange(sampleX, dtype=PRECISION_TYPE)
        y_idx = bd.arange(sampleY, dtype=PRECISION_TYPE)

        u = (x_idx + 0.5) / sampleX
        v = (y_idx + 0.5) / sampleY

        # Grid in (y, x) order
        U, V = bd.meshgrid(u, v, indexing="xy")

        theta_x = -half_horizontal + U * horizontalAoV_rad
        theta_y = -half_vertical + V * verticalAoV_rad

        # Distance map
        zClipDist = self._zClipDistance(
            half_horizontal, half_vertical, sampleY, sampleX, self.nearClipping
        )

        D = self.zDistance + bd.swapaxes(zClipDist, 0, 1)

        # Flatten all arrays
        coordinates = bd.stack([theta_x, theta_y, D], axis=-1).reshape(sampleY * sampleX, 3)
        colors = self.rgbArray.reshape(sampleY * sampleX, 3)
        jitter = self._AngularJitter(
            half_horizontal, half_vertical, sampleY, sampleX, D
        ).reshape(sampleY * sampleX)

        # --------------------------------------------------------------
        # Transparency cull: fully transparent pixels never become emitters
        # --------------------------------------------------------------
        if self.alphaArray is not None:
            alpha_flat = self.alphaArray.reshape(sampleY * sampleX)

            # Treat only strictly positive alpha as active.
            # This removes fully transparent PNG pixels from the point cloud.
            keep_mask = alpha_flat > 0

            coordinates = coordinates[keep_mask]
            colors = colors[keep_mask]
            jitter = jitter[keep_mask]

        self.jitterPerPoint = jitter

        points = bd.concatenate([coordinates, colors], axis=1)

        self.pointSource = PointsSource(points)
        self.pointSource.isCartesian = False
        self.pointSource.angleInRad = True


    def _ToNumpy(self, arr):
        if isinstance(arr, np.ndarray):
            return arr
        if hasattr(bd, 'asnumpy'):
            return bd.asnumpy(arr)
        return np.array(arr)


    def _ToBackend(self, arr):
        return bd.array(arr, dtype=PRECISION_TYPE)


    def _ConnectedComponents(self, mask):
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for y in range(h):
            for x in range(w):
                if (not mask[y, x]) or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                coords = []

                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    y0 = max(0, cy - 1)
                    y1 = min(h, cy + 2)
                    x0 = max(0, cx - 1)
                    x1 = min(w, cx + 2)
                    for ny in range(y0, y1):
                        for nx in range(x0, x1):
                            if mask[ny, nx] and (not visited[ny, nx]):
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                components.append(np.array(coords, dtype=np.int32))

        return components


    def _SmoothStep(self, x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)


    def _EstimateEllipse(self, ys, xs):
        yc = float(np.mean(ys))
        xc = float(np.mean(xs))

        dy = ys.astype(np.float64) - yc
        dx = xs.astype(np.float64) - xc
        if len(dx) <= 1:
            cov = np.eye(2, dtype=np.float64)
        else:
            cov = np.cov(np.stack([dx, dy], axis=0))
        cov = cov + np.eye(2, dtype=np.float64) * 1e-6

        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        return yc, xc, vals, vecs


    def _EllipticalRadiusMap(self, h, w, yc, xc, vals, vecs):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        dx = xx - xc
        dy = yy - yc

        v0x, v0y = vecs[0, 0], vecs[1, 0]
        v1x, v1y = vecs[0, 1], vecs[1, 1]

        u = dx * v0x + dy * v0y
        v = dx * v1x + dy * v1y

        s0 = max(np.sqrt(vals[0]) * 2.0, 0.5)
        s1 = max(np.sqrt(vals[1]) * 2.0, 0.5)
        r = np.sqrt((u / s0) ** 2 + (v / s1) ** 2)
        return r, s0, s1


    def _FitGeneralizedGaussian(self, y_local, core_mask, r_map, eq_radius):
        eps = self.highlightEps

        outer_for_baseline = (r_map >= (1.8 + self.highlightBlendWidth)) & (~core_mask)
        if np.any(outer_for_baseline):
            baseline = float(np.percentile(y_local[outer_for_baseline], 35.0))
        else:
            baseline = float(np.percentile(y_local[~core_mask], 20.0)) if np.any(~core_mask) else 0.0
        baseline = max(0.0, baseline)

        shoulder_mask = (~core_mask) & (r_map >= 0.95) & (r_map <= (1.8 + self.highlightFitExpand * 0.35))
        if np.count_nonzero(shoulder_mask) < 8:
            shoulder_mask = (~core_mask) & (r_map >= 0.9)
        if np.count_nonzero(shoulder_mask) < 8:
            return None

        y_obs = np.maximum(y_local[shoulder_mask] - baseline, 0.0)
        r_obs = r_map[shoulder_mask] * max(eq_radius, 1.0)

        if np.max(y_obs) <= eps:
            return None

        w = np.power(np.maximum(y_obs, eps), self.highlightShoulderWeightGamma)
        w /= (np.mean(w) + eps)

        sigma_min = max(eq_radius * self.highlightSigmaBounds[0], 0.5)
        sigma_max = max(eq_radius * self.highlightSigmaBounds[1], sigma_min + 1e-3)
        sigma_candidates = np.geomspace(sigma_min, sigma_max, self.highlightSigmaSamples)

        size_floor = 1.0 + min(self.highlightSizeStrength * (max(eq_radius, 1.0) ** self.highlightSizePower), self.highlightSizeMaxBoost)
        best = None

        for beta in self.highlightBetaCandidates:
            rp = np.power(np.maximum(r_obs, eps), beta)
            for sigma in sigma_candidates:
                model = np.exp(-rp / (sigma ** beta + eps))
                denom = np.sum(w * model * model) + eps
                amp = np.sum(w * model * y_obs) / denom
                peak = baseline + amp

                if peak < size_floor:
                    peak = size_floor
                    amp = max(peak - baseline, 0.0)

                peak = min(peak, self.maxBrightness)
                amp = max(peak - baseline, 0.0)

                pred = amp * model
                mse = np.mean(w * (pred - y_obs) ** 2)

                core_pred = baseline + amp * np.exp(-np.power(np.maximum(r_map[core_mask] * max(eq_radius, 1.0), eps), beta) / (sigma ** beta + eps))
                core_penalty = np.mean(np.square(np.maximum(1.0 - core_pred, 0.0))) if core_pred.size > 0 else 0.0

                # Encourage larger fitted widths to carry proportionally more recovered peak energy.
                size_penalty = 0.05 * (np.log(max(peak, 1.0 + eps)) - np.log(size_floor + eps)) ** 2
                score = mse + 2.5 * core_penalty + size_penalty

                if (best is None) or (score < best['score']):
                    best = {
                        'score': float(score),
                        'baseline': float(baseline),
                        'peak': float(peak),
                        'amp': float(amp),
                        'beta': float(beta),
                        'sigma': float(sigma),
                    }

        return best



class Image2DFlatHighlightExtension(Image2DVariHighlightExtension):
    def __init__(self):
        super().__init__()

        self.zDistance = FAR_DISTANCE


    def LoadFrom8bitRGB(self, rgbImgPath):
        """
        Flat-image override of 8-bit RGB loading.

        Keeps the parent's RGB loading logic, but instead of expecting a separate
        Z image, it assigns one constant depth to the entire image using
        self.zDistance, then immediately generates the point-source cloud.

        Notes
        -----
        - self.zDistance is treated as one unsigned physical distance parameter
          supplied by the user/config.
        - Internally, the system uses negative signed object-space Z, so:
              self.zArray    : positive depth magnitude per pixel
              self.zDistance : negative signed depth used by the geometry/ray system
        """

        # Reuse parent RGB loading behavior as much as possible.
        super().LoadFrom8bitRGB(rgbImgPath)

        if self.rgbArray is None:
            return

        h, w, _ = self.rgbArray.shape

        # Take the user-facing flat distance and convert it into the internal convention.
        flat_depth = bd.abs(bd.array(self.zDistance, dtype=PRECISION_TYPE)).reshape(-1)[0]

        # Store a constant per-pixel zArray, and signed zDistance for the engine.
        self.zArray = bd.ones((h, w), dtype=PRECISION_TYPE) * flat_depth
        self.zDistance = -flat_depth

        # 8-bit flat RGB path is not EXR-direct-depth.
        self._usingEXRDirectDepth = False

        self.UpdatePointSources()


    def ReceiveAndEmitTowards(self, targets, incidents: RayBatch = None, sampleCount: int = 64):
        """
        Receive an incident RayBatch, probabilistically cull the rays that hit this flat
        image plane according to opacity/alpha, then merge survivors with newly emitted rays.

        Compared with Image2DVariDepth:
        - uses one closed-form ray-plane intersection at z = self.zDistance
        - keeps the rest of the mapping/vectorization in array form
        """

        emitted = self.EmitSamplesToward(targets, sampleCount)

        if (incidents is None) or (incidents.IsNoneType()):
            # When this is the furthest layer
            return emitted

        # If there is no opacity mask, nothing to cull.
        if self.alphaArray is None:
            return incidents.Copy().Merge(emitted)

        alpha_eps = 1e-6
        if not bd.any(self.alphaArray > alpha_eps):
            return incidents.Copy().Merge(emitted)

        positions = incidents.Position()     # (N, 3)
        directions = incidents.Direction()   # (N, 3)

        ox = positions[:, 0]
        oy = positions[:, 1]
        oz = positions[:, 2]

        dx = directions[:, 0]
        dy = directions[:, 1]
        dz = directions[:, 2]

        # Single flat plane at z = self.zDistance (negative in object space)
        z_plane = -bd.abs(bd.array(self.zDistance, dtype=PRECISION_TYPE)).reshape(-1)[0]

        dz_eps = 1e-8
        valid_dz = bd.abs(dz) > dz_eps

        # Closed-form ray-plane intersection: o + t d, solve for z = z_plane
        t_hit = (z_plane - oz) / dz
        hit_forward = t_hit > 0
        valid_hit = valid_dz & hit_forward

        # If no ray reaches the plane, pass all through
        if not bd.any(valid_hit):
            return incidents.Copy().Merge(emitted)

        # Hit points
        px = ox + t_hit * dx
        py = oy + t_hit * dy
        pz = oz + t_hit * dz   # numerically should equal z_plane

        # --------------------------------------------------------------
        # World -> image mapping (same convention as parent class)
        # --------------------------------------------------------------
        H, W = self.alphaArray.shape

        horizontalAoV_rad = bd.deg2rad(self.horizontalAoV)
        half_horizontal = horizontalAoV_rad / 2.0

        verticalAoV_rad = horizontalAoV_rad * (H / W)
        half_vertical = verticalAoV_rad / 2.0

        inside = pz < 0.0

        #theta_x = bd.arctan2(px, -pz)
        #theta_y = bd.arctan2(py, -pz)
        theta_x = bd.arctan2(-px, -pz)
        theta_y = bd.arctan2(-py, -pz)

        u = (theta_x + half_horizontal) / horizontalAoV_rad
        v = (theta_y + half_vertical) / verticalAoV_rad

        inside = inside & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        valid_hit = valid_hit & inside

        # Rays that miss the image footprint are not culled by this layer
        if not bd.any(valid_hit):
            return incidents.Copy().Merge(emitted)

        x_img = u * (W - 1)
        y_img = v * (H - 1)

        # --------------------------------------------------------------
        # Bilinear alpha lookup
        # --------------------------------------------------------------
        x0 = bd.floor(x_img).astype(bd.int64)
        y0 = bd.floor(y_img).astype(bd.int64)

        x0 = bd.clip(x0, 0, W - 1)
        y0 = bd.clip(y0, 0, H - 1)

        x1 = bd.clip(x0 + 1, 0, W - 1)
        y1 = bd.clip(y0 + 1, 0, H - 1)

        fx = x_img - x0
        fy = y_img - y0

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        # alpha_img = bd.flip(self.alphaArray, axis=(0, 1))

        a00 = self.alphaArray[y0, x0]
        a10 = self.alphaArray[y0, x1]
        a01 = self.alphaArray[y1, x0]
        a11 = self.alphaArray[y1, x1]

        alpha_local = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11
        alpha_local = bd.clip(alpha_local, 0.0, 1.0)

        # Only rays that truly intersect the plane footprint receive probabilistic culling.
        alpha_local = bd.where(valid_hit, alpha_local, bd.zeros_like(alpha_local))
        alpha_local = bd.where(alpha_local > alpha_eps, alpha_local, bd.zeros_like(alpha_local))

        # --------------------------------------------------------------
        # Opacity as probability-to-drop
        # --------------------------------------------------------------
        rnd = RNG.rand(len(alpha_local))
        keep_mask = rnd >= alpha_local

        through = RayBatch(incidents.value[keep_mask])

        return through.Merge(emitted)

