

from .Standard import StdImager
from Util.Globals import Channels, NEAR_ZERO
from Util.Backend import backend as bd
from Util.ColorPDF import ColorPDF



class Film(StdImager):
    def __init__(self, sr=ColorPDF(), bfd = 42, w = 36, h = 24, horiPx = 1920):
        super().__init__(bfd = bfd, w = w, h = h, horiPx = horiPx)

        # A color PDF component to model the spectral response
        self.colorPDF = sr

        """Distance from image plane (BFD) to the backplate, needed for halation."""
        self.backPlateDistance=.05


        """When supplied, this will be used for more accurate integral"""
        self.emissionPDF = None


        """Dye - spectral response pairs. When key and value are the same it produces an normal image. If not, it creates some Lomography film emulsion displacement style"""
        self.dyeSpectralPairs = {
            Channels.R: Channels.R,
            Channels.G: Channels.G,
            Channels.B: Channels.B,
        }

        """Axial order of the emulsion layers, by default the yellow dye (blue color) is at the front and cyan dye (red color) is at the back. Order pointing towards positive Z direction."""
        self.emulsionOrder = [Channels.B, Channels.G, Channels.R]

        self.dyeCloudPhotonScale = 4096
        self.dyeCloudStrength = 0.1
        self.dyeCloudCorrelated = True
        self.dyeCloudCorrRadiusPxMin = 1
        self.dyeCloudCorrRadiusPxExtra = 2
        self.dyeCloudCorrPasses = 2
        self.dyeCloudMidGray = 0.18
        self.dyeCloudColorBalance = 0.1 # Color 1 <---> 0 Monochrome


        # ==================================================================
        # ============================= Grain ==============================

        self.bleachByPassRatio = 0.55
        self.silverGrainsPerMP = 1280000.0
        self.grainSizeMu = [.45, .65, .45]
        self.grainSizeSigma = [.75, .75, .75]
        self.silverPhotonScale = 2048.0

        self.silverThresh0 = 2.0
        self.silverThreshR0 = 1.5
        self.silverThreshBeta = 1.0
        self.silverThreshJitter = 0.75

        self.silverGrowGamma = 0.5
        self.silverAmax = 1.0

        self.silverPowerAlpha = 1.0
        self.silverEdgeSoftPx = 1.25

        self.silverGrainStrength = 0.35
        self.silverGrainColorBalance = 0.5  # 0 mono, 1 RGB

        self.silverGrainClump = True
        self.silverGrainClumpRadiusPx = 1

        self.silverSensSigma = 0.25

        self.silverPolygonalEdges = True
        # softness in "power distance" units (≈ pixels^2). Larger => softer border transition.
        self.silverPolygonalEdgeSoft = 4.0

        # Optional: mix back a little bit of radial shaping (0 = fully polygonal, 1 = fully radial/circular)
        self.silverRadialEdgeMix = 0.0

        self.GrainGridR = None
        self.GrainGridG = None
        self.GrainGridB = None
        self._grain_grid_H = -1
        self._grain_grid_W = -1


        # ==================================================================
        # ============================= Curve ==============================
        # Enable/disable the curve stage.
        self.curveEnable = True

        # Middle gray anchor (linear). If your pipeline is calibrated, keep this at 0.18; otherwise expose so that your intended midtone sits here.
        self.curveMidGray = 0.5

        # Exposure offset in stops. Positive => brighter / faster (moves curve left).
        self.curveExposureOffsetStops = -2

        # Midtone contrast control. Larger => steeper straight-line region.
        self.curveContrast = 1

        # Richards asymmetry parameter (nu). 1.0 => standard logistic.
        # >1 biases toward a longer toe and a more compressed shoulder;
        # <1 biases toward a snappier toe and longer shoulder.
        self.curveAsymmetry = 1.0

        # Base+fog density and maximum density. These define the transmittance range:
        #   Tmax = 10^(-Dmin), Tmin = 10^(-Dmax)
        self.curveDmin = 0.10
        self.curveDmax = 2.80

        # Optional soft highlight compression before the log domain.
        # If curveSoftClipEnable is True, values above "white" roll off smoothly:
        #   E' = white * E / (E + white)
        self.curveSoftClipEnable = True

        # Define "white" either directly in linear units, or as stops above mid-gray.
        # If curveWhitePoint is None, curveWhitePointStops will be used.
        self.curveWhitePoint = None
        self.curveWhitePointStops = 10.0  # white ~= midGray * 2^stops

        # Per-channel optional trims (stops and contrast multipliers).
        self.curveChannelOffsetStops = [0.0, .25, .5]
        self.curveChannelContrastMul = [1.0, 1.0, 1.0]

        # Numerical stability.
        self.curveEps = 1e-12

        # ==================================================================
        # ============================ Halation ============================

        self.halationIsOptical = False


    def UpdateGrid(self):

        self._GenerateGrains()


    def DensityCurve(self, rgb_image):
        if not getattr(self, 'curveEnable', True):
            return rgb_image

        rgb = bd.asarray(rgb_image)

        # Accept (H,W) or (H,W,1) by promoting to RGB for consistency.
        if rgb.ndim == 2:
            rgb = rgb[:, :, None]
        if rgb.shape[-1] == 1:
            rgb = bd.repeat(rgb, 3, axis=-1)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise ValueError(f"Expect rgb_image of shape (H, W, 3). Got {rgb.shape}")

        rgb = rgb[..., :3]
        rgb = bd.maximum(rgb, 0)

        mid = max(self.curveMidGray, self.curveEps)

        # Resolve soft-clip white point.
        white = getattr(self, 'curveWhitePoint', None)
        if white is None:
            wp_stops = float(getattr(self, 'curveWhitePointStops', 7.0))
            white = mid * (2.0 ** wp_stops)
        white = float(white)
        white = max(white, self.curveEps)

        if getattr(self, 'curveSoftClipEnable', True):
            # Softly limit outliers / fireflies without a hard clamp.
            rgb = (white * rgb) / (rgb + white)

        # log2 helper (numpy/cupy both provide log2, but keep a fallback).
        if hasattr(bd, 'log2'):
            s = bd.log2((rgb + self.curveEps) / mid)
        else:
            s = bd.log((rgb + self.curveEps) / mid) / bd.log(2.0)

        # Global exposure offset in stops.
        s = s + float(getattr(self, 'curveExposureOffsetStops', 0.0))

        # Per-channel trims.
        ch_off = getattr(self, 'curveChannelOffsetStops', [0.0, 0.0, 0.0])
        if ch_off is not None:
            try:
                off = bd.asarray(ch_off, dtype=rgb.dtype).reshape((1, 1, 3))
                s = s + off
            except Exception:
                # If backend dtype/shape handling fails for any reason, ignore trims.
                pass

        # Richards logistic in stops-space.
        k0 = float(getattr(self, 'curveContrast', 1.15))
        nu = float(getattr(self, 'curveAsymmetry', 1.0))
        nu = max(nu, 1e-6)

        # Optional per-channel contrast multipliers.
        k_mul = getattr(self, 'curveChannelContrastMul', [1.0, 1.0, 1.0])
        try:
            km = bd.asarray(k_mul, dtype=rgb.dtype).reshape((1, 1, 3))
        except Exception:
            km = 1.0

        k = k0 * km

        # Center point s0=0 (mid-gray). Any shift is handled by exposure offsets above.
        x = -k * s
        # y = (1 + nu * exp(x))^(-1/nu)
        y = bd.power(1.0 + nu * bd.exp(x), -1.0 / nu)

        # Map to density, then to transmittance.
        dmin = float(getattr(self, 'curveDmin', 0.10))
        dmax = float(getattr(self, 'curveDmax', 2.60))
        if dmax < dmin:
            dmax, dmin = dmin, dmax

        D = dmin + (dmax - dmin) * y

        # Transmittance T = 10^(-D)
        T = bd.power(10.0, -D)

        # Clamp to a sane range for downstream grading.
        T = bd.clip(T, 0.0, 1.0)

        return T


    def ApplyGrainAndNoise(self, rgb_image):
        """Hook for image-domain grain/noise.

        Default: identity.
        Must return rgb_image.
        """

        dye = self._DyeCloud(rgb_image)
        silver = self._ApplySilverGrain(rgb_image)

        return silver * self.bleachByPassRatio + dye * (1.0 - self.bleachByPassRatio)


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ApplyColorPDF(self, intersectRayBatch, rayPos, radiant, wavelength, chan):
        """Film spectral weighting hook.

        StdImager deposits energy into the per-channel grids using `chan` (0/1/2).
        Film additionally weights each ray by the film spectral response:

            weight = self.colorPDF.SpectralResponse(wavelength, resp_channel)

        Where `resp_channel` can be remapped from the dye/deposit channel using
        `self.dyeSpectralPairs` (Lomography-like emulsion displacement / channel cross-talk).

        If `self.emissionPDF` is provided, an importance-correction term is applied:

            weight *= 1 / (self.emissionPDF.SpectralResponse(wavelength, emit_channel) + NEAR_ZERO)

        Parameters
        ----------
        intersectRayBatch : RayBatch
            Full incident raybatch (available for subclasses; not required here).
        rayPos : (N,2) int array
            Pixel coordinates (already in-bounds).
        radiant : (N,) array
            Per-ray radiance to be deposited (already filtered to hit + in-bounds).
        wavelength : (N,) array
            Per-ray wavelength.
        chan : (N,) int array
            Deposit channel (dye channel); 0=R, 1=G, 2=B.

        Returns
        -------
        radiant : (N,) array
            Weighted radiance.
        """

        # --- map dye/deposit channel -> response channel (defaults to identity) ---
        def _as_int(ch):
            # Dict keys/values may be Channels enum members or raw ints
            return int(getattr(ch, "value", ch))

        mapR = _as_int(self.dyeSpectralPairs[Channels.R])
        mapG = _as_int(self.dyeSpectralPairs[Channels.G])
        mapB = _as_int(self.dyeSpectralPairs[Channels.B])

        mapAry = bd.asarray([mapR, mapG, mapB]).astype(int)  # shape (3,)
        chan_resp = mapAry[chan]  # shape (N,)

        # --- film spectral response weight ---
        film_w = self.colorPDF.SpectralResponse(wavelength, chan_resp)

        # Optional importance correction: divide by emission sampling pdf-like response
        if self.emissionPDF is not None:
            emit_w = self.emissionPDF.SpectralResponse(wavelength, chan)
            film_w = film_w / (emit_w + NEAR_ZERO)

        return radiant * film_w


    def _ApplyHalation(self, intersectRayBatch, rgb_image):
        """Optional halation hook.

        """

        if self.halationIsOptical:
            return self._HalationOptical(intersectRayBatch)
        else:
            return self._Halation2D(rgb_image)



    def _ApplySilverGrain(self, rgb_image):
        """Silver-halide style grain: discrete grains + power Voronoi + exposure-driven development."""
        rgb = bd.asarray(rgb_image)

        eps = 1e-12
        rgb = bd.maximum(rgb, 0)
        H, W = rgb.shape[0], rgb.shape[1]

        if (self.GrainGridR is None) or (self.GrainGridG is None) or (self.GrainGridB is None):
            self._GenerateGrains(rgb_image)
        if int(self._grain_grid_H) != int(H) or int(self._grain_grid_W) != int(W):
            self._GenerateGrains(rgb_image)

        gR = self._develop_grains_from_exposure(rgb[..., 0], self.GrainGridR)
        gG = self._develop_grains_from_exposure(rgb[..., 1], self.GrainGridG)
        gB = self._develop_grains_from_exposure(rgb[..., 2], self.GrainGridB)

        fieldR = self._rasterize_power_voronoi(gR, H, W)
        fieldG = self._rasterize_power_voronoi(gG, H, W)
        fieldB = self._rasterize_power_voronoi(gB, H, W)

        if self.silverGrainClump:
            rad = int(self.silverGrainClumpRadiusPx)
            if rad > 0:
                fieldR = bd.maximum(fieldR, self._box_blur_2d(fieldR, rad) * 0.85)
                fieldG = bd.maximum(fieldG, self._box_blur_2d(fieldG, rad) * 0.85)
                fieldB = bd.maximum(fieldB, self._box_blur_2d(fieldB, rad) * 0.85)

        eta = max(0.0, min(1.0,  self.silverGrainColorBalance))
        if eta < 1.0:
            mono = fieldR * 0.2126 + fieldG * 0.7152 + fieldB * 0.0722
            fieldR = (1.0 - eta) * mono + eta * fieldR
            fieldG = (1.0 - eta) * mono + eta * fieldG
            fieldB = (1.0 - eta) * mono + eta * fieldB

        # zero-mean per channel so exposure doesn't drift
        fieldR = fieldR - bd.mean(fieldR)
        fieldG = fieldG - bd.mean(fieldG)
        fieldB = fieldB - bd.mean(fieldB)

        k = self.silverGrainStrength
        k = max(0.0, k)

        logR = bd.log(rgb[..., 0] + eps) + k * fieldR
        logG = bd.log(rgb[..., 1] + eps) + k * fieldG
        logB = bd.log(rgb[..., 2] + eps) + k * fieldB

        out = bd.stack((bd.exp(logR), bd.exp(logG), bd.exp(logB)), axis=-1)
        return bd.maximum(out, 0)


    def _GenerateGrains(self, im=None):

        if im is not None:
            H = im.shape[0]
            W = im.shape[1]
        else:
            H = int(self.verticalPx)
            W = int(self.horizontalPx)

        self._grain_grid_H = H
        self._grain_grid_W = W

        gpm = self.silverGrainsPerMP
        K = int(max(1024, round(gpm * (H * W / 1e6))))

        mus = self.grainSizeMu
        sigs = self.grainSizeSigma

        def _make_layer(mu, sigma):
            x = bd.random.random(K) * (W - 1)
            y = bd.random.random(K) * (H - 1)

            m = max(0.25, float(mu))
            s = max(0.05, float(sigma))
            phi = bd.sqrt(bd.log(1.0 + (s * s) / (m * m)))
            mu_l = bd.log(m) - 0.5 * (phi * phi)
            r = bd.exp(mu_l + phi * bd.random.standard_normal(K))
            r = bd.clip(r, 0.35, max(0.75, 6.0 * m))

            sens_sigma = self.silverSensSigma
            sens = bd.exp(sens_sigma * bd.random.standard_normal(K))
            sens = bd.clip(sens, 0.25, 4.0)

            a = bd.zeros(K, dtype=bd.float32)
            return bd.stack((x.astype(bd.float32),
                             y.astype(bd.float32),
                             r.astype(bd.float32),
                             sens.astype(bd.float32),
                             a), axis=1)

        self.GrainGridR = _make_layer(mus[0], sigs[0])
        self.GrainGridG = _make_layer(mus[1], sigs[1])
        self.GrainGridB = _make_layer(mus[2], sigs[2])


    def _bilinear_sample(self, img2d, x, y):
        H, W = img2d.shape
        x0 = bd.floor(x).astype(int)
        y0 = bd.floor(y).astype(int)
        x1 = bd.clip(x0 + 1, 0, W - 1)
        y1 = bd.clip(y0 + 1, 0, H - 1)

        x0 = bd.clip(x0, 0, W - 1)
        y0 = bd.clip(y0, 0, H - 1)

        fx = (x - x0).astype(bd.float32)
        fy = (y - y0).astype(bd.float32)

        Ia = img2d[y0, x0]
        Ib = img2d[y0, x1]
        Ic = img2d[y1, x0]
        Id = img2d[y1, x1]

        return (Ia * (1.0 - fx) * (1.0 - fy) +
                Ib * fx * (1.0 - fy) +
                Ic * (1.0 - fx) * fy +
                Id * fx * fy)


    def _develop_grains_from_exposure(self, exposure2d, grid):
        exp2d = bd.asarray(exposure2d)
        exp2d = bd.maximum(exp2d, 0)

        x = grid[:, 0]
        y = grid[:, 1]
        r = grid[:, 2]
        sens = grid[:, 3]

        E = self._bilinear_sample(exp2d, x, y)
        area = 3.14159265 * r * r
        H_i = E * area

        photon_scale = float(getattr(self, "silverPhotonScale", 2048.0))
        lam = bd.maximum(H_i * photon_scale * sens, 0)

        try:
            n = bd.random.poisson(lam)
        except Exception:
            z = bd.random.standard_normal(lam.shape)
            n = bd.maximum(lam + bd.sqrt(bd.maximum(lam, 1e-12)) * z, 0)

        T0 = float(getattr(self, "silverThresh0", 2.0))
        r0 = float(getattr(self, "silverThreshR0", 1.5))
        beta = float(getattr(self, "silverThreshBeta", 1.0))
        jitter = float(getattr(self, "silverThreshJitter", 0.75))

        Ti = T0 * bd.power(bd.maximum(r / r0, 1e-6), beta)
        if jitter > 0:
            Ti = Ti + jitter * bd.random.standard_normal(Ti.shape)

        gamma = float(getattr(self, "silverGrowGamma", 0.9))
        amax = float(getattr(self, "silverAmax", 1.0))
        dn = bd.maximum(n - Ti, 0)
        a = amax * (1.0 - bd.exp(-gamma * dn))
        a = bd.clip(a, 0.0, amax)

        grid[:, 4] = a.astype(bd.float32)
        return grid


    def _shift_fill(self, arr, dy: int, dx: int, fill_value):
        H, W = arr.shape
        out = bd.full((H, W), fill_value, dtype=arr.dtype)

        y0_src = max(0, -dy)
        y1_src = min(H, H - dy)
        x0_src = max(0, -dx)
        x1_src = min(W, W - dx)

        y0_dst = max(0, dy)
        y1_dst = y0_dst + (y1_src - y0_src)
        x0_dst = max(0, dx)
        x1_dst = x0_dst + (x1_src - x0_src)

        if (y1_src > y0_src) and (x1_src > x0_src):
            out[y0_dst:y1_dst, x0_dst:x1_dst] = arr[y0_src:y1_src, x0_src:x1_src]
        return out


    def _rasterize_power_voronoi(self, grid, H: int, W: int):
        """Jump Flood power-Voronoi rasterization to get crisp grain cells."""

        seed_id = bd.full((H, W), -1, dtype=bd.int32)
        alpha = self.silverPowerAlpha
        edge_soft = self.silverEdgeSoftPx

        seed_x = bd.full((H, W), -1.0, dtype=bd.float32)
        seed_y = bd.full((H, W), -1.0, dtype=bd.float32)
        seed_r = bd.full((H, W), 0.0, dtype=bd.float32)
        seed_a = bd.full((H, W), 0.0, dtype=bd.float32)
        seed_w = bd.full((H, W), 0.0, dtype=bd.float32)

        xi = bd.clip(bd.rint(grid[:, 0]).astype(int), 0, W - 1)
        yi = bd.clip(bd.rint(grid[:, 1]).astype(int), 0, H - 1)

        a = grid[:, 4]
        r = grid[:, 2]
        w = alpha * (r * r)

        # Reduce duplicate seeds on CPU (K-sized only)
        lin = yi * W + xi
        order = bd.argsort(lin)
        lin_s = lin[order]
        x_s = grid[:, 0][order]
        y_s = grid[:, 1][order]
        r_s = r[order]
        a_s = a[order]
        w_s = w[order]

        use_cpu = False
        try:
            if hasattr(lin_s, "get"):
                lin_cpu = lin_s.get()
                x_cpu = x_s.get()
                y_cpu = y_s.get()
                r_cpu = r_s.get()
                a_cpu = a_s.get()
                w_cpu = w_s.get()
                use_cpu = True
        except Exception:
            use_cpu = False

        if use_cpu:
            import numpy as _np
            key = _np.lexsort((-r_cpu, -a_cpu, lin_cpu))
            lin2 = lin_cpu[key]
            x2 = x_cpu[key]
            y2 = y_cpu[key]
            r2 = r_cpu[key]
            a2 = a_cpu[key]
            w2 = w_cpu[key]
            first = _np.concatenate(([True], lin2[1:] != lin2[:-1]))
            lin_u = lin2[first]
            x_u = x2[first]
            y_u = y2[first]
            r_u = r2[first]
            a_u = a2[first]
            w_u = w2[first]

            lin_u_b = bd.asarray(lin_u)
            yy_u = (lin_u_b // W).astype(int)
            xx_u = (lin_u_b - yy_u * W).astype(int)

            seed_id[yy_u, xx_u] = bd.asarray(lin_u).astype(bd.int32)

            seed_x[yy_u, xx_u] = bd.asarray(x_u)
            seed_y[yy_u, xx_u] = bd.asarray(y_u)
            seed_r[yy_u, xx_u] = bd.asarray(r_u)
            seed_a[yy_u, xx_u] = bd.asarray(a_u)
            seed_w[yy_u, xx_u] = bd.asarray(w_u)
        else:
            seed_x[yi, xi] = grid[:, 0].astype(bd.float32)
            seed_y[yi, xi] = grid[:, 1].astype(bd.float32)
            seed_r[yi, xi] = r.astype(bd.float32)
            seed_a[yi, xi] = a.astype(bd.float32)
            seed_w[yi, xi] = w.astype(bd.float32)

            lin = yi * W + xi
            seed_id[yi, xi] = lin.astype(bd.int32)

        yy = bd.arange(H, dtype=bd.float32)[:, None]
        xx = bd.arange(W, dtype=bd.float32)[None, :]
        inf = bd.asarray(1e30, dtype=bd.float32)

        def _power_dist(sx, sy, sw):
            dx = xx - sx
            dy = yy - sy
            d = dx * dx + dy * dy - sw
            return bd.where(sx < 0, inf, d)

        best_d = _power_dist(seed_x, seed_y, seed_w)

        m = int(max(H, W))
        step = 1
        while step < m:
            step <<= 1
        step >>= 1

        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]

        while step >= 1:
            for oy, ox in offsets:
                dy = oy * step
                dx = ox * step

                cx = self._shift_fill(seed_x, dy, dx, -1.0)
                cy = self._shift_fill(seed_y, dy, dx, -1.0)
                cr = self._shift_fill(seed_r, dy, dx, 0.0)
                ca = self._shift_fill(seed_a, dy, dx, 0.0)
                cw = self._shift_fill(seed_w, dy, dx, 0.0)

                cd = _power_dist(cx, cy, cw)
                mask = cd < best_d

                best_d = bd.where(mask, cd, best_d)
                seed_x = bd.where(mask, cx, seed_x)
                seed_y = bd.where(mask, cy, seed_y)
                seed_r = bd.where(mask, cr, seed_r)
                seed_a = bd.where(mask, ca, seed_a)
                seed_w = bd.where(mask, cw, seed_w)

                cid = self._shift_fill(seed_id, dy, dx, -1)
                seed_id = bd.where(mask, cid, seed_id)
            step //= 2

        poly_on = bool(getattr(self, "silverPolygonalEdges", True))
        poly_soft = float(getattr(self, "silverPolygonalEdgeSoft", 4.0))
        rad_mix = float(getattr(self, "silverRadialEdgeMix", 0.0))
        rad_mix = max(0.0, min(1.0, rad_mix))

        # --- polygonal edge (distance-to-border proxy) ---
        poly_edge = None
        if poly_on:
            second = inf
            # use 8-neighborhood at 1px to find competing seeds across borders
            for oy, ox in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1), (0, 1),
                           (1, -1), (1, 0), (1, 1)]:
                nx = self._shift_fill(seed_x, oy, ox, -1.0)
                ny = self._shift_fill(seed_y, oy, ox, -1.0)
                nw = self._shift_fill(seed_w, oy, ox, 0.0)
                nid = self._shift_fill(seed_id, oy, ox, -1)

                nd = _power_dist(nx, ny, nw)
                cand = bd.where(nid != seed_id, nd, inf)
                second = bd.minimum(second, cand)

            delta = bd.maximum(second - best_d, 0.0)  # 0 at borders, larger in interior
            t = bd.clip(delta / (poly_soft + 1e-12), 0.0, 1.0)
            poly_edge = t * t * (3.0 - 2.0 * t)  # smoothstep

        # --- optional radial mask (if you still want some roundness) ---
        rad_edge = None
        if edge_soft > 0 and rad_mix > 0.0:
            dx = xx - seed_x
            dy = yy - seed_y
            dist = bd.sqrt(dx * dx + dy * dy + 1e-12)
            rr = bd.maximum(seed_r, 1e-6)
            t = bd.clip((rr - dist) / edge_soft, 0.0, 1.0)
            rad_edge = t * t * (3.0 - 2.0 * t)

        # Choose edge
        if poly_edge is None:
            # fallback: old behavior
            if edge_soft > 0:
                return seed_a * rad_edge
            return seed_a

        edge = poly_edge
        if rad_edge is not None:
            edge = (1.0 - rad_mix) * edge + rad_mix * rad_edge

        return seed_a * edge


    def _DyeCloud(self, rgb_image):
        """Apply a physically-motivated grain/noise model to a linear RGB radiance image.

        Goals:
          - Photon shot noise per channel (Poisson)
          - Noise intensity + size loosely coupled to exposure
          - Optional spatial correlation ("clumpy" grain)

        Assumptions:
          - `rgb_image` is linear scene-referred radiance / exposure-like signal.
          - Uses backend `bd` (numpy/cupy) so this runs on CPU/GPU.
        """

        rgb = bd.asarray(rgb_image)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            return rgb_image

        eps = 1e-12
        rgb = bd.maximum(rgb, 0)

        # ----------------------------------------------------------------------------------
        # Exposure proxy (for coupling strength + correlation)
        # ----------------------------------------------------------------------------------
        # luminance in linear space
        lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722

        mid = 0.18
        mean_lum = bd.mean(lum)
        exp_norm = mean_lum / (mean_lum + (mid + eps))  # in [0,1)

        # ----------------------------------------------------------------------------------
        # 1) Photon shot noise (Poisson per channel)
        # ----------------------------------------------------------------------------------

        lam = rgb * self.dyeCloudPhotonScale
        lam = bd.maximum(lam, 0)

        try:
            counts = bd.random.poisson(lam)
            rgb_noisy = counts / self.dyeCloudPhotonScale
        except Exception:
            # Fallback: Gaussian approximation if poisson isn't available
            n = bd.random.standard_normal(lam.shape)
            rgb_noisy = (lam + bd.sqrt(bd.maximum(lam, eps)) * n) / self.dyeCloudPhotonScale
            rgb_noisy = bd.maximum(rgb_noisy, 0)

        # ----------------------------------------------------------------------------------
        # 2) Optional self.grainCorrelated grain texture (multiplicative)
        # ----------------------------------------------------------------------------------
        if self.dyeCloudStrength > 0:
            shadow_pow = float(getattr(self, "grain_shadow_power", 1.25))
            shadow_weight = bd.power(1.0 - (lum / (lum + (mid + eps))), shadow_pow)
            strength_map = self.dyeCloudStrength * (0.25 + 0.75 * shadow_weight)

            # exp_norm may be numpy/cupy scalar; convert carefully
            try:
                exp_n = float(exp_norm.get())
            except Exception:
                exp_n = float(exp_norm)

            r = int(round(self.dyeCloudCorrRadiusPxMin + (1.0 - exp_n) * self.dyeCloudCorrRadiusPxExtra))
            r = max(0, r)

            def _box_blur_2d(x2d, radius_px: int):
                """Fast box blur via integral image. Works for numpy/cupy."""
                if radius_px <= 0:
                    return x2d
                k = int(radius_px)
                ks = 2 * k + 1
                try:
                    xpad = bd.pad(x2d, ((k, k), (k, k)), mode='reflect')
                except Exception:
                    xpad = bd.pad(x2d, ((k, k), (k, k)), mode='edge')
                ii = bd.cumsum(bd.cumsum(xpad, axis=0), axis=1)
                s = (ii[ks:, ks:] - ii[:-ks, ks:] - ii[ks:, :-ks] + ii[:-ks, :-ks])
                return s / (ks * ks)

            def _correlate(x2d, radius_px: int, passes_n: int):
                y = x2d
                for _ in range(max(1, int(passes_n))):
                    y = _box_blur_2d(y, radius_px)
                return y

            tex = bd.random.standard_normal(rgb.shape)
            if self.dyeCloudCorrelated and r > 0:
                tex_r = self._correlate_2d(tex[..., 0], r, self.dyeCloudCorrPasses)
                tex_g = self._correlate_2d(tex[..., 1], r, self.dyeCloudCorrPasses)
                tex_b = self._correlate_2d(tex[..., 2], r, self.dyeCloudCorrPasses)
                tex = bd.stack((tex_r, tex_g, tex_b), axis=-1)

            # Normalize so grain_strength is stable across resolutions
            tex = tex / (bd.std(tex) + eps)
            rgb_noisy = rgb_noisy * (1.0 + tex * strength_map[..., None])

        resultIm = self._blend_bw_color(rgb_noisy, self.dyeCloudColorBalance)

        return bd.maximum(resultIm, 0)


    def _HalationOptical(self, intersectRayBatch):
        pass


    def _Halation2D(self, rgb_image):
        pass


    # ==================================================================


    def _box_blur_2d(self, x2d, radius_px: int):
        """Box blur that preserves input size.

        Implemented via summed-area table (integral image).
        Works with NumPy and CuPy backends.
        """
        if radius_px <= 0:
            return x2d

        k = int(radius_px)
        ks = 2 * k + 1

        # Pad input so output matches original size
        try:
            xpad = bd.pad(x2d, ((k, k), (k, k)), mode='reflect')
        except Exception:
            xpad = bd.pad(x2d, ((k, k), (k, k)), mode='edge')

        # Integral image (summed-area table)
        ii = bd.cumsum(bd.cumsum(xpad, axis=0), axis=1)

        # Add a zero border so window-sum indexing is exact and output is (H, W)
        ii = bd.pad(ii, ((1, 0), (1, 0)), mode='constant', constant_values=0)

        H, W = x2d.shape[0], x2d.shape[1]

        # Window sums for each (H, W) location
        s = (
                ii[ks:ks + H, ks:ks + W]
                - ii[0:H, ks:ks + W]
                - ii[ks:ks + H, 0:W]
                + ii[0:H, 0:W]
        )
        return s / (ks * ks)


    def _correlate_2d(self, x2d, radius_px: int, passes_n: int):
        """Apply repeated box blurs to approximate a Gaussian blur."""
        y = x2d
        for _ in range(max(1, int(passes_n))):
            y = self._box_blur_2d(y, radius_px)
        return y


    def _blend_bw_color(self, noise_rgb, bw_weight: float):
        """Blend RGB noise with a monochrome (luma) noise component.

        bw_weight: 0 -> keep RGB noise; 1 -> purely monochrome noise replicated to RGB.
        """
        bw_weight = float(max(0.0, min(1.0, bw_weight)))
        if bw_weight <= 0.0:
            return noise_rgb
        if bw_weight >= 1.0:
            n_bw = (noise_rgb[..., 0] * 0.2126 + noise_rgb[..., 1] * 0.7152 + noise_rgb[..., 2] * 0.0722)[..., None]
            return bd.concatenate((n_bw, n_bw, n_bw), axis=-1)

        n_bw = (noise_rgb[..., 0] * 0.2126 + noise_rgb[..., 1] * 0.7152 + noise_rgb[..., 2] * 0.0722)[..., None]
        n_bw3 = bd.concatenate((n_bw, n_bw, n_bw), axis=-1)
        return (1.0 - bw_weight) * noise_rgb + bw_weight * n_bw3




