

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

        self.grainPhotonScale = 4096

        self.grainStrength = 0.2

        self.grainCorrelated = False

        self.grainCorrRadiusPxMin = 1
        self.grainCorrRadiusPxExtra = 2

        self.grainCorrPasses = 2

        self.grainMidGray = 0.18


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


    def ApplyGrainAndNoise(self, rgb_image):
        """Hook for image-domain grain/noise.

        Default: identity.
        Must return rgb_image.
        """
        rgb_image = self._Grain(rgb_image)

        return rgb_image


    def _ApplyHalation(self, intersectRayBatch, rgb_image):
        """Optional halation hook.

        Left as identity by default. Subclasses / future work can implement:
          - back-plate bounce / scattering using `self.backPlateDistance`
          - per-layer absorption using `self.emulsionOrder`

        If you implement `_HalationBounce`, you can call it here.
        """
        # Placeholder: no halation yet
        return rgb_image


    def _Grain(self, rgb_image):
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

        lam = rgb * self.grainPhotonScale
        lam = bd.maximum(lam, 0)

        try:
            counts = bd.random.poisson(lam)
            rgb_noisy = counts / self.grainPhotonScale
        except Exception:
            # Fallback: Gaussian approximation if poisson isn't available
            n = bd.random.standard_normal(lam.shape)
            rgb_noisy = (lam + bd.sqrt(bd.maximum(lam, eps)) * n) / self.grainPhotonScale
            rgb_noisy = bd.maximum(rgb_noisy, 0)

        # ----------------------------------------------------------------------------------
        # 2) Optional self.grainCorrelated grain texture (multiplicative)
        # ----------------------------------------------------------------------------------
        if self.grainStrength > 0:
            shadow_pow = float(getattr(self, "grain_shadow_power", 1.25))
            shadow_weight = bd.power(1.0 - (lum / (lum + (mid + eps))), shadow_pow)
            strength_map = self.grainStrength * (0.25 + 0.75 * shadow_weight)

            # exp_norm may be numpy/cupy scalar; convert carefully
            try:
                exp_n = float(exp_norm.get())
            except Exception:
                exp_n = float(exp_norm)

            r = int(round(self.grainCorrRadiusPxMin + (1.0 - exp_n) * self.grainCorrRadiusPxExtra))
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
            if self.grainCorrelated and r > 0:
                tex_r = _correlate(tex[..., 0], r, self.grainCorrPasses)
                tex_g = _correlate(tex[..., 1], r, self.grainCorrPasses)
                tex_b = _correlate(tex[..., 2], r, self.grainCorrPasses)
                tex = bd.stack((tex_r, tex_g, tex_b), axis=-1)

            # Normalize so grain_strength is stable across resolutions
            tex = tex / (bd.std(tex) + eps)
            rgb_noisy = rgb_noisy * (1.0 + tex * strength_map[..., None])

        return bd.maximum(rgb_noisy, 0)


    def _DensityCurve(self, rgb_image):
        pass


