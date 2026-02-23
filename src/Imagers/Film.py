

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


    def ApplyDensityCurve(self, image, overExpRadiance):
        """
        Given an already rendered image, apply the density curve to it.

        :param image: target image.
        :param overExpRadiance: radiance level that will be regarded as the maximum overexposed radiance, i.e., the top of the density curve.
        """

        pass


    def _integralRays(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):
        """
        Film integral:
          - Same channel-tag deposition as StdImager._integralRaysChannelBased
          - Each ray is additionally weighted by the film spectral response:
                weight = self.colorPDF.SpectralResponse(wavelength, resp_channel)
          - If self.emissionPDF is provided, apply importance correction:
                weight *= 1 / (self.emissionPDF.SpectralResponse(wavelength, emit_channel) + NEAR_ZERO)

        Notes:
          - Deposit channel is ALWAYS the ray's channel tag (the "dye" channel).
          - Response channel can be remapped via dyeSpectralPairs (Lomography-like displacement).
        """

        pxPitch = self.width / self.horizontalPx
        pxOffset = bd.array([self.horizontalPx / 2, self.verticalPx / 2, 0])

        # Rays that arrived at the image plane
        rayHitMask = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Convert hit positions to pixel coords
        rayPos = bd.floor(intersectRayBatch.Position()[rayHitMask] / pxPitch + pxOffset).astype(int)[:, :2]

        # Per-ray quantities
        radiant = intersectRayBatch.PolarizedRadiance(polarized)[rayHitMask]
        wavelength = intersectRayBatch.Wavelength()[rayHitMask]
        chan_emit = intersectRayBatch.Channel()[rayHitMask].astype(int)  # deposit channel (dye channel)

        # Mask out hits outside the imager area (avoid negative indexing issues)
        in_bounds = (
                (rayPos[:, 0] >= 0) & (rayPos[:, 0] < self.horizontalPx) &
                (rayPos[:, 1] >= 0) & (rayPos[:, 1] < self.verticalPx)
        )
        rayPos = rayPos[in_bounds]
        radiant = radiant[in_bounds]
        wavelength = wavelength[in_bounds]
        chan_emit = chan_emit[in_bounds]

        # ------------------------------------------------------------
        # Determine which channel's *spectral response* curve to use
        # (dyeSpectralPairs allows remapping; defaults are identity)
        # ------------------------------------------------------------
        # Build a small mapping array indexable by chan_emit (0/1/2).
        # Dict values may be Channels enum members or ints; handle both.
        def _as_int(ch):
            return int(getattr(ch, "value", ch))

        mapR = _as_int(self.dyeSpectralPairs[Channels.R])
        mapG = _as_int(self.dyeSpectralPairs[Channels.G])
        mapB = _as_int(self.dyeSpectralPairs[Channels.B])

        mapAry = bd.asarray([mapR, mapG, mapB]).astype(int)  # shape (3,)
        chan_resp = mapAry[chan_emit]  # shape (N,)

        # ------------------------------------------------------------
        # Film spectral response weight
        # ------------------------------------------------------------
        film_w = self.colorPDF.SpectralResponse(wavelength, chan_resp)

        # Optional importance correction: divide by emission sampling pdf-like response
        # (This is optional by design; if emissionPDF is None, it behaves like a "look".)
        if self.emissionPDF is not None:
            emit_w = self.emissionPDF.SpectralResponse(wavelength, chan_emit)
            film_w = film_w / (emit_w + NEAR_ZERO)

        radiant = radiant * film_w

        # ------------------------------------------------------------
        # Deposit into per-channel grids
        # ------------------------------------------------------------
        radiantGridR = bd.zeros((self.horizontalPx, self.verticalPx))
        radiantGridG = bd.zeros((self.horizontalPx, self.verticalPx))
        radiantGridB = bd.zeros((self.horizontalPx, self.verticalPx))

        for c, grid in ((0, radiantGridR), (1, radiantGridG), (2, radiantGridB)):
            m = (chan_emit == c)
            if not bd.any(m):
                continue

            pos_c = rayPos[m]
            rad_c = radiant[m]

            # Prune over-exposed outliers (same style as StdImager)
            if (overExpNoiseRemoval is not None) and (bd.max(rad_c) > bd.mean(rad_c)):
                # print(self._IncidentStats(intersectRayBatch))
                rad_c = self._PruneHighOutliers(rad_c, overExpNoiseRemoval)

            bd.add.at(grid, (pos_c[:, 0], pos_c[:, 1]), rad_c)

        rgb_image = bd.stack((radiantGridR, radiantGridG, radiantGridB), axis=-1)

        # Monte Carlo accumulation
        if baseImg is not None:
            rgb_image = baseImg + rgb_image

        return rgb_image


    def _HalationBounce(self):
        pass



