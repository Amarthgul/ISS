

from .Standard import StdImager
from Util.Globals import Channels




class Film(StdImager):
    def __init__(self, sr):
        super().__init__(bfd = 42, w = 36, h = 24, horiPx = 1920)

        # A color PDF component to model the spectral response
        self.spectralResponse = sr

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

        # Use the same logic as the stdImager
        # But use the spectralResponse to get the PDF of the three channel, use the

        pass


    def _HalationBounce(self):
        pass



