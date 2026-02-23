

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


    def _ApplyGrainAndNoise(self, rgb_image):
        """Hook for image-domain grain/noise.

        Default: identity.
        Must return rgb_image.
        """
        return rgb_image


    def _Grain(self, rgb_image):
        pass


    def _DensityCurve(self, rgb_image):
        pass


