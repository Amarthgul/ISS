

from .Standard import StdImager
from Material import Material
from Raytracing.Polarization import DielectricCoherency
from Surfaces.Surface import Surface
from Util.Globals import INFINITY, ZERO, ONE, RNG
from Util.Backend import backend as bd
from Util.ColorPDF import ColorPDF
from Util.ImageIO import SaveFormat, SaveAsDNG, SaveAsEXR


class PolSensor(StdImager):
    """
    Photodiode Array.
    This class refers to both CMOS and CCD.
    """

    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 1920):
        super().__init__(bfd, w, h, horiPx)



    def FilterAoLP(self, image):
        return self.FilterShadeAoLP(image)


    def FilterShadeAoLP(self, image):
        """Convert RGB-channel AoLP radians to a cyclic RGB display map.

        Finite channel angles have equal weight in a doubled-angle circular
        mean. Colors interpolate through yellow (0/180 degrees), blue (45),
        cyan (90), and red (135). Undefined or cancelling orientations are
        black. This display map is not a per-channel quantitative AoLP map.
        """
        valid = bd.isfinite(image)
        angles = bd.where(valid, image, 0)
        x = bd.sum(bd.where(valid, bd.cos(2 * angles), 0), axis=-1)
        y = bd.sum(bd.where(valid, bd.sin(2 * angles), 0), axis=-1)
        phase = bd.mod(0.5 * bd.arctan2(y, x), bd.pi) * (4 / bd.pi)
        sector = bd.floor(phase).astype(int)
        fraction = (phase - sector)[..., None]
        sector %= 4
        palette = bd.array([[1., 1., 0.], [0., 0., 1.], [0., 1., 1.],
                            [1., 0., 0.], [1., 1., 0.]])
        rgb = palette[sector] * (1 - fraction) + palette[sector + 1] * fraction
        defined = bd.hypot(x, y) > 1e-6 * bd.sum(valid, axis=-1)
        return bd.where(defined[..., None], rgb, 0)

