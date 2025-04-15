
from .Standard import StdImager
from Material import Material
from Surfaces.Surface import Surface
from Util.Globals import INFINITY
from Util.Backend import backend as bd

class PDA(StdImager):
    """
    Photodiode Array. 
    This class refers to both CMOS and CCD. 
    """

    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 1920):
        super().__init__(bfd, w, h, horiPx)

        """This is the thickness of the UVIR cut glass"""
        self.tUVIR = 1.5

        """This is the distance between the UVIR glass and the sensor plane"""
        self.t = .5

        self.material = "UVIR"

        self.surfaces = []


    def GetUVIR(self):
        """
        Accquire the surfaces of the UVIR glass.
        """

        diagonal = bd.sqrt((self.width/2.0)**2 + (self.height/2.0)**2) + 5

        newS = Surface(INFINITY, self.tUVIR, diagonal, self.material)

        self.surfaces = [
            Surface(INFINITY, self.tUVIR, diagonal, self.material),
            Surface(INFINITY, self.t, diagonal)
        ]

        return self.surfaces



