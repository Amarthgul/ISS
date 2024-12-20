
from .Imager import Imager


class PDA(Imager):
    """
    Photodiode Array. 
    This class refers to both CMOS and CCD. 
    """

    def __init__(self):
        self.t_UVIR = 1.5

