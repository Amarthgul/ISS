

from .Surface import * 


class EvenAspheric(Surface):
    """
    Even aspheric surface.
    """ 
    def __init__(self):

        self.asphCoef = [] # Aspherical coefficients 

        self.cType = CurvatureType.EvenAspheric