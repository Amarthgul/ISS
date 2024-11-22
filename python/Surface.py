
import numpy as np 
from enum import Enum


from Material import * 


_PLACEHOLDER_RI = 1.5


class SurfaceType(Enum):
    Standard = 0      # Sperical element 
    EvenAspheric = 1  # Common ASPH 
    Cylindrical = 2   # For Anamorphics 
    Stop = 3          # Main diaphragm 
    Gate = 4          # Feild stop that restraining the rays 
    


class Surface:
    def __init__(self, r, t, sd, m = "AIR"):
        self.radius = r
        self.thickness = t
        self.clearSemiDiameter = sd 
        self.material = Material(m)
        
        self.chamfer = None 

        self.cumulativeThickness = None 
        self.frontVertex = None 
        self.radiusCenter = None 

        self._isMonochromatic = False 
        self._monoRI = _PLACEHOLDER_RI

        self.asphCoef = []


    def SetAsMonochromatic(self, monoRI = _PLACEHOLDER_RI):
        """
        Set this surface as monochromatic, index of refraction then will not change based on wavelength. 

        :param monoRI: refractive index that will be used as the constant RI in calculation. 
        """
        # This method is useful for monochromatic sims like pseudo B&W, also used during testing.  
        self._isMonochromatic = True 
        self._monoRI = monoRI 


    def SetCumulative(self, cd):
        self.cumulativeThickness = cd 

        if(self._isImagePlane): return 

        self.radiusCenter = np.array([0, 0, self.radius + self.cumulativeThickness])


    def SetFrontVertex(self, vec3pos):
        self.frontVertex = vec3pos


class Stop(Surface):
    def __init__(self, t):
        self.radius = np.inf
        self.thickness = t 
        self.clearSemiDiameter = np.inf # Will be updated 
        self.material = None 


