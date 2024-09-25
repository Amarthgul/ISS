
import numpy as np 
from Material import * 

_PLACEHOLDER_RI = 1.5


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

        self._isObject = False

        self._isImagePlane = False 
        self._ipSize = []
        self._ipCount = []

        self._isMonochromatic = False 
        self._monoRI = _PLACEHOLDER_RI

        # Add asph 
        

    def SetAsObject(self):
        self._isObject = True


    def SetAsImagePlane(self, width = 36, height = 24, horizontal = 6000, vertical = 4000):
        self._isImagePlane = True  
        self._ipSize = np.array([width, height])
        self._ipCount = np.array([horizontal, vertical])


    def SetAsMonochromatic(self, monoRI = _PLACEHOLDER_RI):
        """
        Set this surface as monochromatic, index of refraction then will not change regardless of wavelength. 

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

    def IsImagePlane(self):
        return self._isImagePlane
    
    def ImagePlaneSize(self):
        return self._ipSize
    
    def ImagePlanePx(self):
        return self._ipCount
