
import numpy as np 
from enum import Enum


from Material import * 


_PLACEHOLDER_RI = 1.5



class CurvatureType(Enum):
    Standard = 0      # Sperical element 
    
    EvenAspheric = 1  # Common ASPH 
    Cylindrical = 2   # For Anamorphics 
    Parabolic = 3     # Mostly for reflective optics 


# ==================================================================
""" ============================================================ """
# ==================================================================


class Surface:
    """
    Normal spherical surface 
    """
    def __init__(self, r, t, sd, m = "AIR"):
        self.radius = r
        self.thickness = t
        self.clearSemiDiameter = sd 
        self.material = Material(m)

        self.cumulativeThickness = None 
        self.frontVertex = None 
        self.radiusCenter = None 

        self.cType = CurvatureType.Standard


    def SetCumulative(self, cd):
        self.cumulativeThickness = cd 

        self.radiusCenter = np.array([0, 0, self.radius + self.cumulativeThickness])


    def SetFrontVertex(self, vec3pos):
        self.frontVertex = vec3pos


    def Intersection(self, incomingRays):
        pass 


    def Normal(self, intersections):
        pass 





