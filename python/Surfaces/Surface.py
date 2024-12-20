
import numpy as np 
from enum import Enum

from Material import * 


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
    Standard spherical surface 
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


    def Intersection(self, incomingRaybatch):
        """
        Given a raybatch, calculate the intersection of these raybatch with this surface and return the intersection coordinates. 
        """
        pass 


    def Normal(self, intersections):
        """
        Given the intersections, calculate the normal direction on these intersection points. 
        The intersections must be on the surface, otherwise the result may be undefined.  
        """
        
        pass 

    def Section(self, planeOrientation):
        """
        Given a plane, return the expression of the surface on this plane.
        Mostly for initial setup of the lens. 
        """

        pass 



