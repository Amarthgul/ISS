

from enum import Enum

# from sys import path as pythonpath

# print("\n ,".join(pythonpath))

from Util.Backend import backend as bd 
from Util.Misc import Magnitude 
from Util.Globals import ORIGIN 
from Material import Material 

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

        # Position of the center of the surface in world space, vector (x, y, z)
        self.frontVertex = None 

        # Distance from front vertex to the origin, scaler t_z 
        self.cumulativeThickness = None 

        # Position of the center of radius, vector (x, y, z)
        self.radiusCenter = None 
        

        self._axis = None

        self.cType = CurvatureType.Standard


    def SetVertices(self, frontVtx, radiusVtx):
        self.frontVertex = frontVtx
        self.radiusCenter = radiusVtx

        self.cumulativeThickness = Magnitude(self.frontVertex - ORIGIN)
        self._axis = self.frontVertex - self.radiusCenter


    def Intersection(self, incomingRaybatch):
        """
        Given a raybatch, calculate the intersection of these rays on this surface and return the intersection coordinates. 

        :return: An array of intersections and an array of vingetted. 
        """
        pass 


    def Normal(self, intersections):
        """
        Given the intersections, calculate the normal direction on these intersection points. 
        The intersections must be on the surface, otherwise the result may be undefined.  
        """
        
        pass 


    def CrossSection(self, planeOrientation):
        """
        Given a plane, return the expression of the surface on this plane.
        Mostly for initial setup of the lens. 
        """

        pass 


    def RayReaction(self, incidentRaybatch):
        """
        Deal with all the reactions the rays have upon reaching the surface. 
        """

        refreacted = None
        reflected = None 
        vignetted = None



        return refreacted, reflected, vignetted




def main():
    testSurface = Surface(20, 1, 4)


if __name__ == "__main__":
    main()