

from enum import Enum

# from sys import path as pythonpath

# print("\n ,".join(pythonpath))

from Util.Backend import backend as bd 
from Util.Misc import Magnitude, Normalized
from Util.Globals import ORIGIN, OBJ_FACING
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

        """Whether this surface share the same optical axis as the lens"""
        self.IsOnAxis = True
        # By default the surface is treated to have the same optical axis as the lens 

        """Position of the center of the surface in world space, vector (x, y, z)"""
        self.frontVertex = None 

        """Distance from front vertex to the origin, scaler t_z""" 
        self.cumulativeThickness = None 

        """Position of the center of radius, vector (x, y, z)"""
        self.radiusCenter = None 
        
        """Local optical axis of the surface, vector (x, y, z)"""
        self._axis = OBJ_FACING
        # By default it is parallel to Z and facing object side

        """Inverse transform matrix to offset the incident when the surface is off axis"""
        self._inverseTransform = bd.identity(4)
        # If the surface is on axis, then use the identity matrix


        self.cType = CurvatureType.Standard


    # ==============================================================
    """ ====================== Setting up ====================== """
    # ==============================================================

    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the vertices. This is for when the surface share the same optical axis with the lens. 
        """

        # The local optical axis remains the same as OBJ FACING 
        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([0, 0, cumulativeT])
        self.radiusCenter = bd.array([0, 0, cumulativeT + self.radius])  


    def SetVertices(self, frontVtx, radiusVtx):
        """
        Given the front vertex and center of radius, calculate the cumulative thickness and the local optical axis. This is for when the surface is not on the optical axis of the lens. 
        """

        # If this is called, it is sufficient to believe that 
        # the surface is not on the same optical axis with the lens. 
        self.IsOnAxis = False 

        self.frontVertex = frontVtx
        self.radiusCenter = radiusVtx

        self.cumulativeThickness = Magnitude(self.frontVertex - ORIGIN)
        self._axis = Normalized(self.frontVertex - self.radiusCenter)

        # TODO: add inverse transformation matrix calculation here 


    # ==============================================================
    """ ===================== Calculations ===================== """
    # ==============================================================

    def Intersection(self, incomingRaybatch):
        """
        Given a raybatch, calculate the intersection of these rays on this surface and return the intersection coordinates. 

        :return: An array of intersections and an array of vingetted. 
        """
        position = incomingRaybatch.Position()
        direction = incomingRaybatch.Direction()

        if(not self.IsOnAxis):
            # TODO: Add inverse transfrom here 
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
    testSurface.SetCumulative(2)

if __name__ == "__main__":
    main()