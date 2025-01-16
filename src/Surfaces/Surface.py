

from enum import Enum
import matplotlib.pyplot as plt

from Util.Backend import backend as bd 
from Util.Backend import constant
from Util.Misc import Magnitude, Normalized, ArrayNormalized
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO
from Util.PlotTest import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch
from Raytracing.Refraction import Refract
from Raytracing.RayBatch import RayBatch 
from Material import Material 



class CurvatureType(Enum):
    Standard = 0      # Sperical element 
    EvenAspheric = 1  # Common ASPH 
    Cylindrical = 2   # For Anamorphics 
    Parabolic = 3     # Mostly for reflective optics 




# ==================================================================
""" ============================================================ """
# ==================================================================


"""
All the methods that calculate ray related results must return 3 parts:
-  Refraction or direct result 
-  Reflection or secondary result 
-  Vignetted 
"""



class Surface:
    """
    Standard spherical surface. 
    """
    def __init__(self, r, t, sd, m = "AIR"):
        self.radius = constant(r)
        self.thickness = constant(t)
        self.clearSemiDiameter = constant(sd)
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
        
        """Thickness or z location of the clear semi diameter edge, scaler t_sd"""
        self.sdCumulative = None

        """Local optical axis of the surface, normalized vector (x, y, z)"""
        self._axis = OBJ_FACING
        # By default it is parallel to Z and facing object side

        """Vector poinring from radius center to the front vertex"""
        self._radiusDirection = None

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
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT + self.radius])  
        self._radiusDirection = self.frontVertex - self.radiusCenter

        self.sdCumulative = cumulativeT + self.radius + bd.sqrt(self.radius**TWO - self.clearSemiDiameter**TWO) * bd.sign(-self.radius)


    def SetVertices(self, frontVtx, radiusVtx):
        """
        Given the front vertex and center of radius, calculate the cumulative thickness and the local optical axis. This is for when the surface is not on the optical axis of the lens. 
        """

        # If this is called, it is sufficient to believe that 
        # the surface is not on the same optical axis with the lens. 
        self.IsOnAxis = False 

        self.frontVertex = frontVtx
        self.radiusCenter = radiusVtx
        self._radiusDirection = self.frontVertex - self.radiusCenter

        self.cumulativeThickness = Magnitude(self.frontVertex - ORIGIN)
        self._axis = Normalized(self.frontVertex - self.radiusCenter)

        # This semi diameter thickness might be inaccurate, due to potnetial transformation of the surface.
        self.sdThickness = self.cumulativeThickness + self.radius + bd.sqrt(self.radius**TWO - self.clearSemiDiameter**TWO) * bd.sign(-self.radius)
        
        # TODO: add inverse transformation matrix calculation 


    # ==============================================================
    """ ===================== Calculations ===================== """
    # ==============================================================

    def RI(self, wavelength):
        return self.material.RI(wavelength)


    def DrawSurface(self):

        DrawSpherical(
            self.radius,
            self.clearSemiDiameter,
            self.cumulativeThickness
            )


    def Intersection(self, incomingRaybatch):
        """
        Given a raybatch, calculate the intersection of these rays on this surface and return the intersection coordinates. 

        :param incomingRaybatch: RayBatch that will be tested for intersection. 

        :return: An array of intersections, a bull secondary array, the bool array of vingetted. 
        """

        # TODO: add case for when radius is infinity 

        position = incomingRaybatch.Position()
        direction = incomingRaybatch.Direction()

        if(not self.IsOnAxis):
            # TODO: Add inverse transfrom here to compensate the off axis element position, if any.
            pass 

        # Translate to sphere's local space
        oc = position - self.radiusCenter
        
        # Coefficients for quadratic equation
        a = bd.sum(direction**TWO, axis=1) 
        b = 2.0 * bd.sum(oc * direction, axis=1)
        c = bd.sum(oc**TWO, axis=1) - self.radius**TWO

        # Discriminant
        discriminant = b ** TWO - constant(4) * a * c
        
        # Some rays are not going to interset with the sphere at all, select only the ones that will have an intersection with the sphere  
        intersetIndices = discriminant > 0

        # Calculate t values
        t1 = (-b - bd.sqrt(discriminant)) / (TWO * a)
        t2 = (-b + bd.sqrt(discriminant)) / (TWO * a)

        # This t value is to determine which side of the spherical surface is the right intersection 
        t = t1 
        mask = bd.sign(self.radius) != bd.sign(direction[:, 2])
        t[mask] = t2[mask]

        # Intersection points
        p = position[intersetIndices] + t[intersetIndices][:, bd.newaxis] * direction[intersetIndices]

        # Among the spherical intersections, some will be outside of this surface, select only the ones that do land on the surface based on the clear semi diameter 
        clear = bd.sqrt(p[:, 0]**TWO + p[:, 1]**TWO) < self.clearSemiDiameter

        # Vector and line are different, it might happen that the line intersect with the sphere but the vector does not. Here t1 and t2 are used to judge if the vector itself actually does not intersect 
        clear &= ~((t1[intersetIndices]<0) & (t2[intersetIndices]<0))

        intersetIndices[intersetIndices] = clear

        return p[clear], None, ~intersetIndices
    

    def Normal(self, intersections):
        """
        Given the intersections, calculate the normal direction on these intersection points. 
        The intersections must be on the surface, otherwise the result may be undefined. 

        :param intersections: points on the surface. 

        :return: Normalized normals of the intersection points on this surface. 
        """

        return ArrayNormalized(intersections - self.radiusCenter)


    def CrossSection(self, planeOrientation):
        """
        Given a plane, return the expression of the surface on this plane.
        Mostly for initial setup of the lens. 
        """

        pass 


    def RayReaction(self, incidentRaybatch):
        """
        Deal with all the reactions the rays have upon reaching the surface. This includes: 
        - Refraction. 
        - Reflection. 
        - Vignette.
        - Scattering. 
        - Color shift. 
        """

        

        refreacted = None
        reflected = None 
        vignetted = None



        return refreacted, reflected, vignetted


    def NaiveTrace(self, incidentRaybatch, previousRI, inverted=False):
        """
        Given a raybatch, deal with the primary reaction this surface has. For an refractive element, only calculate the refractions, vingette and TIR are returned and not calculated. 
        """

        # First find the intersections 
        intersections, _temp, boolVig = self.Intersection(incidentRaybatch)

        #DrawPoints(intersections)
        #self.DrawSurface() # Draw call=========
        #DrawDirection(position, direction) # Draw call=========
        

        # The normal should be pointing at the oppoiste z direction as the indicent raybatch 
        desiredDirection = -bd.sign(incidentRaybatch.Direction()[:, 2])[~boolVig] 
        
        normals = self.Normal(intersections)

        normals[desiredDirection != bd.sign(normals[:, 2])] *= -1

        #DrawRaybatch(incidentRaybatch) # Draw call=========
        #DrawNormal(intersections, normals, lineWidths=1) # Draw call=========
        #plt.draw() # Draw call=========
        #plt.pause(10) # Draw call=========
        
        # Truncate the rays that are vignetted 
        directions = incidentRaybatch.Direction()[~boolVig]
        currentRI = self.material.RI(incidentRaybatch.Wavelength()[~boolVig])
        previousRI = previousRI[~boolVig]

        # If the ray hits from the behind, RI needs to be swapped 
        if(inverted):
            currentRI, previousRI = previousRI, currentRI 

        # Only the non vignetted rays goes into refraction 
        refracted, TIR, _temp = Refract(directions, normals, previousRI, currentRI)

        # This _temp is for a different use from the _temp above 
        _temp = RayBatch(bd.copy(incidentRaybatch.value[~boolVig][~TIR]))
        _temp.SetPosition(intersections[~TIR])
        _temp.SetDirection(refracted)

        return _temp, TIR, boolVig



def main():
    testSurface = Surface(20, 1, 4)
    testSurface.SetCumulative(2)

if __name__ == "__main__":
    main()