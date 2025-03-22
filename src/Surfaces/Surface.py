

from enum import Enum
import matplotlib.pyplot as plt

from Util.Backend import backend as bd 
from Util.Backend import constant
from Util.Sampling import CircularDistribution
from Util.Misc import Magnitude, ArrayMagnitude, Normalized, ArrayNormalized, PointsInTriangle
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, SetUnifScale, RemoveBG, AddXYZ, DrawEllipse, DrawClearBoundary
from Raytracing.Refraction import Refract
from Raytracing.Reflection import Reflect
from Raytracing.Polarization import SenkrechtUndParallel, PolarizeRB, ResidueRB, FresnelReflectance, QuantitativePolarize
from Raytracing.RayBatch import RayBatch 
from Raytracing.Raypath import RayPath
from Raytracing.Emission import EmitField
from Material import Material 



class CurvatureType(Enum):
    Standard = 0      # Sperical element 
    EvenAspheric = 1  # Common ASPH 
    Cylindrical = 2   # For Anamorphics 
    Parabolic = 3     # Mostly for reflective optics 



# Field stop constrains the rays to a certain area, out of area rays are vignetted.
# This is also used to define the boundary of each surface. A typicaly spherical surface has a circular boundary, thus circular field stop. Some surfaces, such as anaomorphics, have rectangular field stops. Additionally, the imager typically also has a rectangular field stop.
class FieldStopType(Enum):
    Circular = 0
    Rectangular = 1







# ==================================================================
""" ============================================================ """
# ==================================================================


"""
All the methods that calculate raybatch related results should return 3 parts:
-  Refraction rays (or direct resul). 
-  Reflection rays (or secondary result). 
-  Vignetted rays. 
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

        """Longitudinal-direction clear boundary"""
        self.clearBoundaryL = None 
        # This could be assigned during lens update, if needed 

        """Tangential-direction clear boundary. While it is called tangential, it may not be entirly perpendicular to the optical axis, it could also be chamfered."""
        self.clearBoundaryT = None 
        # This could be assigned during lens update, if needed 

        """This flag makes the surface clear boundary to directly connects to the previsous one's semi diameter edge. Effectively removing Longitudinal CB abd leaving only the Tangential one"""
        self.disableBoundaryL = False


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

        """Whether this surface is the starting or ending surface of a group"""
        self.isGroupTerminal = False

        """Local optical axis of the surface, normalized vector (x, y, z)"""
        self._axis = OBJ_FACING
        # By default it is parallel to Z and facing object side

        """Vector poinring from radius center to the front vertex"""
        self._radiusDirection = None

        """Inverse transform matrix to offset the incident when the surface is off axis"""
        self._inverseTransform = bd.identity(4)
        # If the surface is on axis, then use the identity matrix

        """Type of the curvature. By default it is standard spherical surface"""
        self.cType = CurvatureType.Standard

        
    # ==============================================================
    """ ====================== Setting up ====================== """
    # ==============================================================


    def SetCumulative(self, cumulativeT):
        """
        Given the cumulative thickness, calculate the vertices. This is for when the surface share the same optical axis with the lens. 
        """
        cumulativeT = bd.array(cumulativeT)

        # The local optical axis remains the same as OBJ FACING 
        self.cumulativeThickness = cumulativeT
        self.frontVertex = bd.array([ZERO, ZERO, cumulativeT])
        self.radiusCenter = bd.array([ZERO, ZERO, cumulativeT + self.radius])  
        self._radiusDirection = self.frontVertex - self.radiusCenter

        if(self.radius == INFINITY):
            # When r=inf, the cumulative thickness at the edge is the same as the cumulative thickness of the vertex. 
            self.sdCumulative = cumulativeT
        else:
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
            self.cumulativeThickness, 
            surfaceColor=SURFACE_COLOR,
            )
        
        if(self.clearBoundaryL is not None):
            DrawClearBoundary(self.clearBoundaryL.E1, self.clearBoundaryL.E2, surfaceColor=BOUNDARY_COLOR, opacity=0.05)

        if(self.clearBoundaryT is not None):
            DrawClearBoundary(self.clearBoundaryT.E1, self.clearBoundaryT.E2, surfaceColor=BOUNDARY_COLOR, opacity=0.05) 


    def zRange(self):
        """
        Return the range of the surface on the optical axis, presume the surface is on the optical axis. If the surface is off axis, the return result may be undefined. 

        :return: the vertex z and the edge z value of the surface. 
        """

        vertexZ = self.frontVertex[Axis.Z.value]

        if(self.radius is INFINITY):
            return vertexZ, vertexZ # edge z will be the same as vertex z when r is infinity 

        # Only calculate edge z when r is not inf
        edgeZ = bd.sqrt(self.radius**2 - self.clearSemiDiameter**2)

        if(self.radius > 0):
            edgeZ = vertexZ - edgeZ
        elif(self.radius < 0):
            edgeZ = vertexZ + edgeZ

        return vertexZ, edgeZ
        

    def IsAirMaterial(self):
        """
        If this surface is followed by air as material. 
        """
        return self.material.name == "AIR"

        
    def Intersection(self, incidentRaybatch):
        """
        Given a raybatch, calculate the intersection of these rays on this surface and return the intersection coordinates. 

        :param incomingRaybatch: RayBatch that will be tested for intersection. 

        :return: An array of intersections, a bull secondary array, the bool array of vingetted. 
        """

        if(self.radius == INFINITY):
            return self._PlaneIntersection(incidentRaybatch)
        else:
            return self._SphericalIntersection(incidentRaybatch)
        

    def Normal(self, intersections):
        """
        Given the intersections, calculate the normal direction on these intersection points. 
        The intersections must be on the surface, otherwise the result may be undefined. 

        :param intersections: points on the surface. 

        :return: Normalized normals of the intersection points on this surface. 
        """
        # TODO: consider adding the raybatch as an argument and use it to calculate the right normal direction 
        if(self.radius == INFINITY):
            copied = bd.array([ZERO, ZERO, -ONE])
            return bd.tile(copied, (intersections.shape[0], 1))
        else:
            return ArrayNormalized(intersections - self.radiusCenter)


    def CrossSection(self, planeOrientation):
        """
        Given a plane, return the expression of the surface on this plane.
        Mostly for initial setup of the lens. 
        """

        pass 


    def NaiveTrace(self, incidentRaybatch, previousRI, inverted=False):
        """
        Given a raybatch, deal with the primary reaction this surface has. For an refractive element, only calculate the refractions, vingette and TIR are returned but not calculated. 
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

        # DrawRaybatch(incidentRaybatch) # Draw call=========
        # DrawNormal(intersections, normals, lineWidths=1) # Draw call=========
        # plt.draw() # Draw call=========
        # plt.pause(10) # Draw call=========
        
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


    def Trace(self, incidentRaybatch, previousRI, inverted=False, reflection=False):
        """
        Deal with all the reactions the rays have upon reaching the surface. This includes: 
        - Refraction. The main contributor to image formation. 
        - Reflection. Including both mirror, specular, and diffuse reflection, also consider the polarization based on the Fresnel equation. 
        - Vignette. Rays that are vignetted from the main imaging surface may enter the lens barrel and absorbed, or reflected again by the clear boundaries. 
        This is the main method that should be called when tracing rays through the lens to accquire an image.

        :param incidentRaybatch: main raybatch at question. 
        :param previousRI: array representing the RI of the surface before that, i.e., the RI of the medium the rays are currently in. 
        :param inverted: bool for whether or not the rays are aiming from behind.
        :param reflection: bool for whether or not to calculate refections. 

        :return: a raybatch of refracted rays, bool array indicating TIR, bool array indicating vignetted, and a raybatch that contains all the rays that becomes non-sequential
        """

        # First find the intersections 
        intersections, _temp, boolVig = self.Intersection(incidentRaybatch)
        
        # The normal should be pointing at the oppoiste z direction as the indicent raybatch 
        desiredDirection = -bd.sign(incidentRaybatch.Direction()[:, 2])[~boolVig] 
        # Apply desired direction to the normals 
        normals = self.Normal(intersections)
        normals[desiredDirection != bd.sign(normals[:, 2])] *= -1
        
        # Truncate the rays that are vignetted 
        directions = incidentRaybatch.Direction()[~boolVig]

        # Accquire the index of refractions (resp. wavelength)
        n1 = self.material.RI(incidentRaybatch.Wavelength()[~boolVig])
        n2 = previousRI[~boolVig]

        # If the ray hits from the behind, RI needs to be swapped 
        if(inverted):
            n1, n2 = n2, n1 

        # Only the non vignetted rays goes into refraction 
        refracted, TIR, _temp = Refract(directions, normals, n2, n1)

        

        # DrawDirection(intersections, reflected, lineColor="b") # ======= Draw call

        refractedRB = RayBatch(bd.copy(incidentRaybatch.value[~boolVig][~TIR]))
        refractedRB.SetPosition(intersections[~TIR])
        refractedRB.SetDirection(refracted)


        reflectedRB = RayBatch(bd.copy(incidentRaybatch.value[~boolVig][~TIR]))
        if(reflection):
            # These reflected are the reflected componenet form the refracted due to fresnel  
            reflected = Reflect(directions, normals)

            reflectedRB.SetPosition(intersections[~TIR])
            reflectedRB.SetDirection(reflected[~TIR])
            # DrawRaybatch(reflectedRB, lineColor="g") # ======= Draw call
            # TIR are the reverted selection 
            tirRB = RayBatch(bd.copy(incidentRaybatch.value[~boolVig][TIR]))
            tirRB.SetPosition(intersections[TIR])
            tirRB.SetDirection(reflected[TIR])

        
            #print(tirRB.PolarizedRadiance())

            # ==============================================================
            # ========================= Polarization =======================

            # Reflectance ratio along senkrecht and parallel direction (Fresnel equation)
            R_s, R_p = FresnelReflectance(normals[~TIR], directions[~TIR], refracted, n1[~TIR], n2[~TIR])
            

            # Accquire s and p direction for polarization, reflection and refraction 
            senkrecht, parallel = SenkrechtUndParallel(directions, normals)
            

            # DrawDirection(intersections, senkrecht, lineColor="r", lineLength=1) # ============ Draw call
            # DrawDirection(intersections, parallel, lineColor="b", lineLength=1) # ============ Draw call

            # DrawDirection(intersections, normals, lineColor="g", lineLength=2)# ============ Draw call
            # DrawDirection(intersections, reflected, lineColor="purple", lineLength=2)# ============ Draw call

            senkrecht, parallel = QuantitativePolarize(
                incidentRaybatch.PolarizationMat()[~boolVig][~TIR],
                senkrecht[~TIR][:, :2], 
                parallel[~TIR][:, :2], 
                R_s, 
                R_p
            )

            # senkrecht = senkrecht[~TIR][:, :2] * R_s[:, bd.newaxis]
            # parallel  = parallel[~TIR][:, :2]  * R_p[:, bd.newaxis]

            # for pos, mat in zip(intersections, incidentRaybatch.PolarizationMat()[~boolVig]):
            #     DrawEllipse(mat, pos)# ============ Draw call
            
            refractedRB = PolarizeRB(refractedRB, senkrecht, parallel)


            # for pos, mat in zip(intersections, reflectedRB.PolarizationMat()):
            #     DrawEllipse(mat, pos, lColor="m")# ============ Draw call

            #print(reflectedRB.PolarizationMat())
            
            reflectedRB = ResidueRB(reflectedRB, senkrecht, parallel)
            # print(reflectedRB.PolarizationMat(), "\n\n")

            # for pos, mat in zip(intersections, refractedRB.PolarizationMat()):
            #     DrawEllipse(mat, pos)# ============ Draw call

            # for pos, mat in zip(reflectedRB.Position(), reflectedRB.PolarizationMat()):
            #     DrawEllipse(mat, pos, lColor="c")# ============ Draw call

            # for pos, mat in zip(tirRB.Position(), tirRB.PolarizationMat()):
            #     DrawEllipse(mat, pos, lColor="b")# ============ Draw call

            # print(refractedRB.PolarizedRadiance())
            # print(reflectedRB.value.shape)
            # print(tirRB.value.shape)
            # print("\n")

            # Copy the vignetted rays to prepare for clear boundary check 
            vigRB = RayBatch(bd.copy(incidentRaybatch.value[boolVig]))


            if( (self.clearBoundaryL is not None) and bd.any(boolVig) and (not inverted) ):

                vigReflRBL, _NonInterMask = self.clearBoundaryL.Trace(vigRB, previousRI[boolVig])
                if(self.clearBoundaryT is not None and bd.any(_NonInterMask)):
                    # Theoeretically, if there is a tansverse boundary, then the vignetted rays that are not intersected with the longitudinal boundary should intersect with the transverse boundary. 
                    # Create an RB using the rays that did not intersect L boundary
                    vigReflRBT = vigRB.Mask(~_NonInterMask)
                    vigReflRBT, _ = self.clearBoundaryT.Trace(vigReflRBT, previousRI[boolVig][~_NonInterMask])

                    vigReflRBL = vigReflRBL.Merge(vigReflRBT)

                # These clear boundary reflections may also contain their own TIR
                reflectedRB = reflectedRB.Merge(vigReflRBL)
                
            reflectedRB = reflectedRB.Merge(tirRB)

        return refractedRB, TIR, boolVig, reflectedRB
    


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _PlaneIntersection(self, incomingRaybatch):
        """
        Given a plane, calculate the intersection of the rays on the plane.
        """

        position = incomingRaybatch.Position()
        direction = incomingRaybatch.Direction()

        denom = bd.dot(direction, self._axis)
        
        # Avoid division by zero for parallel vectors
        parallel_mask = bd.isclose(denom, 0)
        
        # Compute t for each vector
        t = bd.dot((self.frontVertex - position), self._axis) / denom
        t[parallel_mask] = bd.nan  # Assign NaN for parallel vectors
        
        # Compute the intersection points
        intersections = position + t[:, bd.newaxis] * direction

        # Calculate the bool mask for valid intersections within the region 
        fsMask = self._FieldStopMask(intersections)

        # Note that just like the spherical case, the returned intersection contains only the ones that actually falls onto the surface, those outside of the clear semi-diameter are excluded
        return intersections[fsMask], \
            bd.zeros(intersections.shape[0]).astype(bd.bool_),\
            parallel_mask | (~fsMask)


    def _SphericalIntersection(self, incomingRaybatch):
        """
        Given a spherical surface, calculate the intersection of the rays on the surface.
        """
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

        # print("theoretical inter: ", bd.sum(intersetIndices))

        # Calculate t values
        t1 = (-b - bd.sqrt(discriminant)) / (TWO * a)
        t2 = (-b + bd.sqrt(discriminant)) / (TWO * a)

        # This t value is to determine which side of the spherical surface is the right intersection 
        t = t1 
        mask = bd.sign(self.radius) != bd.sign(direction[:, 2])
        t[mask] = t2[mask]

        # Intersection points
        p = position[intersetIndices] + t[intersetIndices][:, bd.newaxis] * direction[intersetIndices]

        #DrawPoints(p)

        # Among the spherical intersections, some will be outside of this surface, select only the ones that do land on the surface based on the clear semi diameter 
        clear = self._FieldStopMask(p)

        # Vector and line are different, it might happen that the line intersect with the sphere but the vector does not. Here t1 and t2 are used to judge if the vector itself actually does not intersect 
        clear &= ~((t1[intersetIndices]<0) & (t2[intersetIndices]<0))

        intersetIndices[intersetIndices] = clear

        # print("Actual intersection ", bd.sum(intersetIndices))

        return p[clear], \
            bd.zeros(p.shape[0]).astype(bd.bool_), \
            ~intersetIndices


    def _FieldStopMask(self, intersections):
        """
        Given the intersections, filter out the ones that are outside of the field stop. 
        """
        # TODO: add tilt shift handling here
        return bd.sqrt(intersections[:, 0]**TWO + intersections[:, 1]**TWO) < self.clearSemiDiameter



def main():

    testRP = RayPath()

    sampleTar = CircularDistribution(zDepth=3) * bd.array([22, 22, 1])
    testRB = EmitField(30, 0, distance=50, sampleTargets=sampleTar)
    testRP.Append(testRB, None, None)
    airMaterial = Material()

    testSurface1 = Surface(45, 1, 22, "E-KZFH1")
    testSurface1.SetCumulative(3)
    testSurface2 = Surface(-50, 1, 22)
    testSurface2.SetCumulative(17)

    testRB, _tir, _vig, reflectedRB = testSurface1.Trace(testRB, airMaterial.RI(testRB.Wavelength()))
    # DrawDirection(reflectedRB.Position(), reflectedRB.Direction(), lineColor="purple", lineLength=2)# ============ Draw call
    # print(reflectedRB.PolarizedRadiance())
    testRB.SetIndex(0)
    testRP.Append(testRB, _tir, _vig)

    testRB, _tir, _vig, reflectedRB = testSurface2.Trace(testRB, testSurface1.RI(testRB.Wavelength()))
    # DrawDirection(reflectedRB.Position(), reflectedRB.Direction(), lineColor="purple", lineLength=2)# ============ Draw call
    # print(reflectedRB.PolarizedRadiance())
    testRB.SetIndex(1)
    testRP.Append(testRB, _tir, _vig)

    testSurface1.DrawSurface()
    testSurface2.DrawSurface()
    testRP.DrawPath(omitIncident=False)
    #DrawRaybatch(testRB, lLength=53, arrowRatio=0)
    SetUnifScale(50)
    #AddXYZ()
    RemoveBG()
    plt.show()


if __name__ == "__main__":
    main()