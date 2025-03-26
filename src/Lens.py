


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Util.PltPlot import DrawSpherical, DrawRaybatch, DrawPoint, DrawDirection, DrawPoints
from Util.Backend import constant
from Util.Backend import backend as bd 
from Util.Globals import ZERO, ONE, TWO, Axis, LambdaLines, AXIAL_ZERO
from Util.ColorWavelength import WavelengthToRGB
from Util.Misc import AxialDistance, TransversalDistance
from Util.SpatialEllipse import SpatialCircle

from Surfaces.Stop import Stop
from Surfaces.Surface import Surface
from Surfaces.Pupil import Pupil
from Surfaces.PrincipalPlane import PrincipalPlane
from Surfaces.ClearBoundary import ClearBoundary

from Material import Material
from Raytracing.RayBatch import RayBatch, GenerateEmpty
from Raytracing.Raypath import RayPath
from Raytracing.Emission import EmitFromStop, EmitFromObjectSpace, EmitFromPoint, EmitField, EmitFromPointFullFrontal



class Lens:
    def __init__(self):
        self.fNumber = constant(2.0)
        self.focalLength = constant(50.0)

        self.focalPoint = None 

        self.surfaces = []
        self.env = Material("AIR") # The environment it is submerged in, air by default

        self.lenses = [] # Every pair of surfaces that form a lens
        self.groups = [] # Cemented lenses become a group 
        self.groupMaxSemi = {}

        self.rayPath = [] # Rays with only position info on each surface 
        
        self.entrancePupil = Pupil() 
        self.frontPincipalPlane = PrincipalPlane() 

        """Index of stop amopng the lens surfaces for easier indexing"""
        self.stopIndex = None 

        """Axial direction length of all the surfaces. From front vertex of the first surface to the last surface"""
        self.totalAxialLength = None 

        """Minimum object distance, i.e., min focus distance"""
        self.MOD = None 
        # This property is useful when there are floating lens element involved and the position of lens groups are dependent on interpolation between inf and MOD focus. 

        self.isAfocal = False 


        self._lastSurfaceIndex = 0

        self._temp = None # Variable for developing and not to be taken serieously 



    def AddSurface(self, inputSurface):
        self.surfaces.append(inputSurface)


    def UpdateLens(self):
        """
        Iterate throught the elements and update their relative parameters. 
        Separate the surfaces into lenses and groups.
        Create clear boundaries for each surfaces. 
        Start a ray trace and finds the entrance pupil. 
        """

        currentT = constant(0.0)

        for i in range(len(self.surfaces)):
            self.surfaces[i].SetCumulative(bd.copy(currentT))
            if(i != len(self.surfaces)-1):
                # The last surface's thickness of a lens is not useful for the lens 
                currentT += self.surfaces[i].thickness
            if isinstance(self.surfaces[i], Stop):
                self.stopIndex = i 
            self._lastSurfaceIndex = i

        # Total axial length, counting from the first surface vertex to the last  
        self.totalAxialLength = currentT
        self._PartitionGroups()
        self._CreateClearBoundary()

        # Afocal system does not have a principal plane and the entrance pupil is the first stop 
        if(self.isAfocal):
            self._UpdateAfocal()
            return 
        
        self.entrancePupil.SetFirstElementSD(self.surfaces[0].clearSemiDiameter) 

        self._TraceEntrancePupil()
        self._TraceFocalPrincipal() 
        
        
    def DrawLens(self):
        """
        Iterate through the surfaces and draw them.
        """
        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                self.surfaces[i].DrawSurface()


    def BFD(self):
        return self.surfaces[self._lastSurfaceIndex].thickness


    def SetAperture(self, aperture):
        """
        Set the aperture of the lens. 

        :param aperture: f-number of the lens.
        """
        
        calculatedPupilSize = self.focalLength / aperture
        
        self.entrancePupil.SetPupilSize(calculatedPupilSize / TWO)
        

    def Propagate(self, rayBatch, recordPath=False, reflection=False, iteCount=2):
        """
        Propagate the raybatch through the lens. 

        :param recordPath: when enabled, all paths of rays will be recorded. For production use, turn this off to avoid unnecessary memory usage. 
        :param reflection: when enabled, rays will reflect based on the Fresnel reflectance, non-sequential rays propagation will also be calculated. This defaults to False to reduce time and memory consumption. 
        :param iteCount: number of iterations for calculating non-sequential propagation. Due to reflected rays will also have both refraction and reflection, the number of rays will increase geometrically. Ite Count larger than 3 could potentially stagger the computer. 
        
        :return: primary imaging RB, ray path if recorded, and the reflected RB. 
        """

        if(recordPath):
            self.rayPath = RayPath()
            self.rayPath.Append(rayBatch, None, None)

        reflectedRB = RayBatch()

        # This pass of propagation traces the primary imaging components, for photographic application, that is the refractive imaging. 
        # If enabled, the reflected rays will be creted, recorded, and returned during the process. 
        
        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                rayBatch, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                    rayBatch, 
                    self._FindPreviousRI(i, rayBatch), 
                    reflection = reflection)
                
                #print(i, "th surface intersect ", rayBatch.value.shape)

                # The index of main RB is where they are after a surface
                rayBatch.SetIndex(i)

                if(reflection):
                    # For the reflected rays, the surface index means where they are before a surface
                    _reflectedRB.RadiantKill()
                    # _reflectedRB.SetIndex(i)
                    reflectedRB.Merge(_reflectedRB)

                
                if(recordPath):
                    self.rayPath.Append(rayBatch, _tir, _vig)


        # DrawRaybatch(reflectedRB, lLength=2, lineColor="r") # =========== Draw call
        

        if (reflection):
            reflectedRB.SurfaceKill(0)
            color = ["r", "b"]
            exitRB = RayBatch(None)
            for _c in range(iteCount):
                #print("In ", _c, " th reflection iteration")
                reflectedRB = self._BounceReflection(reflectedRB)
                exitRB.Merge(reflectedRB.TrimExitRays(self._lastSurfaceIndex)) 


            # After 3 times of reflection, these still facing negative Z might as well be dropped to save space 
            reflectedRB.Merge(exitRB)
            reflectedRB = reflectedRB.GetRaysFacing()
            
            reflectedRB = self._PropagateReflectedThrough(reflectedRB)

        # print(rayBatch.SurfaceIndex())

        # self.rayPath.DrawPath(40)

        # Return the sequential and non-sequential rays. Note that since this is the first pass of non-sequential propagation, most non-sequential rays are pointing at object space, there needs to be further calculations for them to arrive at the imager. 
        return rayBatch, self.rayPath, reflectedRB


    def BestFocusBFD(self, distance):
        """
        Calculate the best back focal distance given an object distance. This is achieved by finding the smallest RMS spot position in the exit rays. 
        """
        
        if(self.isAfocal):
            return self.totalAxialLength + .1


        focusRB = EmitField(0, 0, distance, self.entrancePupil.GetSamplePoints())

        focusRP = RayPath()

        focusRP.Append(focusRB, None, None)

        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                focusRB, _tir, _vig = self.surfaces[i].NaiveTrace(
                    focusRB, self._FindPreviousRI(i, focusRB))
                
                focusRP.Append(focusRB, _tir, _vig)

        # Accquring the position and direction of the exiting rays 
        lastIndex = len(focusRP.position) - 1
        lastPos = focusRP.position[lastIndex]
        lastDir = focusRP.direction[lastIndex]

        # Subtract the z position of the best focus with the axial length of the lens, yielding the back focal length for best focus 
        return focusRP.FindConvergingPoint(lastPos, lastDir)[Axis.Z.value] - self.totalAxialLength


    def GetInfo(self):

        info = "- Lens Info: \n" +\
            str(len(self.lenses)) + " lenses arranged in " + str(len(self.groups)) + " groups"
        
        if(not self.isAfocal):
            info +=  "\nFocal Length:   \t" + str(self.focalLength) +"\n" +\
            "Max working N:  \tf/" + str(self.focalLength / self.entrancePupil.GetMaxPupilSize())

        info += "\n" +\
            "Max pupil dia:  \t" + str(self.entrancePupil.GetMaxPupilSize()) + "\n" +\
            "Axial length:   \t" + str(self.totalAxialLength)
        
        if(not self.isAfocal):
            info += "\n" +\
            "Principal plane:\t" + str(self.frontPincipalPlane.GetInnerZ()) + "\n" +\
            "Focal point:    \t" + str(self.focalPoint[Axis.Z.value]) + "\n"
            
        return info


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _UpdateAfocal(self):
        """
        This method is for updating an afocal lens, which does not have a focal length. 
        """
        firstSD = 0

        for i in range(len(self.surfaces)):
            if (not isinstance(self.surfaces[i], Stop)):
                firstSD = self.surfaces[i].clearSemiDiameter
                break 

        self.surfaces[self.stopIndex].EnforceSemiDiameter(firstSD)
        self.entrancePupil.SetFirstElementSD(firstSD)

        pupilPoint = bd.array([
            [ZERO,      ZERO,          self.surfaces[0].cumulativeThickness], 
            [firstSD,   ZERO,          self.surfaces[0].cumulativeThickness]
        ])

        self.entrancePupil.SetSamplePoints(pupilPoint)


    def _PartitionGroups(self):
        """
        Iterate through the surfaces, part them into lenses and groups. 
        """

        # Clear the previous record, if any 
        self.lenses = []
        self.groups = []

        # ===================== Count the lenses =======================
        lens = []
        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                if(not self.surfaces[i].IsAirMaterial()):
                    lens.append(i)
                    lens.append(i+1)
                    self.lenses.append(lens)
                    lens = []


        # ===================== Count the groups =======================
        currentInGroup = False
        currentGroup = []

        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                if( not currentInGroup and not self.surfaces[i].IsAirMaterial()):
                    # The first surface of a group 
                    currentInGroup = True
                    currentGroup.append(i)
                    self.surfaces[i].isGroupTerminal = True

                elif(currentInGroup and self.surfaces[i].IsAirMaterial()):
                    # The last surface of a group 
                    currentInGroup = False
                    currentGroup.append(i)
                    self.surfaces[i].isGroupTerminal = True
                    self.groups.append(currentGroup)
                    currentGroup = []

                else:
                    currentGroup.append(i)
                
        largestR = 0 

        # ======= Find the largest semi diameter in each group ========
        for g in range(len(self.groups)):
            for s in self.groups[g]:
                if (largestR < self.surfaces[s].clearSemiDiameter):
                    largestR = self.surfaces[s].clearSemiDiameter
            self.groupMaxSemi.update({g: largestR})
            largestR = 0 

        # print(self.groupMaxSemi)

                
    def _CreateClearBoundary(self):
        for key, value in self.groupMaxSemi.items():
            # The key here is group index, and value is the max clear semi diameter of the group 

            for s in self.groups[key]:
                # Iterate through the surfaces in the group 

                # ============ Deal with the first surface (of a group) ============
                if(s == self.groups[key][0]):
                    # C1 must be connected to the clear semi diameter of this surface 
                    C1 = SpatialCircle(self.surfaces[s].sdCumulative, 
                                       self.surfaces[s].clearSemiDiameter)

                    # For lens groups, the first surface typically have SD smaller than the group max SD, such as the first group after the stop in a Planar setup 
                    if (self.surfaces[s].clearSemiDiameter < value):

                        # Check if there are surfaces after 
                        if(len(self.groups[key]) >= 2):
                            # There must be a border at the next surface's SD z position 
                            C3 = SpatialCircle(self.surfaces[s+1].sdCumulative, value)
                            
                            sdDifference = value - self.surfaces[s].clearSemiDiameter
                            sdzDifference = self.surfaces[s+1].sdCumulative - self.surfaces[s].sdCumulative

                            if(sdzDifference > sdDifference):
                                C2 = SpatialCircle(
                                    self.surfaces[s].sdCumulative + sdDifference, value)
                            else:
                                C2 = SpatialCircle(
                                    self.surfaces[s].sdCumulative,
                                    self.surfaces[s+1].clearSemiDiameter-sdzDifference
                                )
                            self.surfaces[s+1].clearBoundaryT = ClearBoundary(C1, C2)
                            self.surfaces[s+1].clearBoundaryL = ClearBoundary(C2, C3)

                    else:
                        # This else is only possible when the SD of this surface is equal to the group max SD. So check if next surface exist and use its SD z position.
                        if(len(self.groups[key]) >= 2):
                            C2 = SpatialCircle(self.surfaces[s+1].sdCumulative, value) 
                            self.surfaces[s+1].clearBoundaryL = ClearBoundary(C1, C2)


                # ============ Deal with the last surface (of a group) =============
                if(s == self.groups[key][-1]):
                    # C3 must be connected to the clear semi diameter of this surface
                    C3 = SpatialCircle(self.surfaces[s].sdCumulative, 
                                       self.surfaces[s].clearSemiDiameter)
                    
                    if (self.surfaces[s].clearSemiDiameter < value):

                        # Check if there are surfaces before  
                        if(len(self.groups[key]) >= 2):
                            C1 = SpatialCircle(self.surfaces[s-1].sdCumulative, value)

                            sdDifference = value - self.surfaces[s].clearSemiDiameter
                            sdzDifference = self.surfaces[s].sdCumulative - self.surfaces[s-1].sdCumulative

                            if(sdzDifference > sdDifference):
                                C2 = SpatialCircle(
                                    self.surfaces[s].sdCumulative - sdDifference, value)
                            else:
                                C2 = SpatialCircle(
                                    self.surfaces[s].sdCumulative,
                                    self.surfaces[s-1].clearSemiDiameter-sdzDifference
                                )

                            # Last surface might already have boundaries created when processing previous surfaces, but the coverage may not be correct. 
                            if(self.surfaces[s].clearBoundaryT == None or self.surfaces[s].clearBoundaryL == None):
                                self.surfaces[s].clearBoundaryL = ClearBoundary(C1, C2)
                                self.surfaces[s].clearBoundaryT = ClearBoundary(C2, C3)

                    else:
                        # This else is only possible when the SD of this surface is equal to the group max SD. But being the last surface, it might also happen that the previous surface already created a boundary for it. 

                        if (self.surfaces[s].clearBoundaryL == None):
                            C2 = SpatialCircle(self.surfaces[s-1].sdCumulative, value)

                            self.surfaces[s].clearBoundaryL = ClearBoundary(C2, C3) 


                # ================ Deal with middle surface, if any ================
                else: 
                    pass 

                # print("Surface ", s, " max csd ", value)
                if(self.surfaces[s].clearSemiDiameter < value):
                    pass 
                    
        # Second pass enforce the surface with disableBoundaryL set 
        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                if(self.surfaces[i].disableBoundaryL):
                    # Remove the Longitudinal clear boundary 
                    self.surfaces[i].clearBoundaryL = None
                    C1 = SpatialCircle(self.surfaces[i-1].sdCumulative, self.surfaces[i-1].clearSemiDiameter)
                    C2 = SpatialCircle(self.surfaces[i].sdCumulative, self.surfaces[i].clearSemiDiameter)
                    self.surfaces[i].clearBoundaryT = ClearBoundary(C1, C2)


    def _TraceEntrancePupil(self, stopSD=None, sampleCount=11, wavelength = LambdaLines['D']):
        """
        Start rays from the center of the stop, casting forward and trace the size and location of the entrance pupil. 

        :param stopSD: Semi diameter of the stop, if not given, this will be calculated from the adjacent surfaces. 
        :param sampleCount: number of sample sets to be emitted from the stop. 
        :param wavelength: the wavelength at which the calculations will be performed. 
        """

        
        # The physical diaghram of the stop should not be bigger than the adjacent surfaces, as such, if not given, the smallest of the two will be used.
        if(stopSD == None):
            priorSD = self.surfaces[self.stopIndex-1].clearSemiDiameter
            postSD = self.surfaces[self.stopIndex+1].clearSemiDiameter
            # Minus a small value to avoid potential error caused by identical values
            stopSD = bd.min(bd.array([priorSD, postSD])) - AXIAL_ZERO


        self.entrancePupil.sampleWavelength = wavelength
        colors = plt.get_cmap('tab10').colors

        objectSideRPs = [RayPath() for i in range(sampleCount)] 
        _tirs = [[]for i in range(sampleCount)]
        _vigs = [[]for i in range(sampleCount)]

        # Create the target and source points 
        targetOne = bd.array([
            ZERO, 
            self.surfaces[self.stopIndex-1].clearSemiDiameter,
            self.surfaces[self.stopIndex-1].sdCumulative
        ])
        targetTwo = bd.array([
            ZERO, 
            -self.surfaces[self.stopIndex-1].clearSemiDiameter,
            self.surfaces[self.stopIndex-1].sdCumulative
        ])

        # Generate raybatch and add them into the path record
        objectSideRBs = [
            self._CastFromStop(
                self.surfaces[self.stopIndex].frontVertex+bd.array([ZERO, p, ZERO]),
                targetOne, targetTwo, wavelength = wavelength)
            for p in bd.linspace(ZERO, stopSD, sampleCount)
        ]
        # for j in range(sampleCount):
        #     objectSideRPs[j].Append(objectSideRBs[j], None, None)
        #     DrawRaybatch(objectSideRBs[j])  # Draw call=========
        
        # Propogate through the surfaces 
        for i in range(len(self.surfaces)):
            forwardIndex = self.stopIndex - i - 1
            if(forwardIndex >= 0):
                #print("At surface ", forwardIndex)
                #self.surfaces[forwardIndex].DrawSurface() # Draw call=========
                for j in range(sampleCount):
                    objectSideRBs[j], _tirs[j], _vigs[j] = self.surfaces[forwardIndex].NaiveTrace(
                        objectSideRBs[j], 
                        self._FindPreviousRI(forwardIndex, objectSideRBs[j]), True
                        ) 
                    objectSideRPs[j].Append(objectSideRBs[j], _tirs[j], _vigs[j])
                #DrawRaybatch(objectSideRBs[i])  # Draw call=========
                #plt.draw()
                #plt.pause(4)

        # for j in range(sampleCount):
        #     objectSideRPs[j].DrawPath(10, omitIncident=False, color=colors[j%len(colors)]) # Draw call=========
        # plt.draw()
        # plt.pause(10)

        poss = [[]for i in range(sampleCount)]
        dirs = [[]for i in range(sampleCount)]
        intersections = [[]for i in range(sampleCount)]
        for j in range(sampleCount):
            poss[j], dirs[j] = objectSideRPs[j].ExitingPairs()
            intersections[j] = objectSideRPs[j].FindConvergingPoint(poss[j], dirs[j])
        #     DrawPoint(intersections[j], color=colors[j%len(colors)])
        #     DrawDirection(poss[j], dirs[j], lineColor=colors[j%len(colors)], lineLength=40, arrowRatio=0)
        # plt.draw()
        # plt.pause(10)
        # intersections = bd.vstack(intersections)
        # wColor = WavelengthToRGB(wavelength).tolist()
        # DrawPoints(intersections, color=tuple(wColor))

        self.entrancePupil.SetSamplePoints(bd.array(intersections))


    def _CastFromStop(self, emissionPoint, targetOne, targetTwo, numRays=20, wavelength = LambdaLines['d']):

        if bd.isclose(self.surfaces[self.stopIndex].cumulativeThickness, self.surfaces[self.stopIndex-1].cumulativeThickness):
            return EmitFromPointFullFrontal(
            emissionPoint, numRays, wavelength = wavelength)
        else:
            return EmitFromPoint(
            emissionPoint,
            targetOne, targetTwo, numRays, wavelength = wavelength)


    def _TraceFocalPrincipal(self, wavelength = LambdaLines['D']):
        """
        Trace the front focal point and principal plane. These two will then be used to calculate the focal length. 

        :param wavelength: the wavelength at which the calculations will be performed. 
        """
        self.frontPincipalPlane.sampleWavelength = wavelength

        frontRB = EmitFromObjectSpace(self.entrancePupil.GetMaxPupilSize() / TWO, numRays=15, halfSide=True)
        frontRP = RayPath()
        frontRP.Append(frontRB, None, None)

        for i in range(len(self.surfaces)):
            if(not isinstance(self.surfaces[i], Stop)):
                frontRB, _tir, _vig = self.surfaces[i].NaiveTrace(
                    frontRB, self._FindPreviousRI(i, frontRB))   
                frontRP.Append(frontRB, _tir, _vig)  
                # self.surfaces[i].DrawSurface() # Draw call =======

        # frontRP.DrawPath(40) # Draw call =======
        # plt.draw() # Draw call =======
        # plt.pause(30) # Draw call =======

        frontRP = frontRP.PruneAll()
        #frontRP.DrawPath(40) # Draw call =======
        
        pos, dir = frontRP.ExitingPairs()
        self.focalPoint = frontRP.FindConvergingPoint(pos, dir)
        # DrawPoint(self.focalPoint) # Draw call =======

        intersections = frontRP.EndToEndIntersection() 
        intersections = intersections[~bd.any(bd.isnan(intersections), axis=1)]
        
        # At this point, the on-axis rays does not have an intersection since they're straight throughout. A separate on-axis point is created, copying the z value of the closest intersection point.
        minPoint = TransversalDistance(intersections)
        minPoint = intersections[bd.argmin(minPoint)]
        minPoint = bd.array([ZERO, ZERO, minPoint[Axis.Z.value]])
        intersections = bd.vstack([intersections, minPoint])

        # Now, with the on-axis point also created, add the updated intersection into the sample points of the principal plane
        self.frontPincipalPlane.SetSamplePoints(intersections)
        #self.frontPincipalPlane.DrawSamplePoints() # Draw call =======

        # Calculate the focal length 
        self.focalLength = self.focalPoint[Axis.Z.value] - self.frontPincipalPlane.GetInnerZ()


    def _FindPreviousRI(self, index, raybatch):
        """
        Find the refractive index of the previous surface. 
        """
        if (index == 0 or isinstance(self.surfaces[index-1], Stop)):
            return self.env.RI(raybatch.Wavelength())
        else:
            return self.surfaces[index -1].RI(raybatch.Wavelength())
        

    def _BounceReflection(self, reflectedRB):
        """
        Iterate through each surface, calculate the reflection inside the surface. 
        """
        returnRB = RayBatch(None)
        

        # For the reflected lights at the first surface (i=0), they are just gone and there is no point in trying to deal with them, thus starting at 1
        for i in range(1, len(self.surfaces)):

            # Find the rays that are in the current surface
            inSurfaceRB = reflectedRB.GetRaysAt(i)

            if(not isinstance(self.surfaces[i], Stop)):
                _surfaceRB, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                    inSurfaceRB, 
                    self._FindPreviousRI(i, inSurfaceRB), 
                    reflection = True)
                
                # _surfaceRB are rays that are refracted into the previous space 
                _surfaceRB.SetIndex(i-1)
                # Keep only the rays that did not intersect with the previous surface 
                inSurfaceRB.Merge(_surfaceRB)
                inSurfaceRB.Merge(_reflectedRB)

                # DrawRaybatch(_surfaceRB, lLength=2, lineColor="b") # =========== Draw call
                # DrawRaybatch(_reflectedRB, lLength=2, lineColor="g") # =========== Draw call
            # The ones that still faces back might as well be droped 

            if((i+1 < len(self.surfaces)) and 
               (not isinstance(self.surfaces[i+1], Stop))):
                # Try to find the ones that will interact with the next surface  
                _surfaceRB, _tir, _vig, _reflectedRB = self.surfaces[i+1].Trace(
                    inSurfaceRB, 
                    self._FindPreviousRI(i+1, inSurfaceRB), 
                    inverted = True,
                    reflection = True)
                
                # _surfaceRB are rays that are refracted into the next space 
                _surfaceRB.SetIndex(i+1)
                inSurfaceRB.Merge(_surfaceRB)
                # Add the reflected rays from previous surfaces. Given the implementation, TIR should already be included in the _reflectedRB
                inSurfaceRB.Merge(_reflectedRB)
                # These _reflectedRB rays should either intersect with the clear boundaries or become sequential again 
            
            returnRB.Merge(inSurfaceRB)
            returnRB.RadiantKill()

        returnRB.SurfaceKill(0)

        return returnRB


    def _PropagateReflectedThrough(self, reflectedRB):
        """
        Propogate the rays in reflectedRB in each surface through the lens. 
        """
        returnRB = RayBatch(None)

        for i in range(1, len(self.surfaces)):

            if(not isinstance(self.surfaces[i], Stop)):

                # Find the rays that are in the current surface
                inSurfaceRB = reflectedRB.GetRaysAt(i)
                returnRB.Merge(inSurfaceRB)

                # Explicitly disable the reflection 
                returnRB, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                        returnRB, 
                        self._FindPreviousRI(i, returnRB), 
                        reflection = False)
                

        return returnRB


def main():

    test = Lens() 




    


if __name__ == "__main__":
    main()