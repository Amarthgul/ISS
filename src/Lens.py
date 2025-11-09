


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from dask.dataframe.methods import boundary_slice
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re

from xarray.plot import surface

from Util.PltPlot import DrawSpherical, DrawRaybatch, DrawPoint, DrawDirection, DrawPoints, SetUnifScale, AddXYZ, RemoveBG
from Util.Backend import constant
from Util.Backend import backend as bd 
from Util.Globals import ZERO, ONE, TWO, Axis, LambdaLines, AXIAL_ZERO, PBR
from Util.ColorWavelength import WavelengthToRGB
from Util.Misc import AxialDistance, TransversalDistance, RectPath
from Util.SpatialEllipse import SpatialCircle
from Util.Sampling import RandomEllipticalDistribution
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm

from Surfaces.Stop import Stop
from Surfaces.Surface import Surface
from Surfaces.EvenAspheric import EvenAspheric
from Surfaces.Pupil import Pupil
from Surfaces.PrincipalPlane import PrincipalPlane
from Surfaces.ClearBoundary import ClearBoundary
from Surfaces.MetalBoundary import MetalBoundary

from Material import Material
from Raytracing.RayBatch import RayBatch, GenerateEmpty
from Raytracing.Raypath import RayPath
from Raytracing.Emission import EmitFromStop, EmitFromObjectSpace, EmitFromPoint, EmitField, EmitFromPointFullFrontal



class Lens:
    def __init__(self):
        self.fNumber = constant(2.0)

        """The actual focal length of the lens."""
        self.focalLength = constant(50.0)

        self.focalPoint = None 

        self.surfaces = []
        self.env = Material("AIR") # The environment it is submerged in, air by default

        self.lenses = [] # Every pair of surfaces that form a lens
        self.groups = [] # Cemented lenses become a group 
        self.groupMaxSemi = {}

        self.rayPath = [] # Rays with only position info on each surface 
        
        self.entrancePupil = Pupil() 
        self.frontPrincipalPlane = PrincipalPlane()

        self.diaphragm = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmS.svg"))

        """Index of stop among the lens surfaces for easier indexing"""
        self.stopIndex = None 

        """Axial direction length of all the surfaces. From front vertex of the first surface to the last surface"""
        self.totalAxialLength = None 

        """Minimum object distance, i.e., min focus distance"""
        self.MOD = None 

        self.isAfocal = False 

        self.TestAttr = 0 

        self._lastSurfaceIndex = 0

        self._temp = None # Variable for developing and not to be taken serieously 



    def AddSurface(self, inputSurface):
        self.surfaces.append(inputSurface)


    def AddRearGroup(self, rearGroup, rearGroupDistace=None):
        """
        Append a rear group
        """
        if(len(rearGroup) == 0): return

        self.surfaces[len(self.surfaces)-1].thickness =  .2

        for s in rearGroup:
            self.AddSurface(s)


    def AddFrontGroup(self, frontGroup):
        if(len(frontGroup) == 0): return

        newGroup = []

        for s in frontGroup:
            newGroup.append(s)
        for s in self.surfaces:
            newGroup.append(s)

        self.surfaces = newGroup


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
                # For stop, simply mark its index as the lens' stop index, nothing else need to be done for it
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


    def SetAperture(self, fNumber, useDiaphragm=True):
        """
        Set the aperture of the lens.
        Note that if there is a diaphragm, then this operation will make some attributes less useful.

        :param fNumber: f-number of the lens.
        """


        calculatedPupilDiameter = self.focalLength / fNumber
        # in millimeter or whatever the focal length is using

        fullPupilDiameter = self.entrancePupil.GetMaxPupilSize()

        ratio = calculatedPupilDiameter**2 / fullPupilDiameter**2
        if ratio < 1:
            self.diaphragm.Reset()
            self.diaphragm.DuplicateAroundCenter()
            self.diaphragm.StopDownToRatio(ratio)
        print("desired ratio: ", ratio)

        if useDiaphragm:
            self.entrancePupil.SetPupilShape(self.diaphragm.toImage(), ratio)
        else:
            self.entrancePupil.SetPupilSize(calculatedPupilDiameter / TWO)
        

    def Propagate(self, rayBatch, recordPath=False, reflection=False, iteCount=2):
        """
        Propagate the raybatch through the lens. 

        :param recordPath: when enabled, all paths of rays will be recorded. For production use, turn this off to avoid unnecessary memory usage. 
        :param reflection: when enabled, rays will reflect based on the Fresnel reflectance, non-sequential rays propagation will also be calculated. This defaults to "False" in order to reduce time and memory consumption.
        :param iteCount: number of iterations for calculating non-sequential propagation. Due to reflected rays will also have both refraction and reflection, the number of rays will increase geometrically. Ite Count larger than 3 could potentially stagger the computer. 
        
        :return: primary imaging RB, ray path if recorded, and the reflected RB. 
        """

        if(recordPath):
            self.rayPath = RayPath()
            self.rayPath.Append(rayBatch, None, None)

        reflectedRB = RayBatch()

        # This pass of propagation traces the primary imaging components, for photographic application, that is the refractive imaging.
        
        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                rayBatch, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                    rayBatch, 
                    self._FindPreviousRI(i, rayBatch), 
                    reflection = reflection)

                # The index of main RB is where they are after a surface
                rayBatch.SetIndex(i)

                # If reflection is enabled, the reflected rays will be created, recorded, and returned during the process.
                if(reflection):
                    # For the reflected rays, the surface index means where they are before a surface
                    _reflectedRB.RadiantKill()
                    reflectedRB.Merge(_reflectedRB)

                
                if(recordPath):
                    self.rayPath.Append(rayBatch, _tir, _vig)


        # DrawRaybatch(reflectedRB, lLength=2, lineColor="r") # =========== Draw call
        

        if (reflection):
            AltMethod = True
            if(AltMethod):
                reflectedRBC = reflectedRB.Copy()
                holderRB = RayBatch(None)
                for _c in range(iteCount):
                    outputRB, remainderRB = self._BounceReflectionAlt(reflectedRBC)
                    holderRB.Merge(outputRB)

                    reflectedRB, _ = self._BounceReflection(reflectedRB)

                reflectedRB = self._PropagateReflectedThrough(reflectedRB)
                reflectedRB = reflectedRB.GetDirectionalRay()
                reflectedRB.Merge(holderRB)

            else:
                #print(reflectedRB.SurfaceDistributionInfo())
                color = ["r", "b"]
                exitRB = RayBatch(None)

                for _c in range(iteCount):
                    reflectedRB, _ = self._BounceReflection(reflectedRB)
                    exitRB.Merge(reflectedRB.TrimExitRays(self._lastSurfaceIndex))
                # Register the total reflected rays
                reflectedRB.Merge(exitRB)

                reflectedRB = reflectedRB.GetDirectionalRay()
                reflectedRB = self._PropagateReflectedThrough(reflectedRB)

            print("Total rad: ", reflectedRB.PolarizedRadiance().sum())

        # print(rayBatch.SurfaceIndex())

        # self.rayPath.DrawPath(40)

        # Return the sequential and non-sequential rays. Note that since this is the first pass of non-sequential propagation, most non-sequential rays are pointing at object space, there needs to be further calculations for them to arrive at the imager. 
        return rayBatch, self.rayPath, reflectedRB


    def BestFocusBFD(self, distance):
        """
        Calculate the best back focal distance given an object distance.
        Please note that this is achieved by finding the smallest RMS spot position in the exit rays without considering the density/radiance distribution. So quite often for wide open lenses, the smallest RMS might actually be slightly out of focus.

        :param distance: distance of the source in meters.
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


    def GetFirstElementSamples(self, sampleCount=512):
        """
        Randomly sample points from the 1st surface of the lens. The z position of the samples are determined by the z position of the first surface edge. 
        """
        firstElementDepth = self.surfaces[0].sdCumulative

        return RandomEllipticalDistribution(
            major_axis=self.surfaces[0].clearSemiDiameter,
            minor_axis=self.surfaces[0].clearSemiDiameter,
            zDepth=firstElementDepth,
            samplePoints=sampleCount,
            groupByPoint=True)


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
            "Principal plane:\t" + str(self.frontPrincipalPlane.GetInnerZ()) + "\n" +\
            "Focal point:    \t" + str(self.focalPoint[Axis.Z.value]) + "\n"
            
        return info


    def GetAoV(self, useNominal=None, unitInDegrees=True, halfAngle=True, w=36.0, h=24.0):
        """
        Acquire the angle of view of the lens on a given imager. By default, the imager is assumed to be the standard 135 format still photography dimension, i.e., 36mmx24mm.

        :param useNominal: a nominal focal length, when provided, this will be used instead of the actual physical focal length.
        :param unitInDegrees: whether to use units in degrees.
        :param w: width of the imager.
        :param h: height of the imager.

        :return:  angle of view of the lens on the given imager.
        """

        if useNominal is not None:
            f = useNominal
        else:
            f = self.focalLength

        full_frame_diag = bd.hypot(w, h)

        # Angle of view formula: 2 * arctan(sensor_diag / (2 * focal_length))
        diag_rad = 2.0 * bd.arctan(full_frame_diag / (2.0 * f))
        w_rad =  2.0 * bd.arctan(w / (2.0 * f))
        h_rad = 2.0 * bd.arctan(h / (2.0 * f))

        mult = 0.5 if halfAngle else 1.0

        if unitInDegrees:
            return bd.rad2deg(bd.array([w_rad, h_rad, diag_rad])) * mult
        else:
            return bd.array([w_rad, h_rad, diag_rad]) * mult


    def SurfaceReport(self):
        info = "- Surface Info: \n"
        index = 1
        for i in self.surfaces:
            if(isinstance(i, Stop)):
                info += "  STOP \t\t\t\t\t" + "{:.3f}".format(i.thickness) + "\n"
            else:
                info += ("  " + str(index) + ":" +
                         " \t" +  "{:.4f}".format(i.radius) +
                         " \t\t" + "{:.3f}".format(i.thickness) +
                         " \t" + str(i.material.category) +
                         " \t"  + str(i.material.name) + "\n")

            index += 1

        info = re.sub(r'(?<=\s)inf(?=\s)', 'INFINITY', info)

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


    def _DirectConnectPrevious(self, _index, sType=PBR.GLASS):

        if (sType == PBR.GLASS):
            boundary = ClearBoundary
        elif (sType == PBR.METAL):
            boundary = MetalBoundary


        # Remove the Longitudinal clear boundary
        self.surfaces[_index].clearBoundaryL = None
        self.surfaces[_index].clearBoundaryT = None
        previousSurfaceIndex = self._PreviousSurfaceIndex(_index)

        _C1 = SpatialCircle(self.surfaces[previousSurfaceIndex].sdCumulative, self.surfaces[previousSurfaceIndex].clearSemiDiameter)
        _C2 = SpatialCircle(self.surfaces[_index].sdCumulative, self.surfaces[_index].clearSemiDiameter)

        self.surfaces[_index].clearBoundaryT = boundary(_C1, _C2)


    def _BevelTowardsImageSpace(self, _index, _maxGroupSD, caller=1, sType=PBR.GLASS):
        """Requires given indexed surface to have a smaller SD than the surface in front of it

        :param _index: index of the surface in lens surface list.
        :param _maxGroupSD: maximum semi diameter of the current croup.
        :param caller: When caller is 1, it means caller sits at the positive side of the Z axis compare to the other surface. Since this method is intended for the last surface of a group, caller is defaulted at 1.
        """

        if(sType == PBR.GLASS):
            boundary = ClearBoundary
        elif(sType == PBR.METAL):
            boundary = MetalBoundary

        _imageSideIndex = _index if (caller == 1) else self._NextSurfaceIndex(_index)
        _objectSideIndex = _index if (caller == -1) else self._PreviousSurfaceIndex(_imageSideIndex)

        _C3 = SpatialCircle(self.surfaces[_imageSideIndex].sdCumulative,
                            self.surfaces[_imageSideIndex].clearSemiDiameter)
        _C1 = SpatialCircle(self.surfaces[_objectSideIndex].sdCumulative, _maxGroupSD)

        sdDifference = _maxGroupSD - self.surfaces[_imageSideIndex].clearSemiDiameter
        sdzDifference = self.surfaces[_imageSideIndex].sdCumulative - self.surfaces[_objectSideIndex].sdCumulative

        if (sdzDifference > sdDifference):
            _C2 = SpatialCircle(
                self.surfaces[_imageSideIndex].sdCumulative - sdDifference, _maxGroupSD)
        else:
            _C2 = SpatialCircle(
                self.surfaces[_imageSideIndex].sdCumulative,
                self.surfaces[_objectSideIndex].clearSemiDiameter - sdzDifference
            )

        # Last surface might already have boundaries created when processing previous surfaces, but the coverage may not be correct.
        if (self.surfaces[_imageSideIndex].clearBoundaryT is None) or (
                self.surfaces[_imageSideIndex].clearBoundaryL is None):
            self.surfaces[_imageSideIndex].clearBoundaryL = boundary(_C1, _C2)
            self.surfaces[_imageSideIndex].clearBoundaryT = boundary(_C2, _C3)


    def _BevelTowardsObjectSpace(self, _index, _maxGroupSD, caller=-1, sType=PBR.GLASS):
        """Requires given indexed surface to have a larger SD than the surface in front of it.

        :param _index: index of the surface in lens surface list.
        :param _maxGroupSD: maximum semi diameter of the current croup.
        :param caller: When caller is 1, it means caller sits at the positive side of the Z axis compare to the other surface. Since this method is intended for the first surface of a group, caller is defaulted at -1.
        """

        if (sType == PBR.GLASS):
            boundary = ClearBoundary
        elif (sType == PBR.METAL):
            boundary = MetalBoundary

        _objectSideIndex = _index if (caller == -1) else self._PreviousSurfaceIndex(_index)
        _imageSideIndex = _index if (caller == 1) else self._NextSurfaceIndex(_objectSideIndex)

        _C1 = SpatialCircle(self.surfaces[_objectSideIndex].sdCumulative,
                            self.surfaces[_objectSideIndex].clearSemiDiameter)

        # There must be a border at the next surface's SD z position
        _C3 = SpatialCircle(self.surfaces[_imageSideIndex].sdCumulative, _maxGroupSD)

        sdDifference = _maxGroupSD - self.surfaces[_objectSideIndex].clearSemiDiameter
        sdzDifference = self.surfaces[_imageSideIndex].sdCumulative - self.surfaces[_objectSideIndex].sdCumulative

        if (sdzDifference > sdDifference):
            _C2 = SpatialCircle(
                self.surfaces[_objectSideIndex].sdCumulative + sdDifference, _maxGroupSD)
        else:
            _C2 = SpatialCircle(
                self.surfaces[_objectSideIndex].sdCumulative,
                self.surfaces[_imageSideIndex].clearSemiDiameter - sdzDifference
            )

        self.surfaces[_imageSideIndex].clearBoundaryT = boundary(_C1, _C2)
        self.surfaces[_imageSideIndex].clearBoundaryL = boundary(_C2, _C3)


    def _CreateClearBoundary(self, createOuterBound=False) -> None:
        """Create clear boundaries for the lens. This includes the boundaries that connects surfaces and make each element sealed solids, and also mechanical parts lying in between the lens elements."""

        self._CreateClearBoundaryInnerGroup()

        if (createOuterBound):
            self._CreateClearBoundaryOuterGroup()


    def _CreateClearBoundaryInnerGroup(self) -> None:
        """Create clear boundaries that connects surfaces and make each element a sealed solid."""

        #TODO: consider splitting this into methods? The current implementation has way to many if nesting...

        for groupIndex, maxGroupSD in self.groupMaxSemi.items():
            # The key here is group index, and value is the max clear semi diameter of the group

            LastSufraceDC = False

            for s in self.groups[groupIndex]:
                # Iterate through the surfaces in the group

                if (self.surfaces[s].disableBoundaryL):
                    self._DirectConnectPrevious(s)
                    # Then skip other checks
                    continue

                # ============ Deal with the first surface (of a group) ============
                if (s == self.groups[groupIndex][0]):
                    # C1 must be connected to the clear semi diameter of this surface
                    C1 = SpatialCircle(self.surfaces[s].sdCumulative,
                                       self.surfaces[s].clearSemiDiameter)

                    # For lens groups, the first surface typically have either the same SD, or smaller SD than the group max SD, such as the first group after the stop in a Planar setup
                    if (self.surfaces[s].clearSemiDiameter < maxGroupSD and len(self.groups[groupIndex]) >= 2):
                        self._BevelTowardsObjectSpace(s, maxGroupSD)


                    else:
                        # This else case is only possible when the SD of this surface is equal to the group max SD (because itself will be the max).
                        # So, it checks if next surface exist.
                        if (len(self.groups[groupIndex]) >= 2):

                            # Even for a single element, it may happen that the 2 surfaces have different SD.
                            # First see if they are lucky enough to share the same SD
                            if (self.surfaces[s + 1].clearSemiDiameter == maxGroupSD):
                                C2 = SpatialCircle(self.surfaces[s + 1].sdCumulative, maxGroupSD)
                                self.surfaces[s + 1].clearBoundaryL = ClearBoundary(C1, C2)

                            # When the check did not pass, this means the other surface has a different SD. But since we already made sure the current surface has the max SD of the group, this means the next one can only be smaller. Thus, make a bevel towards the image space.
                            else:
                                self._BevelTowardsImageSpace(s+1, maxGroupSD)



                # ============ Deal with the last surface (of a group) =============
                elif (s == self.groups[groupIndex][-1]):

                    # C3 must be connected to the clear semi diameter of this surface
                    C3 = SpatialCircle(self.surfaces[s].sdCumulative,
                                       self.surfaces[s].clearSemiDiameter)

                    if (self.surfaces[s].clearSemiDiameter < maxGroupSD) and (len(self.groups[groupIndex]) >= 2):
                        self._BevelTowardsImageSpace(s, maxGroupSD)

                    elif (len(self.groups[groupIndex]) >= 2) and (
                            self.surfaces[s].clearSemiDiameter > self.surfaces[s - 1].clearSemiDiameter) and (
                    self.surfaces[s - 1].disableBoundaryL):
                        self._BevelTowardsObjectSpace(s, maxGroupSD, 1)

                    else:
                        # This else is only possible when the SD of this surface is equal to the group max SD.
                        # But being the last surface, it might also happen that the previous surface already created a boundary for it. So another check is needed.
                        if (self.surfaces[s].clearBoundaryL == None):
                            C2 = SpatialCircle(self.surfaces[s - 1].sdCumulative, maxGroupSD)
                            self.surfaces[s].clearBoundaryL = ClearBoundary(C2, C3)


                # ================ Deal with middle surface, if any ================
                else:
                    if (self.surfaces[s].clearSemiDiameter == self.surfaces[s - 1].clearSemiDiameter):
                        self._DirectConnectPrevious(s)

                # print("Surface ", s, " max csd ", value)
                if (self.surfaces[s].clearSemiDiameter < maxGroupSD):
                    pass


    def _CreateClearBoundaryOuterGroup(self) -> None:
        """Create metalic clear boundaries that lay in between lens groups.
        These boundaries represent the lens barrel and other mechanical constructs."""


        for i in range(len(self.groups) - 1):

            currentGroupLastIndex = self.groups[i][-1]
            nextGroupFirstIndex = self.groups[i + 1][0]

            currentSurface = self.surfaces[currentGroupLastIndex]
            nextSurface = self.surfaces[nextGroupFirstIndex]

            C1 = SpatialCircle(currentSurface.sdCumulative, currentSurface.clearSemiDiameter)
            C2 = SpatialCircle(nextSurface.sdCumulative, nextSurface.clearSemiDiameter)

            # This happens when the current surface "encapsulates" the next one, in which case a metal boundary is not very useful to have, thus might as well just skip it.
            if (currentSurface.sdCumulative > nextSurface.sdCumulative):
                continue

            if(currentSurface.clearSemiDiameter > nextSurface.clearSemiDiameter):
                self._BevelTowardsImageSpace(nextGroupFirstIndex, currentSurface.clearSemiDiameter, caller=1, sType=PBR.METAL)

            elif(nextSurface.clearSemiDiameter > currentSurface.clearSemiDiameter):
                self._BevelTowardsObjectSpace(nextGroupFirstIndex, nextSurface.clearSemiDiameter, caller=1, sType=PBR.METAL)

            else:
                self.surfaces[nextGroupFirstIndex].clearBoundaryT = MetalBoundary(C1, C2)


    def _TraceEntrancePupil(self, stopSD=None, sampleCount=4, wavelength = LambdaLines['D']):
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
        # self.DrawLens()
        for i in range(len(self.surfaces)):
            forwardIndex = self.stopIndex - i - 1
            if(forwardIndex >= 0):
                #print("At surface ", forwardIndex)
                #self.surfaces[forwardIndex].DrawSurface() # Draw call=========
                for j in range(sampleCount):
                    objectSideRBs[j], _tirs[j], _vigs[j] = self.surfaces[forwardIndex].NaiveTrace(
                        objectSideRBs[j], 
                        self._FindPreviousRI(forwardIndex, objectSideRBs[j]),
                        True
                        ) 
                    objectSideRPs[j].Append(objectSideRBs[j], _tirs[j], _vigs[j])

                # DrawRaybatch(objectSideRBs[i])  # Draw call=========
                # plt.draw()
                # plt.pause(8)

        # for j in range(sampleCount):
        #     objectSideRPs[j].DrawPath(10, omitIncident=False, color=colors[j%len(colors)]) # Draw call=========
        # plt.draw()
        # plt.pause(60)

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
            return EmitFromPoint(emissionPoint, targetOne, targetTwo, numRays, wavelength = wavelength)


    def _TraceFocalPrincipal(self, wavelength = LambdaLines['D']):
        """
        Trace the front focal point and principal plane. These two will then be used to calculate the focal length. 

        :param wavelength: the wavelength at which the calculations will be performed. 
        """
        self.frontPrincipalPlane.sampleWavelength = wavelength

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
        # frontRP.DrawPath(40) # Draw call =======
        
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
        self.frontPrincipalPlane.SetSamplePoints(intersections)
        #self.frontPincipalPlane.DrawSamplePoints() # Draw call =======

        # Calculate the focal length 
        self.focalLength = self.focalPoint[Axis.Z.value] - self.frontPrincipalPlane.GetInnerZ()


    def _FindPreviousRI(self, index, raybatch):
        """
        Find the refractive index of the previous surface. 
        """

        if (index == 0 or isinstance(self.surfaces[index-1], Stop)):
            return self.env.RI(raybatch.Wavelength())

        else:
            return self.surfaces[self._PreviousSurfaceIndex(index)].RI(raybatch.Wavelength())
        

    def _PreviousSurfaceIndex(self, index):

        if(index == 0):
            return 0

        else:
            if(isinstance(self.surfaces[index-1], Stop)):
                return self._PreviousSurfaceIndex(index - 1)
            else:
                return index-1


    def _NextSurfaceIndex(self, index):

        if(index == len(self.surfaces)-1):
            return index

        else:
            if (isinstance(self.surfaces[index+1], Stop)):
                return self._NextSurfaceIndex(index + 1)
            else:
                return index+1


    def _ZeroCondition(self, index):
        """In some cases, a previous surface may have 0 thickness, i.e., it is directly on top of the current surface. This method tries to detect this condition"""

        if(self.surfaces[index-1].thickness==0 and not isinstance(self.surfaces[index-1], Stop)):
            # Check if the previous surface is not a Stop and has a 0 thickness
            return index-1, index

        elif(self.surfaces[index-1].thickness==0 and
             isinstance(self.surfaces[index - 1], Stop) and
             self.surfaces[index-2].thickness==0):
            # It is reasonable to assume there's only 1 Stop
            return True

        return False


    def _BounceReflection(self, reflectedRB):
        """
        Iterate through each surface, calculate the reflection inside the surface, including the clear boundaries.
        """
        returnRB = RayBatch(None)
        holderRB = RayBatch(None)

        # For the reflected lights at the first surface (i=0), they are just gone and there is no point in trying to deal with them, thus starting at index 1
        for i in range(1, len(self.surfaces)):

            # Placeholder to avoid scope error
            _reflectedRB = RayBatch(None)

            """========================================================================="""
            # Find the rays that are in the current surface space
            inSurfaceRB = reflectedRB.GetRaysAt(i-1)

            if(self.IsPhysicalSurface(i)):
                _surfaceRB, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                    inSurfaceRB, 
                    self._FindPreviousRI(i, inSurfaceRB), 
                    reflection = True,
                    useClearBoundary = True)

                # _surfaceRB would be the ones that entered surface i
                _surfaceRB.SetIndex(i)

                returnRB.Merge(_surfaceRB)

            """========================================================================="""

            # Reset inSurfaceRB and append the reflected rays if previous step generated any
            inSurfaceRB = reflectedRB.GetRaysAt(i - 1)
            if (not _reflectedRB.IsNone()): inSurfaceRB.Merge(_reflectedRB)

            if(self.IsPhysicalSurface(i - 1)):
                # Try to find the ones that will interact with the previous surface
                _surfaceRB, _tir, _vig, _reflectedRB = self.surfaces[i - 1].Trace(
                    inSurfaceRB,
                    self._FindPreviousRI(i - 1, inSurfaceRB),
                    inverted = True,
                    reflection = True,
                    useClearBoundary = False)
                
                # _surfaceRB are rays that are refracted into the next space 
                _surfaceRB.SetIndex(i - 1)

                returnRB.Merge(_surfaceRB)
                returnRB.Merge(_reflectedRB)


        return returnRB, None


    def _BounceReflectionAlt(self, reflectedRB):
        """Iterate through each surface, calculate both reflection and refractions."""

        holderRB = RayBatch(None)

        # Use this as a holder RB for iterations. I hate how things have to be this complicated and how Cupy likely will make a mess out of this variable due to their memory management...
        iterRB = RayBatch(None)

        """======================= Propagate backwards ======================="""
        # From last surface back toward the first, carrying each surface's backward reflection into the previous surface's space by refraction.
        for i in range(len(self.surfaces) - 1, 0, -1):

            # Skip this surface if it's a stop or other virtual surface type
            if not self.IsPhysicalSurface(i):
                continue

            # Get only the rays facing backward (toward previous surface) and merge them into the backward propagation RB
            iterRB.Merge(reflectedRB.GetRaysAt(i).GetDirectionalRay(False))

            previousSurfaceIndex = self._PreviousSurfaceIndex(i)

            # Trace these rays at the previous surface, i.e., send them backward.
            iterRB, _tir, _vig, _reflectedRB = self.surfaces[previousSurfaceIndex].Trace(
                iterRB,
                self._FindPreviousRI(previousSurfaceIndex, iterRB),
                inverted=True,
                reflection=True,
                useClearBoundary=False
            )

            #print("Reflected in ", i ,"th surface: ", _reflectedRB.PolarizedRadiance().sum())

            # The iterRB should now be in the previous surface space
            iterRB.SetIndex(previousSurfaceIndex)

            # Append the reflected rays in this surface into the return RB for record. During the backward propagation, holder is only used to hold the scatter reflections in each surface tracing.
            holderRB.Merge(_reflectedRB)

        # After the backward tracing is done, append the iterRB into the holder as well. holder now should have both the scattered reflections in each surface and the refracted rays that reached the first surface from the last.
        holderRB.Merge(iterRB)

        #print("Total rad forward: ", holderRB.GetDirectionalRay().PolarizedRadiance().sum())
        #print("Total rad backward: ", holderRB.GetDirectionalRay(False).PolarizedRadiance().sum())

        # Since iterRB is already appended into the main holder, it's safe to reset it for next iteration.
        iterRB = RayBatch(None)

        #print("Inside alt method: \n" + holderRB.SurfaceDistributionInfo())
        #print(holderRB.value.shape)

        #self.DrawLens() #Drawcall ========================================================================
        #SetUnifScale(50)  # Drawcall ========================================================================
        #AddXYZ()  # Drawcall ========================================================================
        #RemoveBG()  # Drawcall ========================================================================

        remainRB = RayBatch(None)
        """======================= Propagate forwards ======================="""
        # Starting from the first physical surface, carry the backward-reflected rays forward through the stack, together with any forward-facing content already present in reflectedRB at each surface.
        for i in range(1, len(self.surfaces)):

            # Skip this surface if it's a stop or other virtual surface type
            if not self.IsPhysicalSurface(i): continue

            # Get rays in the currently surface space that are also facing to the image side, then merge these rays into the iterRB
            iterRB.Merge(holderRB.GetRaysAt(i - 1).GetDirectionalRay())
            iterRB.Merge(reflectedRB.GetRaysAt(i - 1).GetDirectionalRay())

            #print("=== Rad at ", i, " before forward prop size ", iterRB.GetRaysAt(i-1).value.shape, " with rad",  iterRB.GetRaysAt(i-1).PolarizedRadiance().sum())
            #print("At surface index ", i)


            #DrawRaybatch(iterRB.GetRaysAt(i-1), lLength=1)#Drawcall ========================================================================

            # Send the iterRB forward (towards image direction)
            iterRB, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                iterRB,
                self._FindPreviousRI(i, iterRB),
                reflection=True,
                useClearBoundary=True
            )

            iterRB.SetIndex(i)

            # Put the unused rays into remainRB
            remainRB.Merge(_reflectedRB)

            #DrawRaybatch(iterRB.GetRaysAt(i), lLength=1)#Drawcall ========================================================================
            #plt.draw()#Drawcall ========================================================================
            #plt.pause(10)#Drawcall ========================================================================

            #print("    Rad at ", i, " after forward prop ", iterRB.GetRaysAt(i).value.shape, " with rad", iterRB.GetRaysAt(i).PolarizedRadiance().sum())

        return iterRB, remainRB


    def _PropagateReflectedThrough(self, reflectedRB):
        """
        Propagate the rays in reflectedRB in each surface through the lens.
        """
        returnRB = RayBatch(None)

        for i in range(1, len(self.surfaces)):

            if(not isinstance(self.surfaces[i], Stop)):

                # Find the rays that are in the current surface space.
                # Since the index means "which surface is the ray located after", a minus one is needed.
                inSurfaceRB = reflectedRB.GetRaysAt(i-1)
                returnRB.Merge(inSurfaceRB)

                # Explicitly disable the reflection 
                returnRB, _tir, _vig, _reflectedRB = self.surfaces[i].Trace(
                        returnRB, 
                        self._FindPreviousRI(i, returnRB), 
                        reflection = False)
                

        return returnRB


    def IsPhysicalSurface(self, index):
        """
        Quick check whether the given surface index corresponds to a physical surface.
        """

        givenSurface = self.surfaces[index]

        return not isinstance(givenSurface, Stop)




def main():

    test = Lens() 




    


if __name__ == "__main__":
    main()