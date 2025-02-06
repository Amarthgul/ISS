


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Util.PltPlot import DrawSpherical, DrawRaybatch, DrawPoint, DrawDirection, DrawPoints
from Util.Backend import constant
from Util.Backend import backend as bd 
from Util.Globals import ZERO, ONE, TWO, Axis, LambdaLines, AXIAL_ZERO
from Util.Misc import AxialDistance, WavelengthToRGB, TransversalDistance
from Surfaces.Stop import Stop
from Surfaces.Pupil import Pupil
from Surfaces.PrincipalPlane import PrincipalPlane
from Material import Material
from Raytracing.RayBatch import RayBatch, GenerateEmpty
from Raytracing.Raypath import RayPath
from Raytracing.Emission import EmitFromStop, EmitFromObjectSpace, EmitFromPoint, EmitField


class Lens:
    def __init__(self):
        self.fNumber = constant(2.0)
        self.focalLength = constant(50.0)

        self.focalPoint = None 

        self.surfaces = []
        self.env = Material("AIR") # The environment it is submerged in, air by default 
        
        self.rayBatch = RayBatch([])
        self.rayPath = [] # Rays with only position info on each surface 
        
        self.entrancePupil = Pupil() 
        self.frontPincipalPlane = PrincipalPlane() 

        """Index of stop amopng the lens surfaces for easier indexing"""
        self.stopIndex = None 

        """Axial direction length of all the surfaces. From front vertex of the first surface to the last surface"""
        self.totalAxialLength = None 

        self._lastSurfaceIndex = 0

        self._temp = None # Variable for developing and not to be taken serieously 



    def AddSurface(self, inputSurface):
        self.surfaces.append(inputSurface)


    def UpdateLens(self):
        """
        Iterate throught the elements and update their relative parameters. 
        Including starting a ray trace and finds the entrance and exit pupil. 
        """
        #self.entrancePupil.clearSemiDiameter = self.entrancePupilDia/TWO


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
        

    def SetIncidentRaybatch(self, raybatch):
        """
        Set the incident raybatch for the lens. 

        :param raybatch: the raybatch to be used to propagate through the lens.
        """
        self.rayBatch = raybatch


    def Propagate(self):
        """
        Propagate the raybatch through the lens. 
        """

        # For production use, turn this off to avoid unnecessary memory usage.
        recordPath = False

        self.rayPath = RayPath()
        if(recordPath):
            self.rayPath.Append(self.rayBatch, None, None)

        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                self.rayBatch, _tir, _vig = self.surfaces[i].Trace(
                    self.rayBatch, self._FindPreviousRI(i, self.rayBatch))
                
                if(recordPath):
                    self.rayPath.Append(self.rayBatch, _tir, _vig)

        # self.rayPath.DrawPath(40)
        return self.rayBatch, self.rayPath


    def BestFocusBFD(self, distance):
        """
        Calculate the best back focal distance given an object distance. This is achieved by finding the smallest RMS spot position in the exit rays. 
        """
        
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
            "Focal Length:   \t" + str(self.focalLength) + "\n" +\
            "Max working N:  \tf/" + str(self.focalLength / self.entrancePupil.GetMaxPupilSize()) + "\n" +\
            "Max pupil dia:  \t" + str(self.entrancePupil.GetMaxPupilSize()) + "\n" +\
            "Axial length:   \t" + str(self.totalAxialLength) + "\n" +\
            "Principal plane:\t" + str(self.frontPincipalPlane.GetInnerZ()) + "\n" +\
            "Focal point:    \t" + str(self.focalPoint[Axis.Z.value]) + "\n"
            
        return info


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


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

        # TODO: if the stop is constant and is the same with one of the surfaces, this might be undefined. Add support for 0-d stop. 
        # Generate raybatch and add them into the path record
        objectSideRBs = [
            EmitFromPoint(
            self.surfaces[self.stopIndex].frontVertex+bd.array([ZERO, p, ZERO]),
            targetOne, targetTwo, wavelength = wavelength)
            for p in bd.linspace(ZERO, stopSD, sampleCount)
        ]
        for j in range(sampleCount):
            objectSideRPs[j].Append(objectSideRBs[j], None, None)
        
        # Propogate through the surfaces 
        for i in range(len(self.surfaces)):
            forwardIndex = self.stopIndex - i - 1
            if(forwardIndex >= 0):
                print("At surface ", forwardIndex)
                #self.surfaces[forwardIndex].DrawSurface() # Draw call=========
                for j in range(sampleCount):
                    objectSideRBs[j], _tirs[j], _vigs[j] = self.surfaces[forwardIndex].NaiveTrace(
                        objectSideRBs[j], 
                        self._FindPreviousRI(forwardIndex, objectSideRBs[j]), True
                        ) 
                    objectSideRPs[j].Append(objectSideRBs[j], _tirs[j], _vigs[j])
                #DrawRaybatch(objectSideRB)  # Draw call=========
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

        #frontRP.DrawPath(40) # Draw call =======
        #plt.draw() # Draw call =======
        #plt.pause(10) # Draw call =======
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
        




def main():
    singlet = Lens() 


    


if __name__ == "__main__":
    main()