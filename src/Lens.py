


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Util.PltPlot import DrawSpherical, DrawRaybatch, DrawPoint, DrawDirection, DrawPoints
from Util.Backend import constant
from Util.Backend import backend as bd 
from Util.Globals import ZERO, ONE, TWO, Axis, LambdaLines
from Util.Misc import AxialDistance, WavelengthToRGB, TransversalDistance
from Surfaces import Surface, Stop
from Surfaces.Pupil import Pupil
from Surfaces.PrincipalPlane import PrincipalPlane
from Material import Material
from Raytracing.RayBatch import RayBatch, GenerateEmpty
from Raytracing.Raypath import RayPath
from Raytracing.Emission import EmitFromStop, EmitFromObjectSpace, EmitFromPoint


class Lens:
    def __init__(self):
        self.entrancePupilDia = constant(25.0)

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
        self.entrancePupil.clearSemiDiameter = self.entrancePupilDia/TWO


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

        # make sure the surfaces are already set before calling init ray tracing 
        #self.LensStatRayTracing() 
        self._TraceFocalPrincipal() 
        self._TraceEntrancePupil()
        

    def LensStatRayTracing(self):
        """
        At the position of the stop, start a ray tracing and try to find the pupils, the principal planes and the nodal points.
        """

        # Enable ray path for debugging
        _EnableRayPath = True

        # Sicne this method is only called once during setup, it is not written very efficiently.


        objectSideRB = None 
        imageSideRB = None
        stopIndex = None # Array index of the stop among the surfaces 

        if(_EnableRayPath):
            objectSideRP = RayPath()
            imagesideRP = RayPath()

        # Find the stop and generate a raybatch from the center of it
        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Stop):
                objectSideRB, imageSideRB = EmitFromStop(
                    i,
                    self.surfaces[i].frontVertex,
                    self.surfaces[i-1].clearSemiDiameter,
                    self.surfaces[i+1].clearSemiDiameter,
                    self.surfaces[i-1].sdThickness,
                    self.surfaces[i+1].sdThickness, 
                    numRays=40
                )
                stopIndex = i
                if(_EnableRayPath):
                    objectSideRP.Append(objectSideRB)
                    imagesideRP.Append(imageSideRB)
                break

        # Propagate the rays through the lens in both directions
        for i in range(len(self.surfaces)):
            forwardIndex = stopIndex - i - 1
            backwardIndex = stopIndex + i + 1
            #print("Currently in ", i, " th iteration")
            if(forwardIndex >= 0):
                #self.surfaces[forwardIndex].DrawSurface() # Draw call=========
                objectSideRB, _tir, _vig = self.surfaces[forwardIndex].NaiveTrace(
                    objectSideRB, 
                    self._FindPreviousRI(forwardIndex, objectSideRB)
                    )
                if (_EnableRayPath):
                    objectSideRP.Append(objectSideRB, _tir, _vig)
                #DrawRaybatch(objectSideRB) # Draw call=========
                
            if(backwardIndex <= self._lastSurfaceIndex):
                #self.surfaces[backwardIndex].DrawSurface() # Draw call=========
                imageSideRB, _tir, _vig = self.surfaces[backwardIndex].NaiveTrace(
                    imageSideRB, 
                    self._FindPreviousRI(backwardIndex, imageSideRB, True)
                    )
                if (_EnableRayPath):
                    imagesideRP.Append(imageSideRB, _tir, _vig)
                #DrawRaybatch(imageSideRB) # Draw call=========

            #plt.draw()
            #plt.pause(5)

        pos, dir = objectSideRP.ExitingPairs(True)
        DrawDirection(pos, dir, lineLength=30) # Draw call=========
        entPoint = objectSideRP.FindConvergingPoint(pos, dir)
        self.entrancePupil.AddSamplePoint(entPoint)
        self.entrancePupil.DrawSurface()
        


        if (_EnableRayPath):
            frontRP = RayPath()

        # Principal plane 
        frontRB = EmitFromObjectSpace(self.entrancePupilDia / TWO, numRays=60)
        for i in range(len(self.surfaces)):
            if(not isinstance(self.surfaces[i], Stop)):
                self.surfaces[i].DrawSurface() # Draw call=========
                frontRB, _tir, _vig = self.surfaces[i].NaiveTrace(
                    frontRB, 
                    self._FindPreviousRI(i, frontRB)
                )
                if (_EnableRayPath):
                    frontRP.Append(frontRB, _tir, _vig)

        frontRP.DrawPath(10.0)
        zIntersections = frontRP.DepthIntersect(entPoint)
        pupilPlaneDiameter = AxialDistance(zIntersections, Axis.Y.value)

        print("Dia pupils: ", pupilPlaneDiameter)

        if (_EnableRayPath):
            #frontRP.PlotPath(expendEnd = 10)
            #imagesideRP.PlotPath(expendEnd = 10)
            #objectSideRP.PlotPath(expendEnd = 10)
            pass 


    def DrawLens(self):
        """
        Iterate through the surfaces and draw them.
        """
        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                self.surfaces[i].DrawSurface()


    def SetAperture(self, aperture):
        """
        Set the aperture of the lens. 

        :param aperture: f-number of the lens.
        """
        
        # TODO: add some algorithms to include the shape of aperture blades 

        self.entrancePupilDia = self.focalLength / aperture

        

    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _TraceEntrancePupil(self, stopSD=10.0, sampleCount=11, wavelength = LambdaLines['D']):
        """
        Start rays from the center of the stop, casting forward and trace the size and location of the entrance pupil. 

        :param stopSD: Semi diameter of the stop. 
        :param sampleCount: number of sample sets to be emitted from the stop. 
        :param wavelength: the wavelength at which the calculations will be performed. 
        """

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
        #     objectSideRPs[j].DrawPath(10, colors[j%len(colors)])

        poss = [[]for i in range(sampleCount)]
        dirs = [[]for i in range(sampleCount)]
        intersections = [[]for i in range(sampleCount)]
        for j in range(sampleCount):
            poss[j], dirs[j] = objectSideRPs[j].ExitingPairs()
            intersections[j] = objectSideRPs[j].FindConvergingPoint(poss[j], dirs[j])
            #DrawPoint(intersections[j], color=colors[j])
            #DrawDirection(poss[j], dirs[j], lineColor=colors[j], lineLength=40, arrowRatio=0)
        self.entrancePupil.SetSamplePoints(bd.array(intersections))


    def _TraceFocalPrincipal(self, wavelength = LambdaLines['D']):
        """
        Trace the front focal point and principal plane. These two will then be used to calculate the focal length. 

        :param wavelength: the wavelength at which the calculations will be performed. 
        """
        self.frontPincipalPlane.sampleWavelength = wavelength

        frontRB = EmitFromObjectSpace(self.entrancePupilDia / TWO, numRays=15, halfSide=True)
        frontRP = RayPath()
        frontRP.Append(frontRB, None, None)

        for i in range(len(self.surfaces)):
            if(not isinstance(self.surfaces[i], Stop)):
                frontRB, _tir, _vig = self.surfaces[i].NaiveTrace(
                    frontRB, self._FindPreviousRI(i, frontRB))   
                frontRP.Append(frontRB, _tir, _vig)  

        frontRP = frontRP.PruneAll()
        #frontRP.DrawPath(40) # Draw call =======
        
        pos, dir = frontRP.ExitingPairs()
        self.focalPoint = frontRP.FindConvergingPoint(pos, dir)
        # DrawPoint(self.focalPoint) # Draw call =======

        intersections = frontRP.EndToEndIntersection() 
        intersections = intersections[~bd.any(bd.isnan(intersections), axis=1)]
        self.frontPincipalPlane.SetSamplePoints(intersections)
        

        minPoint = TransversalDistance(intersections)
        minPoint = intersections[bd.argmin(minPoint)]
        minPoint = bd.array([ZERO, ZERO, minPoint[Axis.Z.value]])
        #self.frontPincipalPlane.AddSamplePoint(minPoint)
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