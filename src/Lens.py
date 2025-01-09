


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Util.PlotTest import DrawSpherical, DrawRaybatch, DrawPoint, DrawDirection, DrawPoints
from Util.Backend import constant
from Util.Backend import backend as bd 
from Util.Globals import ONE, TWO, Axis
from Util.Misc import AxialDistance
from Surfaces import Surface, Stop
from Surfaces.Pupil import Pupil
from Material import Material
from Raytracing.RayBatch import RayBatch
from Raytracing.Raypath import RayPath
from Raytracing.Emission import EmitFromStop, EmitFromObjectSpace


class Lens:
    def __init__(self):
        self.entrancePupilDia = constant(25)

        self.surfaces = []
        self.env = Material("AIR") # The environment it is submerged in, air by default 
        
        self.rayBatch = RayBatch([])
        self.rayPath = [] # Rays with only position info on each surface 
        
        self.entrancePupil = Pupil() 

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
            self._lastSurfaceIndex = i

        # Total axial length, counting from the first surface vertex to the last  
        self.totalAxialLength = currentT

        # make sure the surfaces are already set before calling init ray tracing 
        self.LensStatRayTracing() 


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
                self.surfaces[i].drawSurface()


    def SaveLens(self, path):
        """
        Save the lens and its parameters to an offline file. 
        """

        pass

    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _FindPreviousRI(self, index, raybatch, inverted = False):
        """
        Find the refractive index of the previous lens element. This is different from surfaces, this method tries to find the lens element, which has front and back surfaces.
        """
        if (index == 0):
            return self.env.RI(raybatch.Wavelength())
        else:
            return self.surfaces[index -1].RI(raybatch.Wavelength())
        


def main():
    singlet = Lens() 


    


if __name__ == "__main__":
    main()