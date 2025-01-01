


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Util.PlotTest import DrawSpherical, DrawRaybatch
from Util.Backend import constant
from Util.Backend import backend as bd 
from Util.Globals import ONE
from Surfaces import Surface, Stop
from Material import Material
from Raytracing.RayBatch import RayBatch
from Raytracing.RayPath import RayPath
from Raytracing.Emission import EmitFromStop 


class Lens:
    def __init__(self):
        self.entrancePupilDia = 25 

        self.surfaces = []
        self.env = Material("AIR") # The environment it is submerged in, air by default 
        
        self.rayBatch = RayBatch([])
        self.rayPath = [] # Rays with only position info on each surface 
        
        self.totalAxialLength = None 

        self.lastSurfaceIndex = 0

        self._temp = None # Variable for developing and not to be taken serieously 



    def AddSurface(self, inputSurface):
        self.surfaces.append(inputSurface)


    def UpdateLens(self):
        """
        Iterate throught the elements and update their relative parameters. 
        Including starting a ray trace and finds the entrance and exit pupil. 
        """
        
        currentT = constant(0.0)

        for i in range(len(self.surfaces)):
            self.surfaces[i].SetCumulative(bd.copy(currentT))
            if(i != len(self.surfaces)-1):
                # The last surface's thickness of a lens is not useful for the lens 
                currentT += self.surfaces[i].thickness
            self.lastSurfaceIndex = i

        # Total axial length, counting from the first surface vertex to the last  
        self.totalAxialLength = currentT

        # make sure the surfaces are already set before calling init ray tracing 
        self.LensStatRayTracing() 


    def LensStatRayTracing(self):
        """
        At the position of the stop, start a ray tracing and try to find the pupils, the principal planes and the nodal points.
        """

        # Sicne this method is only called once during setup, it is not written very efficiently.

        objectSideRB = None 
        imageSideRB = None
        stopIndex = None # Array index of the stop among the surfaces 

        
        objectSideRP = RayPath()
        imagesideRP = RayPath()

        # Generate a raybatch from the center of the stop
        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Stop):
                objectSideRB, imageSideRB = EmitFromStop(
                    i,
                    self.surfaces[i].frontVertex,
                    self.surfaces[i-1].clearSemiDiameter,
                    self.surfaces[i+1].clearSemiDiameter,
                    self.surfaces[i-1].sdThickness,
                    self.surfaces[i+1].sdThickness, 
                    numRays=30
                )
                stopIndex = i
                objectSideRP.Append(objectSideRB)
                imagesideRP.Append(imageSideRB)
                break

        # Propagate the rays through the lens in both directions
        for i in range(len(self.surfaces)):

            forwardIndex = stopIndex - i - 1
            backwardIndex = stopIndex + i + 1

            if(forwardIndex >= 0):
                self.surfaces[forwardIndex].DrawSurface()
                objectSideRB, _tir, _vig = self.surfaces[forwardIndex].NaiveTrace(
                    objectSideRB, 
                    self._FindPreviousRI(forwardIndex, objectSideRB)
                    )
                objectSideRP.Append(objectSideRB, _tir, _vig)
                
            if(backwardIndex <= self.lastSurfaceIndex):
                self.surfaces[backwardIndex].DrawSurface()
                imageSideRB, _tir, _vig = self.surfaces[backwardIndex].NaiveTrace(
                    imageSideRB, 
                    self._FindPreviousRI(backwardIndex, imageSideRB, True)
                    )
                imagesideRP.Append(imageSideRB, _tir, _vig)

        objectSideRP.PlotPath(expendEnd = 10)
        imagesideRP.PlotPath(expendEnd = 10)


    def DrawLens(self):

        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                self.surfaces[i].drawSurface()


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