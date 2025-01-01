


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Util.PlotTest import DrawSpherical
from Util.Backend import constant
from Util.Backend import backend as bd 
from Surfaces import Surface, Stop
from Raytracing.RayBatch import RayBatch
from Raytracing.Emission import EmitFromStop 


class Lens:
    def __init__(self):
        self.entrancePupilDia = 25 

        self.surfaces = []
        self.env = "AIR" # The environment it is submerged in, air by default 
        
        self.rayBatch = RayBatch([])
        self.rayPath = [] # Rays with only position info on each surface 
        
        self.totalAxialLength = None 

        self.lastSurfaceIndex = 0

        self._envMaterial = None 
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
        self.InitRayTracing() 


    def InitRayTracing(self):
        """
        At the position of the stop, start a ray tracing and try to find the pupils.
        """

        # Sicne this method is only called once during setup, it is not written very efficiently.

        objectSideRB = None 
        imageSideRB = None
        stopIndex = None # Array index of the stop among the surfaces 

        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Stop):
                objectSideRB, imageSideRB = EmitFromStop(
                    i,
                    self.surfaces[i].frontVertex,
                    self.surfaces[i-1].clearSemiDiameter,
                    self.surfaces[i+1].clearSemiDiameter,
                    self.surfaces[i-1].sdThickness,
                    self.surfaces[i+1].sdThickness
                )
                stopIndex = i
                break



    def DrawSurfaces(self):

        for i in range(len(self.surfaces)):
            if not isinstance(self.surfaces[i], Stop):
                DrawSpherical(
                    self.surfaces[i].radius,
                    self.surfaces[i].clearSemiDiameter,
                    self.surfaces[i].cumulativeThickness
                    )



def main():
    singlet = Lens() 


    


if __name__ == "__main__":
    main()