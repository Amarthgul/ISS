

import PlotTest
from Surfaces import Surface
from Util import * 
from RayBatch import * 
from Material import * 


import time
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




class Lens:
    def __init__(self):
        self.entrancePupilDia = 25 

        self.surfaces = []
        self.env = "AIR" # The environment it is submerged in, air by default 
        
        self.rayBatch = RayBatch([])
        self.rayPath = [] # Rays with only position info on each surface 

        self.totalLength = 0

        self.spot = [] 
        

        self.lastSurfaceIndex = 0

        self._envMaterial = None 
        self._temp = None # Variable for developing and not to be taken serieously 


    def UpdateLens(self):
        """
        Iterate throught the elements and update their relative parameters. 
        Including starting a ray trace and finds the entrance and exit pupil. 
        """
        
        currentT = 0

        for s in self.surfaces:

            currentT += s.thickness

        self.InitStopRayTracing() 


    def InitStopRayTracing(self):
        """
        At the position of the stop, start a ray tracing and try to find the pupils.
        """
        pass 



def main():
    singlet = Lens() 


    


if __name__ == "__main__":
    main()