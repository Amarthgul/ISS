

import PlotTest
from Surface import *
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
        Iterate throught the elements and update their relative parameters 
        """
        
        currentT = 0

        for s in self.surfaces:

            currentT += s.thickness




def main():
    singlet = Lens() 


    


if __name__ == "__main__":
    main()