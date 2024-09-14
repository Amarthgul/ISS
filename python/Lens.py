

import PlotTest
from Surface import *

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Lens:
    def __init__(self):
        self.elements = []
        self.env = "AIR" # The environment it is submerged in, air by default 
        
        self.rayBatch = []


    def UpdateLens(self):
        """
        Iterate throught the elements and update them
        """
        
        currentT = 0

        for e in self.elements:
            e.SetCumulative(currentT)
            currentT += e.thickness


    def AddSurfacve(self, surface):
        self.elements.append(surface)


    def DrawLens(self, drawSrufaces = True, drawRays = False):
        ax = PlotTest.Setup3Dplot()
        PlotTest.SetUnifScale(ax)
        for l in self.elements:
            PlotTest.DrawSpherical(ax, l.radius, l.clearSemiDiameter, l.cumulativeThickness)
        
        plt.show()

    # ============================================================================
    """ ====================================================================== """
    # ============================================================================

    def _initRays(self, wavelengthSample = 3):
        pass 



def main():
    singlet = Lens() 
    singlet.AddSurfacve(Surface(20, 4, 6, "LAF8"))
    singlet.AddSurfacve(Surface(-10, 4, 6.6))
    singlet.UpdateLens()
    singlet.DrawLens()


if __name__ == "__main__":
    main()