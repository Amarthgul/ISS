

import matplotlib.pyplot as plt

from Util.Backend import backend as bd 
from Util.Backend import constant
from Util.SpatialEllipse import SpatialEllipse
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, SetUnifScale, RemoveBG, AddXYZ, DrawEllipse, DrawClearBoundary



class ClearBoundary():
    """
    A clear boundary is the cylinder shaped surface that connects 2 lens surfaces. 
    """

    def __init__(self, E1, E2):

        """2 rings of class SpatialEllipse that defines this clear boundary"""
        self.E1 = E1 
        self.E2 = E2


        """Describes the material at the other side. When set to None, treat it as Air; it can also be set to a constant float that represents IOR; alternatively it could be a material class."""
        self.exteriorCoating = None 


        """Weight of [0, 1] that controls total diffuse and total mirror reflection. When set to 0, surface reflects as lambertian, when set to 1, reflects like mirror"""
        self.specularReflection = 0.5


    def DrawSurface(self):
        DrawClearBoundary(self.E1, self.E2)
        


    def Intersection(self, incomingRaybatch):
        pass 


    def Normal(self, intersections):
        pass 


    def Trace(self, incidentRaybatch):
        pass 


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    


    
    




def main():

    E1 = SpatialEllipse(bd.array([0, 0, 0]), 
                        bd.array([1, 0, 0]),
                        bd.array([0, 1, 0]), 
                        10, 10)
    
    E2 = SpatialEllipse(bd.array([0, 0, 5]), 
                        bd.array([1, 0, 0]), 
                        bd.array([0, 1, 0]), 
                        15, 15)

    testCB = ClearBoundary(E1, E2)
    testCB.DrawSurface()

    
    SetUnifScale(50)
    #AddXYZ()
    #RemoveBG()
    plt.show()


if __name__ == "__main__":
    main()