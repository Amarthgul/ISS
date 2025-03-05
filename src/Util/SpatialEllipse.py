

import matplotlib.pyplot as plt


from Util.Backend import backend as bd 
from Util.Globals import Axis, ZERO, UP_DIR, HORI_DIR
from Util.PltPlot import DrawClearBoundary






class SpatialEllipse():
    """
    This class is mostly for marking and calculating the clear boundary. 
    """
    # Although the name may be associated with the polarized raidance ellipse that pop up often in this project, this class is not for that thing. 
    # The polarized raidance ellipse is 2D for simplicity and because the polarization is local to the ray, but this spatial ellipse is 3D. 


    def __init__(self, C, u, v, a, b):
        self.center = C
        self.semiAxisDirU = u
        self.semiAxisDirV = v
        self.semiAxisMagA = a
        self.semiAxisMagB = b 

        self.isCircular = True


    def FacingDirection(self):
        """
        Calculate the facing direction of the ellipse. Note that depending on the order of the 2 semi axis, this direction could be pointing towards either the positive or the negaive Z. 
        """
        return bd.cross(self.semiAxisDirU, self.semiAxisDirV)


    def SemiAxisDirection(self):
        if(self.isCircular):
            return self.semiAxisDirU
        else:
            return self.semiAxisDirU, self.semiAxisDirV
        

    def SemiAxisMagnititude(self):
        if (self.isCircular):
            return self.semiAxisMagA
        else:
            return self.semiAxisMagA, self.semiAxisMagB    
    

    def ZCoord(self):
        return self.center[Axis.Z.value]


class SpatialCircle(SpatialEllipse):
    """
    Simplified version of the spatial ellipse.
    """
    def __init__(self, z, r):
        self.ZCoord = bd.array(z)
        self.radius = bd.array(r)

        self.center = bd.array([ZERO, ZERO, self.ZCoord])
        self.semiAxisDirU = bd.copy(UP_DIR)
        self.semiAxisDirV = bd.copy(HORI_DIR)
        self.semiAxisMagA = self.radius
        self.semiAxisMagB = self.radius
        



def main():

    E1 = SpatialEllipse(bd.array([0, 0, 0]), 
                        bd.array([1, 0, 0]),
                        bd.array([0, 1, 0]), 
                        10, 10)
    
    E2 = SpatialEllipse(bd.array([0, 0, 5]), 
                        bd.array([1, 0, 0]), 
                        bd.array([0, 1, 0]), 
                        15, 15)

    C1 = SpatialCircle(0, 10)
    C2 = SpatialCircle(10, 10)

    DrawClearBoundary(C1, C2)
    plt.show()

    print(E1.FacingDirection())


if __name__ == "__main__":
    main()