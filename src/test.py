

import matplotlib.pyplot as plt

from Lens import Lens 
from Surfaces import Surface 
from Raytracing import Emission 
from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO
from Util.Backend import GetBackend, constant
from Util.PlotTest import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints

import cupy as cp 

def main():
    GetBackend()



    r = constant(20)
    sd = constant(4)
    testP = bd.array([1, 2, -10])

    rb = Emission.InitRays(r, sd, testP)
    #print(rb.value)

    testSurface = Surface(r, ZERO, sd, "BAF9")
    testSurface.SetCumulative(ZERO)
    intersections = testSurface.Intersection(rb)

    SetUnifScale()
    DrawRaybatch(rb)
    DrawSpherical(r, sd, constant(0))
    DrawPoints(intersections)

    plt.show()

    


if __name__ == "__main__":
    main()