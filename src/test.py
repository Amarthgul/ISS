

import matplotlib.pyplot as plt

from Lens import Lens 
from Surfaces import Surface 
from Raytracing import Emission 
from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO
from Util.Backend import GetBackend, constant
from Util.PlotTest import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints, DrawNormal

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
    refracted = testSurface.NaiveTrace(rb, ONE)[0]
    #intersections = testSurface.Intersection(rb)[0]
    #normals = testSurface.Normal(intersections)


    SetUnifScale()
    DrawRaybatch(rb, length=11.5)
    DrawRaybatch(refracted, length=2)
    DrawSpherical(r, sd, constant(0))
    #DrawPoints(intersections)
    #DrawNormal(intersections, normals)

    plt.show()

    


if __name__ == "__main__":
    main()