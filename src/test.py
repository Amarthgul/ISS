

import matplotlib.pyplot as plt

from Lens import Lens 
from Surfaces import Surface 
from Raytracing import Emission 
from Raytracing.Raypath import RayPath
from Util.Backend import backend as bd
from Util.Backend import GetBackend, constant
from Util.PltPlot import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints, DrawNormal, RemoveBG, DrawDisk
from Util.Globals import ZERO, ONE, TWO

from ExampleLenses import Biotar50mmf14, Helios58mmf2


def SurfaceTest():
    rp = RayPath()

    r = constant(20)
    sd = constant(4)
    testP = bd.array([1, 2, -10])

    rb = Emission.InitRays(r, sd, testP)
    #print(rb.value)

    rp.Append(rb, None, None)

    testSurface = Surface(r, ZERO, sd, "BAF9")
    testSurface.SetCumulative(ZERO)
    refracted, reflected, vig = testSurface.NaiveTrace(rb, ONE)
    rp.Append(refracted, reflected, vig)

    testSurface = Surface(r, ZERO, sd, "BAF9")
    testSurface.SetCumulative(constant(3))
    refracted, reflected, vig = testSurface.NaiveTrace(refracted, ONE)
    rp.Append(refracted, reflected, vig)


    SetUnifScale()
    #DrawRaybatch(rb, length=11.5)
    #DrawRaybatch(refracted, length=2)
    DrawSpherical(r, sd, constant(0))
    DrawSpherical(r, sd, constant(3))
    rp.PlotPath()
    

    plt.show()



def main():

    SetUnifScale(50)
    AddXYZ()

    testLens = Biotar50mmf14()
    testLens.UpdateLens()


    testLens.DrawLens()
    testLens.entrancePupil.DrawSurface()
    testLens.frontPincipalPlane.DrawSurface()

    RemoveBG()
    #DrawDisk(19)
    plt.show()
    
    


if __name__ == "__main__":
    main()