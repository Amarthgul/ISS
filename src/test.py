

import matplotlib.pyplot as plt

from Lens import Lens 
from Surfaces import Surface 
from Raytracing import Emission 
from Util.Backend import backend as bd
from Util.Backend import GetBackend
from Util.PlotTest import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical

import cupy as cp 

def main():
    GetBackend()



    r = 20
    sd = 4 
    testP = bd.array([1, 2, -10])

    rb = Emission.InitRays(r, sd, testP)
    #print(rb.value)

    ax = Setup3Dplot()
    SetUnifScale(ax)
    DrawRaybatch(ax, rb)
    DrawSpherical(ax, r, sd, 0)
    plt.show()

    


if __name__ == "__main__":
    main()