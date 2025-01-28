

import matplotlib.pyplot as plt
import time

from Surfaces.Surface import Surface
from Surfaces.Stop import Stop
from Lens import Lens 
from Util.Globals import ZERO, ONE, INFINITY
from Util.DataReadWrite import Save, Load
from Util.Backend import backend_name
from Util.PltPlot import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints, DrawNormal, RemoveBG, DrawDisk


# When flagged, lenses will be loaded from file rather than calculated 
LOAD_LENS_FROM_FILE = False 


class Example():
    def __init__(self, input=None, fileName=None):
        self.data = input 
        self.fileName = fileName

    def SaveExample(self):
        Save(self.data, self.fileName)

    def LoadExample(self):
        self.data = Load(self.fileName)


# ==================================================================
""" ==================== Zeiss Biotar 50mm f/1.4 =============== """
# ==================================================================


def _Biotar50mm14Data():
    """
    Zeiss Biotar 500mm f/1.4. 
    Data from US 1786916 Example 2, EFL 100mm. 

    :return: Lens object with only data and not initlized.
    """
    biotar = Lens() 

    biotar.AddSurface(Surface(41.8,     5.375,      18,     "BAF9"))
    biotar.AddSurface(Surface(160.5,    0.825,      18))
    biotar.AddSurface(Surface(22.4,	    7.775,      16,     "SK10"))
    biotar.AddSurface(Surface(-575,	    2.525,      16,     "LZ_LF5"))
    biotar.AddSurface(Surface(14.15,	5.45,       11))
    biotar.AddSurface(Stop(             4))
    biotar.AddSurface(Surface(-19.25,	2.525,      11,     "SF5"))
    biotar.AddSurface(Surface(25.25,	10.61,      14,     "BAF9"))
    biotar.AddSurface(Surface(-26.6,	0.485,      14))
    biotar.AddSurface(Surface(53, 	    6.95,       14.5,   "BAF9"))
    biotar.AddSurface(Surface(-60,	    32.3552,    14.5))

    return biotar 


def Biotar50mmf14():
    """
    Zeiss Biotar 500mm f/1.4. 

    :return: initlized lens object.
    """
    fileName = "ZeissBiotar50mmf1.4"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    biotarExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        biotarExample.LoadExample()
        biotar = biotarExample.data
    else: 
        biotar = _Biotar50mm14Data()
        biotar.UpdateLens()
        biotarExample.data = biotar
        biotarExample.SaveExample()

    return biotar


# ==================================================================
""" ====================== Helios-44 58mm f/2 ================= """
# ==================================================================


def _Helios58mmf2Data():
    """
    Data from Bill Claff.
    """
    helios = Lens()

    helios.AddSurface(Surface(38.070,	4.810,  15,     "LZ_TK14"))
    helios.AddSurface(Surface(136.365,	2.260,  15))
    helios.AddSurface(Surface(25.330,	9.070,  13,     "LZ_TK14"))
    helios.AddSurface(Surface(-124.225,	1.310,  13,     "LZ_LF7"))
    helios.AddSurface(Surface(15.995,	4.700,  10.5))
    helios.AddSurface(Stop(             4.63))
    helios.AddSurface(Surface(-16.620,	1.320,  10.5,   "LZ_LF7"))
    helios.AddSurface(Surface(66.085,	6.250,  13,     "LZ_TK14"))
    helios.AddSurface(Surface(-22.210,	0.500,  13))
    helios.AddSurface(Surface(191.540,	4.940,  13.25,  "LZ_BF16"))
    helios.AddSurface(Surface(-52.725,	37.120, 13.25))

    #helios.entrancePupilDia = 29.5

    return helios


def Helios58mmf2():
    """
    Helios 44, 58mm f/2. 
    
    :return: initlized lens object.
    """

    fileName = "Helios-44-50mmf2"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    HeliosExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        HeliosExample.LoadExample()
        helios = HeliosExample.data
    else: 
        helios = _Helios58mmf2Data()
        helios.UpdateLens()
        HeliosExample.data = helios
        HeliosExample.SaveExample()

    return helios


# ==================================================================
""" ====================== Canon FD 50mm f/1.8 ================= """
# ==================================================================


def _CanonFD50mmf18Data():
    """
    Data from patent JP 1988-081312, Example ML. 
    Note that the original patent contains a teleconverter at the rear, the data below has removed it.
    """
    canon = Lens()

    canon.AddSurface(Surface(37.554,	3.10,   16,     "M-NBFD130"))
    canon.AddSurface(Surface(142.589,	0.29,   16))
    canon.AddSurface(Surface(20.991,	7.66,   14,     "F5"))
    canon.AddSurface(Surface(INFINITY,	1.46,   14,     "PBH4W"))
    canon.AddSurface(Surface(14.665,	5.50,   10))
    canon.AddSurface(Stop(              7.38))
    canon.AddSurface(Surface(-14.336,	1.07,   10,     "E-FD10"))
    canon.AddSurface(Surface(436.258,	4.85,   12,     "M-NBFD130"))
    canon.AddSurface(Surface(-18.860,	0.1,    12))
    canon.AddSurface(Surface(274.101,	2.91,   14,     "M-NBFD130"))
    canon.AddSurface(Surface(-44.781,	34.81,  14))

    return canon


def CanonFD50mmf18():
    """
    Canon FD 50mm f/1.8. 
    
    :return: initlized lens object.
    """

    fileName = "CanonFD50mmf1.8"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    CanonExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        CanonExample.LoadExample()
        canon = CanonExample.data
    else: 
        canon = _CanonFD50mmf18Data()
        canon.UpdateLens()
        CanonExample.data = canon
        CanonExample.SaveExample()

    return canon



def main():
    
    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()

    start = time.time()

    # TODO: this lens needs correction for the axial pupil position.
    exampleLens = CanonFD50mmf18()
    exampleLens.UpdateLens()

    end = time.time()
    print("When setting to ", LOAD_LENS_FROM_FILE, ", program took ", end-start, " to finish.")

    print(exampleLens.GetInfo())

    #exampleLens.DrawLens()
    #exampleLens.entrancePupil.DrawSamplePoints()
    #exampleLens.entrancePupil.DrawSurface()
    #exampleLens.frontPincipalPlane.DrawSamplePoints()
    
    #plt.show()

    

if __name__ == "__main__":
    main()