
"""
Bunch of already modelled lenses for testing purpose.

This is a really early file, should use ZMX loader instead for a much higher efficiency.

"""



import matplotlib.pyplot as plt
import time

from Surfaces.Surface import Surface
from Surfaces.EvenAspheric import EvenAspheric
from Surfaces.Stop import Stop
from Lens import Lens 
from Util.Globals import ZERO, ONE, INFINITY
from Util.DataReadWrite import Save, Load
from Util.Misc import RectPath
from Util.Backend import backend_name
from Util.PltPlot import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints, DrawNormal, RemoveBG, DrawDisk
from ZmxReader import LensFromZmx

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


def Mug():

    def _MugData():
        mug = Lens() 
        mug.AddSurface(Stop(                 0.1))
        mug.AddSurface(Surface(INFINITY,     20,      18,     "BAF9"))
        mug.AddSurface(Surface(INFINITY,     20,      18))
        mug.isAfocal = True 
        return mug 

    fileName = "aNormalMug"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    mugExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        mugExample.LoadExample()
        mug = mugExample.data
    else: 
        mug = _MugData()
        mug.UpdateLens()
        mugExample.data = mug
        mugExample.SaveExample()

    return mug



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
""" ====================== Canon EF 50mm f/1.2 ================= """
# ==================================================================

def _CanonEF50mmf12Data():
    canon = Lens()

    canon.AddSurface(Surface(61.844,    4.99, 22, "E-LASF016"))
    canon.AddSurface(Surface(411.251,   0.24, 22))
    canon.AddSurface(Surface(28.537,	   5.34, 20, "S-LAH55"))
    canon.AddSurface(Surface(41.757,    1.14, 20, ))
    canon.AddSurface(Surface(54.433,    2.16, 19, "PBM27"))
    canon.AddSurface(Surface(19.579,    12.95, 16))

    canon.AddSurface(Stop(7.41))

    canon.AddSurface(Surface(-23.181,   1.4,    15,     "S-TIH10"))
    canon.AddSurface(Surface(196.367,   7.64,   16,     "LAH58"))
    canon.AddSurface(Surface(-29.011,   0.45,   16  ))
    canon.AddSurface(Surface(-27.438,   1.5,    16,     "E-SF15"))
    canon.AddSurface(Surface(442.408,   6.48,   17,     "S-LAH55"))
    canon.AddSurface(Surface(-41.024,   0.15,   17  ))

    canon.AddSurface(EvenAspheric(146.157,   5.87,   18, "J-LASF015",
                                  0, [0.00000e+00, -1.445310E-06, 2.501600E-10, -1.461230E-13, 0.000000E+00]))

    canon.AddSurface(Surface(-61.524,   38.88,  18))

    return canon


def CanonEF50mmf12L():
    """
    Canon EF 50mm f/1.2 L.

    :return: initlized lens object.
    """

    fileName = "CanonEF50mmf1.2L"

    if (backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    CanonExample = Example(None, fileName)

    if (LOAD_LENS_FROM_FILE):
        CanonExample.LoadExample()
        canon = CanonExample.data
    else:
        canon = _CanonEF50mmf12Data()
        canon.UpdateLens()
        CanonExample.data = canon
        CanonExample.SaveExample()

    return canon


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


# ==================================================================
""" ========================== Industar ======================== """
# ==================================================================


def _Industar50_50mmf35Data():
    """
    Soviet lens.
    """
    lens = Lens()

    lens.AddSurface(Surface(17.100, 2.700, 8, "LZ_TK14"))
    lens.AddSurface(Surface(INFINITY, 4.160, 8))
    lens.AddSurface(Surface(-33.570, 1.050, 7, "LZ_LF5"))
    lens.AddSurface(Surface(14.560, 2.500, 7))

    lens.AddSurface(Stop(2.550))

    lens.AddSurface(Surface(346.700, 1.200, 7, "LZ_OF1"))
    lens.AddSurface(Surface(15.000, 4.700, 7, "LZ_TK14"))
    lens.AddSurface(Surface(-23.600, 44.560, 7))

    return lens


def Industar50_50mmf35():
    """
    Industar-50 50mm f/3.5

    :return: initlized lens object.
    """

    fileName = "Industar-50-50mmf3.5"

    if (backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    LensExample = Example(None, fileName)

    if (LOAD_LENS_FROM_FILE):
        LensExample.LoadExample()
        industar = LensExample.data
    else:
        industar = _Industar50_50mmf35Data()
        industar.UpdateLens()
        LensExample.data = industar
        LensExample.SaveExample()

    return industar


# ==================================================================
""" ==================== Zeiss Hologon 15mm f/8 ================ """
# ==================================================================


def _ZeissHologon15mmf8Data():
    """
    Data from patent JP 1988-081312, Example ML. 
    Note that the original patent contains a teleconverter at the rear, the data below has removed it.
    """
    lens = Lens()

    lens.AddSurface(Surface(11.64,	    7.9785,     11.25,     "FD60"))
    lens.AddSurface(Surface(3.843,	    2.703,      3.8))
    lens.AddSurface(Surface(5.6685,	    4.1685,     3.6,     "H-LAK7"))

    _temp =         Surface(INFINITY,	0,          0.8)
    _temp.disableBoundaryL = True
    _temp.stopOnly = True
    lens.AddSurface(_temp)

    lens.AddSurface(Stop(               0))

    _temp =         Surface(INFINITY,	3.48,       0.8,     "H-LAK7")
    _temp.stopOnly = True
    lens.AddSurface(_temp)

    _temp =         Surface(-5.508,	    2.4045,     3.5)
    _temp.disableBoundaryL = True
    lens.AddSurface(_temp)

    lens.AddSurface(Surface(-3.6285,	    5.2365,     3.6,      "FD60"))
    lens.AddSurface(Surface(-8.9835,	    4.2686,     8.7))

    # lens.AddSurface(Surface(INFINITY, 1, 23))
    # lens.AddSurface(Surface(INFINITY, 1, 23))

    return lens


def ZeissHologon15mmf8():
    """
    Zeiss Hologon 15mm f/8 
    
    :return: initlized lens object.
    """

    fileName = "ZeissHologon15mmf8"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    LensExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        LensExample.LoadExample()
        canon = LensExample.data
    else: 
        canon = _ZeissHologon15mmf8Data()
        canon.UpdateLens()
        LensExample.data = canon
        LensExample.SaveExample()

    return canon


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
""" ==================== Zeiss Sonnar 50mm f/1.5 =============== """
# ==================================================================


def _Sonnar50F15Data():
    """
    Zeiss Biotar 500mm f/1.4.
    Data from US 1786916 Example 2, EFL 100mm.

    :return: Lens object with only data and not initlized.
    """
    sonnar = Lens()

    sonnar.AddSurface(Surface(34.605,	4.665,      17,     "H-ZBAF5"))
    sonnar.AddSurface(Surface(216.92,	0.19,       17))
    sonnar.AddSurface(Surface(17.93,	5.905,      14,     "H-ZBAF5"))
    sonnar.AddSurface(Surface(42.89,	3.525,      14,     "FSL5"))
    sonnar.AddSurface(Surface(-323.155,	0.95,       14,     "SF3"))
    sonnar.AddSurface(Surface(11.755,	6.49,       9.5))

    sonnar.AddSurface(Stop(             1.13))

    sonnar.AddSurface(Surface(INFINITY,	1.24,       10,     "KF5"))
    sonnar.AddSurface(Surface(25.545,	9.905,      10,     "S-ZBAF3", True))
    sonnar.AddSurface(Surface(-11.06,	2.285,      10,   "BAL7", True))
    sonnar.AddSurface(Surface(-51.565,	22.728,     11))

    return sonnar


def Sonnar50mmF15():
    """
    Zeiss Biotar 500mm f/1.4.

    :return: initlized lens object.
    """
    fileName = "ZeissSonnar50mmf1.5"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    sonnarExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        sonnarExample.LoadExample()
        sonnar = sonnarExample.data
    else:
        sonnar = _Sonnar50F15Data()
        sonnar.UpdateLens()
        sonnarExample.data = sonnar
        sonnarExample.SaveExample()

    return sonnar


# ==================================================================
""" ===================== Zhongyi 50mm f/-/95  ================= """
# ==================================================================


def _Zhongyi50f095Data():
    zy = Lens()


    zy.AddSurface(Surface(108.29,   6.70,  29,      "H-ZLAF55D"))
    zy.AddSurface(Surface(-450.80,  0.10,  29))
    zy.AddSurface(Surface(30.77,    6.84,  23,      "H-ZLAF68B"))
    zy.AddSurface(Surface(45.50,    2.82,  20))
    zy.AddSurface(Surface(102.69,   1.20,  20.5,    "H-ZF6"))
    zy.AddSurface(Surface(24.50,    9.58,  18.5))

    zy.AddSurface(Stop(7.76))

    zy.AddSurface(Surface(-26.45,   1.37,    18.5,  "H-ZF5"))
    zy.AddSurface(Surface(314.30,   9.15,    21,    "H-ZLAF55D"))
    zy.AddSurface(Surface(-41.48,   0.10,    22))
    zy.AddSurface(Surface(101.17,   5.20,    23,    "H-ZLAF55D"))
    zy.AddSurface(Surface(-1137.00, 3.43,    23))
    zy.AddSurface(Surface(599.80,   5.82,    22.5,  "H-ZLAF55D"))
    zy.AddSurface(Surface(-82.10,   0.10,    22.5))
    zy.AddSurface(Surface(55.88,    5.58,    20.5,  "H-ZLAF55D"))
    zy.AddSurface(Surface(373.20,   1.20,    19.5,  "H-FK61"))
    zy.AddSurface(Surface(29.65,    2.99,    17))
    zy.AddSurface(Surface(62.02,    8.97,    17.5,  "H-LAF2"))
    zy.AddSurface(Surface(-35.78,   1.20,    17.5,  "H-ZF52TT"))
    zy.AddSurface(Surface(372.66,   22.93,   17.5))

    return zy


def Zhongyi50f095():

    fileName = "Zhongyi50f095"

    if(backend_name == 'cupy'):
        fileName += '_CP'
    else:
        fileName += '_NP'

    zyExample = Example(None, fileName)

    if(LOAD_LENS_FROM_FILE):
        zyExample.LoadExample()
        zy = zyExample.data
    else:
        zy = _Zhongyi50f095Data()
        zy.UpdateLens()
        zyExample.data = zy
        zyExample.SaveExample()

    return zy


# ==================================================================


def main():
    
    SetUnifScale(50)
    AddXYZ()
    RemoveBG()

    start = time.time()

    #Industar50_50mmf35() #
    # exampleLens = CanonEF50mmf12L()
    # exampleLens = Industar50_50mmf35()
    reader = LensFromZmx(RectPath(r"resources/Zmx/AdaptAll500mmf8.zmx"))
    # reader = LensFromZmx(RectPath(r"resources/Zmx/CanonFD85f2.8SoftFocus.zmx"))

    exampleLens = reader.GetLens()

    # exampleLens.AddFrontGroup([
    #     Surface(200, 2, 20, "FD60"),
    #     Surface(INFINITY, 1, 20)
    # ])
    exampleLens.UpdateLens()

    end = time.time()
    print("When setting to ", LOAD_LENS_FROM_FILE, ", program took ", end-start, " to finish.")

    print(exampleLens.GetInfo())
    print(exampleLens.SurfaceReport())
    #print("BFD ", exampleLens.BestFocusBFD(200000))

    exampleLens.DrawLens()
    # exampleLens.entrancePupil.DrawSamplePoints()
    # exampleLens.entrancePupil.DrawSurface()
    # exampleLens.frontPincipalPlane.DrawSamplePoints()
    
    plt.show()


    print("End of test")

    

if __name__ == "__main__":
    main()