


from Surfaces import Surface, Stop
from Lens import Lens 
from Util.DataReadWrite import Save, Load




class Example():
    def __init__(self, input, fileName):
        self.data = input 
        self.fileName = fileName

    def Save(self):
        Save(self.data, self.fileName)

    def Load(self):
        self.data = Load(self.fileName)


# TODO: make the lenses into example instances 


def Biotar50mmf14():
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

    biotar.entrancePupilDia = 35.714

    return biotar 


def Helios58mmf2():
    """
    Helios 44, 58mm f/2. 
    Data from Bill Claff.

    :return: Lens object with only data and not initlized.
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

    helios.entrancePupilDia = 29.5

    return helios



