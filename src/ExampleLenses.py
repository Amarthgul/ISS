


from Surfaces import Surface, Stop
from Lens import Lens 


def Biotar50mm14():
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


Biotar50mm14() 

