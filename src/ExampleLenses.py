


from Surfaces import Surface, Stop
from Lens import Lens 


def Biotar50mm14():
    """
    Zeiss Biotar 500mm f/1.4. 
    Data from US 1786916 Example 2, EFL 100mm. 
    """
    biotar = Lens() 

    biotar.AddSurface(Surface(41.8,     5.375,  17, "BAF9"))
    biotar.AddSurface(Surface(160.5,    0.825,  17))
    biotar.AddSurface(Surface(22.4,	    7.775,  16, "SK10"))
    biotar.AddSurface(Surface(-575,	    2.525,  16, "LZ_LF5"))
    biotar.AddSurface(Surface(14.15,	9.45,   11))
    biotar.AddSurface(Surface(-19.25,	2.525,  11, "SF5"))
    biotar.AddSurface(Surface(25.25,	10.61,  13, "BAF9"))
    biotar.AddSurface(Surface(-26.6,	0.485,  13))
    biotar.AddSurface(Surface(53, 	    6.95,   14, "BAF9"))
    biotar.AddSurface(Surface(-60,	    32.3552, 14))

    #biotar.UpdateLens() 

    return biotar 


Biotar50mm14() 

