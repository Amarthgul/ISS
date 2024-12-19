
from PIL import Image
from joblib import Parallel, delayed

from Lens import * 
from python.Imagers.Imager import * 
from python.Util.Misc import * 
from ObjectSpace import * 


from Surfaces import Surface


# Random generator from the Util module
# This uses the same seed so that the result is deterministic  
RNG = rng 


class ImagingSystem:
    def __init__(self):

        self.lens = None 
        self.imager = None 
        self.rayBatch = None 

        self.rayPath = None 

        # Color-wavelength conversion 
        self.primaries = {"R": "C'", "G": "e", "B":"g"}
        self.secondaries = []#["F", "D"]
        self.UVIRcut = ["i", "A'"]

        self.point = None 
        self.inputImage = None 

    
    def AddLens(self, lens):
        self.lens = lens 


    def AddImager(self, imager):
        self.imager = imager 


    def Test(self, image, perPointSample):
        mat = []  

        start = time.time()
        sX = image.sampleX   # Horizontal sample count
        sY = image.sampleY   # Vertical sample count 


    # ==================================================================
    """ ============================================================ """
    # ==================================================================



    
def main():
    biotar = Lens() 

    # Set up the lens 
    # Zeiss Biotar 50 1.4 
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
    # Update immediately after all the surfaces are defined  
    biotar.UpdateLens() 

    # Set up the imager 32.3552 (34.25 for 1500 distance)
    imager = Imager(bfd=35)

    # Assemble the imaging system 
    imgSys = ImagingSystem() 
    imgSys.AddLens(biotar)
    imgSys.AddImager(imager)
    imgSys.imager.SetLensLength(imgSys.lens.totalLength)


    



if __name__ == "__main__":
    main()