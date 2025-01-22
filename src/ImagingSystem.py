
from PIL import Image
import time


from Util.Backend import backend as bd
from Lens import Lens 
from ExampleLenses import Biotar50mmf14
from Imagers import Imager 
from Util.Globals import RNG
from Surfaces import Surface


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

    # Set up the imager 32.3552 (34.25 for 1500 distance)
    imager = Imager(bfd=35)

    # Assemble the imaging system 
    imgSys = ImagingSystem() 
    imgSys.AddLens(Biotar50mmf14())
    imgSys.AddImager(imager)
    imgSys.imager.SetLensLength(imgSys.lens.totalLength)


    



if __name__ == "__main__":
    main()