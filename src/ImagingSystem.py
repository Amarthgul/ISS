
from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.PltPlot import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints, DrawNormal, RemoveBG, DrawDisk
from ExampleLenses import Biotar50mmf14
from Imagers import Imager 
from ObjectSpace import Point
from Raytracing.Emission import EmitField

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
    imager = Imager(bfd=30)

    lens = Biotar50mmf14()

    source = Point()
    source.fieldX = 10
    source.RGB= bd.array([1, 1, 1])

    # Assemble the imaging system 
    imgSys = ImagingSystem() 
    imgSys.AddLens(Biotar50mmf14())
    imgSys.AddImager(imager)
    imgSys.imager.SetLensLength(imgSys.lens.totalAxialLength)

    mainRB = EmitField(
        source.fieldX, 
        source.fieldY, 
        source.distance, 
        lens.entrancePupil.GetSamplePoints(64))
    
    lens.SetIncidentRaybatch(mainRB)
    lens.Propagate()

    image = imager.IntegralRays(mainRB)


    #lens.DrawLens()
    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()
    # plt.show()


if __name__ == "__main__":
    main()