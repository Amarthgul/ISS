
from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import ImageConversion
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2D
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
        pass


    # ==================================================================
    """ ============================================================ """
    # ==================================================================



    
def main():

    lens = Biotar50mmf14()
    lens.SetAperture(8)

    # FD50 seems to be around 32.6

    source = PointsSource()
    source.isCartesian = False
    #source.GenerateSpots(12, 19.25)
    source.SetPoints(bd.array([
        [0,     0, -50000, 1, 1, 1],
        [4.75,     3,  -50000, 1, 1, 1],
        [9.5,    6,  -50000, 1, 1, 1],
        [14.25,    9,  -50000, 1, 1, 1],
        [19.5,    12,  -50000, 1, 1, 1]
        ]))

    imager = StdImager(bfd=32.4)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2D()
    #sourceImage.distance = 1500
    sourceImage.horizontalAoV = 40
    sourceImage.imageDimensionOverride = 1280 
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png") 
    # Henri-Cartier-Bresson.png ISO12233-4k.png  Arrow.png

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    im = ax.imshow(ImageConversion(image))
    while(True):
        #print("- Starting a new sample iteration")
        #mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(10000), 5)
        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(128), 20480)
        #print(mainRB.ToString())

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        
        im.set_data(ImageConversion(image))
        plt.draw()
        plt.pause(0.01)
        #print("  Finished a new sample iteration")
        #print(source.sampleRecord)

    # lens.DrawLens()
    # imager.DrawSurface()
    # mainRP.DrawPath()

    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()
    # plt.show()


if __name__ == "__main__":
    main()