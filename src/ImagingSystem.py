
from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import ImageConversion
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints
from ExampleLenses import Biotar50mmf14
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
    #lens.SetAperture(2.8)

    # Set up the imager 32.3552 (34.25 for 1500 distance)
    imager = StdImager(bfd=32.35)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2D()
    #sourceImage.distance = 1500
    sourceImage.imageDimensionOverride = 1280 
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png") 
    # Henri-Cartier-Bresson.png ISO12233-4k.png  Arrow.png

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    im = ax.imshow(ImageConversion(image))
    while(True):
        #print("- Starting a new sample iteration")
        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(128), 4069)

        #bd.savetxt("tempSave.csv", mainRB.value, delimiter=",")

        # AddXYZ()
        # SetUnifScale(100)
        # DrawRaybatch(mainRB, length=150) # Draw call ==============
        # plt.show()
        # plt.pause(20)

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