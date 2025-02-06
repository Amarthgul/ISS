
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

    imageDistance = 5000

    lens = Biotar50mmf14()
    #lens.SetAperture(22)


    source = PointsSource()
    source.isCartesian = False
    source.GenerateSpots(19, 12)

    # source.SetPoints(bd.array([
    #     [0,     0, -50000, 1, 1, 1],
    #     [4.75,     3,  -50000, 1, 1, 1],
    #     [9.5,    6,  -50000, 1, 1, 1],
    #     [14.25,    9,  -50000, 1, 1, 1],
    #     [19.5,    12,  -50000, 1, 1, 1]
    #     ]))

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2D()
    sourceImage.horizontalAoV = 40
    sourceImage.imageDimensionOverride = 1920 
    sourceImage.distance = imageDistance
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png") 
    #sourceImage.SetupTransitionTest()
    # Henri-Cartier-Bresson.png ISO12233-4k.png  Arrow.png Grid.png

    start = time.time()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    im = ax.imshow(ImageConversion(image))
    iterationCount = 0

    while(True):
        #print("- Starting a new sample iteration")
        #mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(40960), 5)
        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(32), 81620)
        #print(mainRB.ToString())

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        
        im.set_data(ImageConversion(image))
        plt.draw()
        plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start
        print(iterationCount, "th iteration finished a new sample iteration after ", elpased)
        iterationCount += 1

    # lens.DrawLens()
    # imager.DrawSurface()
    # mainRP.DrawPath()

    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()
    plt.draw()
    plt.imshow(ImageConversion(image))


if __name__ == "__main__":
    main()