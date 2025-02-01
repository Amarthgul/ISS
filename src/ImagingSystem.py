
from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import ImageConversion
from ExampleLenses import Biotar50mmf14
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
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
    imager = StdImager(bfd=37)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    source = PointsSource(bd.array([
        [0,     0, -20000, 1, 1, 1],
        [4.75,     3,  -20000, 1, 1, 1],
        [9.5,    6,  -20000, 1, 1, 1],
        [14.25,    9,  -20000, 1, 1, 1],
        [19.5,    12,  -20000, 1, 1, 1]
        ]))

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    im = ax.imshow(ImageConversion(image))
    while(True):
        print("- Starting a new sample iteration")
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(10000), 3)
        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        
        im.set_data(ImageConversion(image))
        plt.draw()
        plt.pause(0.01)
        print("  Finished a new sample iteration")
        print(source.sampleRecord)

    # lens.DrawLens()
    # imager.DrawSurface()
    # mainRP.DrawPath()

    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()
    # plt.show()


if __name__ == "__main__":
    main()