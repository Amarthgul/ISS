
from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.PltPlot import Setup3Dplot, AddXYZ, SetUnifScale, DrawRaybatch, DrawSpherical, DrawPoints, DrawNormal, RemoveBG, DrawDisk
from ExampleLenses import Biotar50mmf14
from Imagers.Standard import StdImager 
from ObjectSpace.ObjectSpace import Point
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

    lens = Biotar50mmf14()

     # Set up the imager 32.3552 (34.25 for 1500 distance)
    imager = StdImager(bfd=10)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)

    source = Point()
    source.fieldX = 10
    source.distance = bd.array(50)
    source.RGB= bd.array([1, 1, 1])


    mainRB = EmitField(
        source.fieldX, 
        source.fieldY, 
        source.distance, 
        lens.entrancePupil.GetSamplePoints(50000))
    
    lens.SetIncidentRaybatch(mainRB)
    mainRB, mainRP = lens.Propagate()

    mainRB, _tir, _vig = imager.IntersectRays(mainRB)
    mainRP.Append(mainRB, _tir, _vig)

    imager.IntegralRays(mainRB)

    lens.DrawLens()
    imager.DrawSurface()
    mainRP.DrawPath()

    SetUnifScale(50)
    AddXYZ()
    RemoveBG()
    plt.show()


if __name__ == "__main__":
    main()