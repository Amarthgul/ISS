

from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.ColorWavelength import ImageConversion
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2D
from Raytracing.Emission import EmitField



def SpotTesting(saveIterationCount = 50):

    imageDistance = 10000


    lens = Biotar50mmf14()

    source = PointsSource()
    source.isCartesian = False
    source.GenerateSpots(19, 12)

    imager = StdImager(lens.BestFocusBFD(imageDistance), horiPx=6000) #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    im = ax.imshow(ImageConversion(image))
    iterationCount = 0

    while(True):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(20480), 5, addSecondary=15)

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        
        im.set_data(ImageConversion(image, amplifier=0.25))
        plt.draw()
        plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start

        print("At ", str(iterationCount), "th iteration after ", str(elpased) )

        if(iterationCount > saveIterationCount):
            imgSave = Image.fromarray(ImageConversion(image), 'RGB')
            imgSave.save(r"resources/Results/NoNormalize_dist"+str(imageDistance)+"_spotTestL"+str(elpased)+".png")
            break

        iterationCount += 1

    plt.draw()
    plt.imshow(ImageConversion(image))


def main():

    SpotTesting()
    
    


if __name__ == "__main__":
    main()