
# For unused functions and tests 

import time
import matplotlib.pyplot as plt
from PIL import Image


from Util.Backend import backend as bd
from Util.Misc import ImageConversion
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2D
from Raytracing.Emission import EmitField



def ISO12233Test(imageDistance = 200000, imageMinSample = 320):

    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    lens = Biotar50mmf14()
    #lens.SetAperture(22)

    source = PointsSource()
    source.isCartesian = False
    source.GenerateSpots(19, 12)

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

    # plt.ion()  # Turn on interactive mode
    # fig, ax = plt.subplots()
    # im = ax.imshow(ImageConversion(image))
    iterationCount = 0

    while(True):
        #print("- Starting a new sample iteration")
        #mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(40960), 5)
        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(32), 40960)

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        
        # im.set_data(ImageConversion(image))
        # plt.draw()
        # plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start
        imMin, imMax, imR = sourceImage.GetSampleRatios()

        print(iterationCount, "th iteration finished a new sample iteration after ", elpased, "  \t Min: ", imMin, " max: ", imMax,  " -Ratio: ", imR)
        iterationCount += 1
        
        if(imMin > imageMinSample):
            imgSave = Image.fromarray(ImageConversion(image), 'RGB')
            imgSave.save(r"resources/Results/Biotar_dist"+str(imageDistance)+"_" + str(imageMinSample) + "Sample.png")
            break

    return elpased 

# =====================================================================
""" =============================================================== """


def main():

    distList = {
        400     :   20, 
        450     :   20,
        500     :   20, 
        550     :   20, 
        600     :   21, 
        700     :   22, 
        800     :   23, 
        950     :   24, 
        1200    :   25, 
        1350    :   26 , 
        1500    :   27, 
        1750    :   28, 
        2000    :   29, 
        2500    :   30, 
        3000    :   33, 
        4000    :   36, 
        5000    :   40, 
        7500    :   50, 
        10000   :   60, 
        20000   :   70, 
        50000   :   80, 
        100000  :   90, 
        200000  :   320
    }

    timeConsumed = []

    for key, value in distList.items():
        timeConsumed.append(ISO12233Test(key, value))

    print(timeConsumed)


if __name__ == "__main__":
    main() 