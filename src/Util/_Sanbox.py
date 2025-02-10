
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
        400     :   40, 
        450     :   40,
        500     :   40, 
        550     :   40, 
        600     :   41, 
        700     :   42, 
        800     :   43, 
        950     :   44, 
        1200    :   45, 
        1350    :   46 , 
        1500    :   47, 
        1750    :   48, 
        2000    :   49, 
        2500    :   50, 
        3000    :   53, 
        4000    :   56, 
        5000    :   60, 
        7500    :   70, 
        10000   :   80, 
        20000   :   90, 
        50000   :   100, 
        100000  :   110, 
        200000  :   320
    }

    timeConsumed = []

    for key, value in distList.items():
        timeConsumed.append(ISO12233Test(key, value))

    print(timeConsumed)


if __name__ == "__main__":
    main() 