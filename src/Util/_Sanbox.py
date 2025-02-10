
# For unused functions and tests 

import time
import matplotlib.pyplot as plt
from PIL import Image


from Util.Backend import backend as bd
from Util.ColorWavelength import ImageConversion
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
        400     :   40,   # 2241.4725670814514
        450     :   40,   # 2212.005831003189
        500     :   40,   # 2203.070353746414
        550     :   40,   # 2200.6272621154785
        600     :   41,   # 2224.8741416931152
        700     :   42,   # 2265.45525097847
        800     :   43,   # 2348.831357240677
        950     :   44,   # 2382.217203617096
        1200    :   45,   # 2403.0952944755554
        1350    :   46 ,  # 2450.764587163925
        1500    :   47,   # 2532.759054660797
        1750    :   48,   # 2566.923010826111
        2000    :   49,   # 2597.173784971237
        2500    :   50,   # 2637.548036813736
        3000    :   53,   # 2789.4950058460236
        4000    :   56,   # 2970.900398015976
        5000    :   60,   # 3127.5452451705933
        7500    :   70,   # 3614.9423286914825
        10000   :   80,   # 4095.878294467926
        20000   :   90,   # 4481.288671731949
        50000   :   100,  # 4329.990266561508
        100000  :   110,  # 3498.272744655609
        200000  :   320   # 5435.468377590179
    }
    distList = {
        100000  :   220,  # 3498.272744655609
        200000  :   420   # 5435.468377590179
    }

    timeConsumed = []

    for key, value in distList.items():
        timeConsumed.append(ISO12233Test(key, value))

    print(timeConsumed)


if __name__ == "__main__":
    main() 