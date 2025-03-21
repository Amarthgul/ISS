

from PIL import Image
import time
import matplotlib.pyplot as plt
import OpenEXR

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, ImageConversionAverage, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG
from Util.Sampling import CircularDistribution
from Util.Misc import ProgressBar
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18, ZeissHologon15mmf8, Mug
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2D
from Raytracing.Emission import EmitField, EmitFieldMultispectral
from Raytracing.Raypath import RayPath


FrameCount = 0 


def ISO12233Test(lens, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):
    
    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2D()
    sourceImage.horizontalAoV = 40 
    sourceImage.imageDimensionOverride = 1920 
    sourceImage.distance = imageDistance
    sourceImage.LoadFrom8bit(r"resources/CustomSheet.png") 

    #sourceImage.SetupTransitionTest()
    # Henri-Cartier-Bresson.png ISO12233-4k.png  CustomSheet.png Grid.png

    start = time.time()

    iterationCount = 0
    perIterRays = 204800 #40960 

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    

    while(True):

        #mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(40960), 5)
        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(128), perIterRays)

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP, reflectedRB = lens.Propagate(reflection=False)

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)

        if(realTimeUpdate):
            im.set_data(ImageConversion(image))
            plt.draw()
            plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start
        imMin, imMax, imR = sourceImage.GetSampleRatios()

        print(iterationCount, "th iteration finished a new sample iteration after ", elpased, "  \t Min: ", imMin, " max: ", imMax,  " -Ratio: ", imR)
        ProgressBar(iterationCount / imageMinSample)

        iterationCount += 1
        
        if(iterationCount > imageMinSample):
            image /= 100 
            global FrameCount
            fn = r"ISO12233Test"+str(imageDistance)+"_"+str(FrameCount)
            SaveAsEXR(image, r"resources/Results/ISO12233", fn)
            break

    FrameCount += 1

    return elpased 


def SpotTesting(objectDistance = 20000, focusDistance = 500, saveIterationCount = 50, realTimeUpdate = False):

    lens = Biotar50mmf14()

    source = PointsSource()
    source.isCartesian = False
    source.GenerateSpots(19, 12, dist=objectDistance)

    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=1920) #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))
        iterationCount = 0

    while(True):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(20480), 5, addSecondary=9)

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP, _ = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        if(realTimeUpdate):
            im.set_data(ImageConversion(image))
            plt.draw()
            plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start

        print("- Focusing ", focusDistance, " for obj at ", objectDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased) )

        if(iterationCount > saveIterationCount):
            fn = r"SpotTest"+str(objectDistance)+"_"+str(FrameCount)
            SaveAsEXR(image, r"resources/Results", fn)
            break

        iterationCount += 1


def ReflectionSpotTesting(lens, sampleSize=512, objectDistance = 20000, focusDistance = 5000, saveIterationCount = 100, realTimeUpdate = True):


    source = PointsSource()
    source.isCartesian = False
    #source.GenerateSpots(19, 12, dist=objectDistance)
    source.GenerateSpots(19, 12, dist=objectDistance)

    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=1920) 
    #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    iterationCount = 0

    while(True):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(sampleSize), 5, addSecondary=5)

        lens.SetIncidentRaybatch(mainRB)

        _mainRB, mainRP, mainRB = lens.Propagate(reflection=True)
        # mainRB.Merge(_mainRB)
        print("highest r: ", bd.max(mainRB.PolarizedRadiance()), "\t average: ", bd.mean(mainRB.PolarizedRadiance()))
        

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image, overExpNoiseRemoval=12)

        if(realTimeUpdate):
            print("Max ", bd.max(image))
            im.set_data(ImageConversion(image, maxModifier=0.5)) #0.002
            plt.draw()
            plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start

        print("\n- Focusing ", focusDistance, " for obj at ", objectDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased))

        iterationCount += 1
        if(iterationCount > saveIterationCount):
            image /= 100 
            global FrameCount
            fn = r"flareTest"+str(FrameCount)
            SaveAsEXR(image, r"resources/Results/SpotTestng", fn)
            break

    FrameCount += 1
     


def MugReflectionSpotTesting(lens=Mug(), sampleSize=512, objectDistance = 20000, focusDistance = 5000, saveIterationCount = 100, realTimeUpdate = True):

    imager = StdImager(2, w = 45, h  =45, horiPx=1920) 
    #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    iterationCount = 0

    while(True):
        mainRB = EmitField(30, 30, distance=1500, sampleTargets = lens.entrancePupil.GetSamplePoints(sampleSize))

        lens.SetIncidentRaybatch(mainRB)

        _mainRB, mainRP, mainRB = lens.Propagate(reflection=True)
        # mainRB.Merge(_mainRB)
        print("highest r: ", bd.max(mainRB.PolarizedRadiance()), "\t average: ", bd.mean(mainRB.PolarizedRadiance()))
        

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image, overExpNoiseRemoval=None)

        if(realTimeUpdate):
            print("Max ", bd.max(image))
            im.set_data(ImageConversion(image, maxModifier=1)) #0.002
            plt.draw()
            plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start

        print("\n- Focusing ", focusDistance, " for obj at ", objectDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased))
        #print(" \t immax ", bd.max(image), "  \t\t imMin", bd.min(image), "\t\t ImAve ", bd.mean(image), "\t\t median ", bd.median(image))

        if(iterationCount > saveIterationCount):
            SaveAsEXR(image, r"resources/Results/SpotTestng", "exrTest")
            break

        iterationCount += 1


def ReflectionTesting(lens):

    lens.DrawLens() # ======= Draw call
    SetUnifScale(50)
    AddXYZ()
    RemoveBG()


    targetR = 15
    mainRB = EmitField(10, 0, 
                       distance=50, 
                       sampleTargets=CircularDistribution(zDepth=3) * bd.array([targetR, targetR, 1]))
    
    mainRB = EmitFieldMultispectral(60, 0, sampleTargets = lens.entrancePupil.     GetSamplePoints(16))

    lens.SetIncidentRaybatch(mainRB)
    mainRB, mainRP, reflectedRB = lens.Propagate(recordPath=True, reflection=True)

    #print(bd.max(reflectedRB.PolarizedRadiance()))

    #print(mainRB.PolarizedRadiance())

    #DrawRaybatch(reflectedRB) # ======= Draw call
    # mainRP.DrawPath() # ======= Draw call

    
    
    plt.show()
    

def RayPathTesting(lens, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):
    
    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

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

    iterationCount = 0
    normalizer = iterationCount + 10
    perIterRays = 4

    normalizer = iterationCount + 10

    #mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(40960), 5)
    #DrawPoints(lens.entrancePupil.GetSamplePoints(10))
    
    mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(64), perIterRays)
    print("Inititally generated ", mainRB.value.shape)

    lens.SetIncidentRaybatch(mainRB)

    mainRB, mainRP, reflectedRB = lens.Propagate(recordPath=True)

    mainRB, _tir, _vig = imager.IntersectRays(mainRB)
    # mainRP.Append(mainRB, _tir, _vig)

    image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)

    #mainRP.DrawPath(expendEnd=40)
    #lens.entrancePupil.DrawSurface()
    lens.DrawLens()
    SetUnifScale(50)
    AddXYZ()
    RemoveBG()
    plt.show()


    elpased = time.time() - start
    imMin, imMax, imR = sourceImage.GetSampleRatios()

    return elpased 


def main():

    objectDistance = [
        350, 500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 15000, 30000, 100000
    ]
    focusDistance = [
        350, 500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 15000, 30000, 100000
    ]

    objectDistance = [
        450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000
    ]
    # objectDistance = [
    #     450, 1500, 7500, 30000, 100000
    # ]

    lens = Biotar50mmf14()
    # lens.SetAperture(4)
    ISO12233Test(lens, imageDistance=100000, imageMinSample=64, realTimeUpdate=False) #4096: 10 hours 

    # for o in objectDistance:
        # ReflectionSpotTesting(CanonFD50mmf18(), sampleSize=256, saveIterationCount=512, realTimeUpdate=False, objectDistance=o)
        # ISO12233Test(lens, imageDistance=o, imageMinSample=256, realTimeUpdate=False)
    
    #SpotTesting(realTimeUpdate=True)

    # MugReflectionSpotTesting(Mug(), sampleSize=40960, saveIterationCount=32, realTimeUpdate=True)
    # ReflectionSpotTesting(CanonFD50mmf18(), sampleSize=256, saveIterationCount=128, realTimeUpdate=True)

    #ReflectionTesting(Mug())

    # testRP.DrawPath()

    # SetUnifScale(50)
    # #AddXYZ()
    # RemoveBG()
    # plt.show()
    
    


if __name__ == "__main__":
    main()