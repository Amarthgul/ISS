

from PIL import Image
import time
import matplotlib.pyplot as plt
import OpenEXR

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, ImageConversionAverage, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG
from Util.Sampling import CircularDistribution
from Util.Misc import ProgressBar, AngleFieldToCartesian
from Util.Globals import PRECISION_TYPE
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18, ZeissHologon15mmf8, Mug
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2DFlat
from Raytracing.Emission import EmitField, EmitFieldMultispectral
from Raytracing.Raypath import RayPath


FrameCount = 0 


def ISO12233Test(lens, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):
    
    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4 lens.BestFocusBFD(imageDistance)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2DFlat()
    sourceImage.horizontalAoV = 40 
    sourceImage.imageDimensionOverride = 1920 
    sourceImage.distance = 750 # imageDistance
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png") 

    #sourceImage.SetupTransitionTest()
    # Henri-Cartier-Bresson.png ISO12233-4k.png  CustomSheet.png Grid.png

    start = time.time()

    iterationCount = 0
    perIterRays = 20480 #40960 

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    

    while(True):

        #mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(40960), 5)
        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), perIterRays) 
        # For image simulation, pupil sample still needs to be very high to avoid pattern from showing up 

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

        #print("End RB size: ", mainRB.value.shape)
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


def SpotTesting(lens, objectDistance = 100000, focusDistance = 750, saveIterationCount = 32, realTimeUpdate = False):

    source = PointsSource()
    source.isCartesian = False
    source.GenerateSpots(19, 12, dist=objectDistance)
    

    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=1920) #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    iterationCount = 0
    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))
        

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
        ProgressBar(iterationCount / saveIterationCount)

        if(iterationCount > saveIterationCount):
            fn = r"SpotTest"+str(objectDistance)+"_"+str(FrameCount)
            SaveAsEXR(image, r"resources/Results", fn)
            break

        iterationCount += 1


def ReflectionSpotTesting(lens, position, focusDistance = 5000, imageMinSample = 100, realTimeUpdate = True):


    source = PointsSource()
    source.GenerateFixPoint(position)

    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=1920) 
    #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty(dataType=PRECISION_TYPE) 

    start = time.time()

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    iterationCount = 0

    while(iterationCount < imageMinSample):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(2048), 5, addSecondary=5)

        _mainRB, mainRP, mainRB = lens.Propagate(mainRB, reflection=True)
  
        if(mainRB.value.shape[0] == 0):
            continue

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

        print("\n- Focusing ", focusDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased))
        ProgressBar(iterationCount / imageMinSample, 100)

        iterationCount += 1

    image /= 10.0
    global FrameCount
    fn = r"SingleFlareTestTransverse"+str(FrameCount)
    SaveAsEXR(image, r"resources/Results/SpotTestng", fn)

    FrameCount += 1
   

def ReflectionSpotPositionOrig(lens, position, focusDistance = 5000, imageMinSample = 100, realTimeUpdate = True):
    """
    Actual position of the spot. 
    """

    source = PointsSource()
    source.GenerateFixPoint(position)

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
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(40960), 5, addSecondary=5)

        mainRB, mainRP, _mainRB = lens.Propagate(mainRB, reflection=False)
        # mainRB.Merge(_mainRB)
        #print("highest r: ", bd.max(mainRB.PolarizedRadiance()), "\t average: ", bd.mean(mainRB.PolarizedRadiance()))
        

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

        print("\n- Focusing ", focusDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased))
        ProgressBar(iterationCount / imageMinSample, 100)

        iterationCount += 1
        if(iterationCount > imageMinSample):
            image /= 100 
            global FrameCount
            fn = r"SingleFlareTestOriginalSpot"+str(FrameCount)
            SaveAsEXR(image, r"resources/Results/SpotTestng", fn)
            break

    FrameCount += 1
   


def MugReflectionSpotTesting(position, lens=Mug(), sampleSize=512, saveIterationCount = 100, realTimeUpdate = True):

    imager = StdImager(2, w = 45, h  =45, horiPx=1920) 
    #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    source = PointsSource()
    source.GenerateFixPoint(position)

    start = time.time()

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    iterationCount = 0

    while(True):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(sampleSize), 5, addSecondary=5)
        


        _mainRB, mainRP, mainRB = lens.Propagate(mainRB, reflection=True)
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

        print("\nAt ", str(iterationCount), "th iteration after ", str(elpased))
        ProgressBar(iterationCount / saveIterationCount, 100)

        if(iterationCount > saveIterationCount):
            image /= 10.
            SaveAsEXR(image, r"resources/Results/mugShot", "exrTest")
            break

        iterationCount += 1


def RayPathTesting(lens, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):
    
    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2DFlat()
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
    objectDistance = [ 
        850, 875, 900, 925, 950, 975, 1000, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 
        5000, 6000, 7000, 8000, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000
    ]
    
    angleFieldX = bd.linspace(-20, 20, len(objectDistance))
    angleFieldY = bd.linspace(-13, 13, len(objectDistance))

    lens = Biotar50mmf14()
    # lens.SetAperture(4)
    #ISO12233Test(lens, imageDistance=100000, imageMinSample=32, realTimeUpdate=False) #4096: 10 hours 

    #for ax, ay, d in zip(angleFieldX, angleFieldY, objectDistance):
        #ISO12233Test(lens, imageDistance=o, imageMinSample=512, realTimeUpdate=False)

        #position = bd.array([1000, 600, -o])
        #position = AngleFieldToCartesian(ax, ay, -d)
        #print("Current origin position: ", position)
        #ReflectionSpotPositionOrig(lens, position, focusDistance=1500, imageMinSample=4096, realTimeUpdate=False)
        #ReflectionSpotTesting(lens, position, focusDistance=1500, imageMinSample=2048, realTimeUpdate=False)
    
    #SpotTesting(lens, realTimeUpdate=False)

    position = bd.array([angleFieldX[0], angleFieldY[0], -bd.array(20000)]) 
    ReflectionSpotTesting(lens, position, focusDistance=1500, imageMinSample=2048, realTimeUpdate=False)
    #MugReflectionSpotTesting(position, Mug(), sampleSize=4096, saveIterationCount=10240, realTimeUpdate=False)

    #ReflectionTesting(Mug())

    # testRP.DrawPath()

    # SetUnifScale(50)
    # #AddXYZ()
    # RemoveBG()
    # plt.show()
    
    


if __name__ == "__main__":
    main()