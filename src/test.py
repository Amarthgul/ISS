

from PIL import Image
import time
import matplotlib.pyplot as plt
import OpenEXR

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, ImageConversionAverage, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG
from Util.Sampling import CircularDistribution
from Util.Misc import ProgressBar, AngleFieldToCartesian, SoundAlarm, RectPath
from Util.Globals import PRECISION_TYPE
from Util.MaterialLookup import FindClosestMaterials, ReadSheet
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18, ZeissHologon15mmf8, Mug
from Imagers.Standard import StdImager
from Imagers.PDA import PDA
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2DFlat
from ObjectSpace.ImageVariDepth import Image2DVariDepth
from Raytracing.Emission import EmitField, EmitFieldMultispectral
from Raytracing.Raypath import RayPath


FrameCount = 0 


def ISO12233Test(lens, AoV=40, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):
    
    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4 lens.BestFocusBFD(imageDistance)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    sourceImage = Image2DFlat()
    sourceImage.horizontalAoV = AoV 
    sourceImage.imageDimensionOverride = 1920 
    sourceImage.distance = imageDistance
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png") 
    # Henri-Cartier-Bresson.png ISO12233-4k.png  CustomSheet.png Grid.png

    start = time.time()

    iterationCount = 0
    perIterRays = 20480 #40960 

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    

    while(True):


        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), perIterRays)
        # For image simulation, pupil sample still needs to be very high to avoid pattern from showing up
        # mainRB = sourceImage.EmitSamplesToward(lens.GetFirstElementSamples(1024), perIterRays)

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)

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
        ProgressBar(iterationCount / imageMinSample, 100)

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

        mainRB, mainRP, _ = lens.Propagate(mainRB)

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


def RayPathTesting(lens, AoV, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):

    lens.DrawLens()
    SetUnifScale(50)
    AddXYZ()
    RemoveBG()

    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    mainRB = EmitField(AoV/2, 0, imageMinSample, sampleTargets=lens.entrancePupil.GetSamplePoints(32))


    mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, recordPath=True)

    # mainRB, _tir, _vig = imager.IntersectRays(mainRB)

    # mainRP.Append(mainRB, _tir, _vig)

    image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)

    mainRP.DrawPath(expendEnd=40)
    #lens.entrancePupil.DrawSurface()

    plt.show()


    elpased = time.time() - start

    return elpased


def PDATest(lens, tUVIR = 1, AoV=40, imageDistance=200000, imageMinSample=320, realTimeUpdate=False):
    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = PDA()
    if(tUVIR > 0):
        imager.tUVIR = tUVIR
        lens.AddRearGroup(imager.GetUVIR())
        lens.UpdateLens()

    # Assemble the imaging system
    imager.SetLensLength(lens.totalAxialLength)
    imager.BFD = lens.BestFocusBFD(imageDistance)
    imager.Update()
    print("Best focus: ", imager.BFD)
    image = imager.AccquireEmpty()

    # lens.DrawLens()
    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()
    # imager.DrawSurface()
    # plt.draw()
    # plt.pause(5)

    sourceImage = Image2DFlat()
    sourceImage.horizontalAoV = AoV
    sourceImage.imageDimensionOverride = 1920
    sourceImage.distance = imageDistance
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png")
    # Henri-Cartier-Bresson.png ISO12233-4k.png  CustomSheet.png Grid.png

    start = time.time()

    iterationCount = 0
    perIterRays = 20480  # 40960

    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    while (True):

        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), perIterRays)
        # For image simulation, pupil sample still needs to be very high to avoid pattern from showing up
        # mainRB = sourceImage.EmitSamplesToward(lens.GetFirstElementSamples(1024), perIterRays)

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)

        if (realTimeUpdate):
            im.set_data(ImageConversion(image))
            plt.draw()
            plt.pause(0.01)

        # print(source.sampleRecord)
        elpased = time.time() - start
        imMin, imMax, imR = sourceImage.GetSampleRatios()

        # print("End RB size: ", mainRB.value.shape)
        print(iterationCount, "th iteration finished a new sample iteration after ", elpased, "  \t Min: ", imMin,
              " max: ", imMax, " -Ratio: ", imR)
        ProgressBar(iterationCount / imageMinSample, 100)

        iterationCount += 1

        if (iterationCount > imageMinSample):
            image /= 100
            global FrameCount
            fn = r"UVIR_Test" + str(imageDistance) + "_" + str(tUVIR)
            SaveAsEXR(image, r"resources/Results/", fn)
            break

    FrameCount += 1

    return elpased


def StereoImageTest():
    targets = bd.array([
        [1, 2, 25],
        [2, 4, 25],
        [-2, 3, 25],
        [1, -2, 25]
    ])

    img = Image2DVariDepth()
    img.imageDimensionOverride = 300
    img.zDepthMappingRange = bd.array([500, 1000])

    img.LoadFrom8bit(r"resources/DualTest_RGB.png", r"resources/DualTest_Z.png")

    img.DrawImage()

    RemoveBG()
    SetUnifScale(1000)
    plt.show()


def MaterialLookUpTest():
    # Example usage:
    excel_file = ReadSheet()

    # Suppose you have n=1.5168 and V=64.1 for the 'd' line
    line = 'e'
    stats = [
        [1.72341,    50.1],
        [1.7899,    48],
        [1.70444,    29.84],
        [1.7899 ,   48],
        [1.76167,    27.34],
        [1.7899  ,  48],
        [1.72056 ,   47.59]
    ]

    for item in stats:
        n_val = item[0]
        v_val = item[1]

        result_df = FindClosestMaterials(excel_file, line, n_val, v_val, top_k=3)
        print("Closest matches:")
        print(result_df)

# ==================================================================
""" ======================== End of Defs ======================= """
# ==================================================================
def main():

    objectDistance = [
        350, 500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 15000, 30000, 100000
    ]
    focusDistance = [
        350, 500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 15000, 30000, 100000
    ]

    MaterialLookUpTest()

    objectDistance = [
        450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000
    ]
    # objectDistance = [ 
    #     850, 875, 900, 925, 950, 975, 1000, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 
    #     5000, 6000, 7000, 8000, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000
    # ]
    
    angleFieldX = bd.linspace(-20, 20, len(objectDistance))
    angleFieldY = bd.linspace(-13, 13, len(objectDistance))

    lens = Biotar50mmf14()
    lens = ZeissHologon15mmf8()
    # lens.SetAperture(4)
    #RayPathTesting(lens, AoV=40)
    # ISO12233Test(lens, AoV=101, imageDistance=100000, imageMinSample=32, realTimeUpdate=False) #4096: 10 hours

    # for t in [0, 0.15, 0.3, 0.45, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.]:
    #     PDATest(lens, t, AoV=104, imageDistance=100000, imageMinSample=512, realTimeUpdate=False)

    # for ax, ay, d in zip(angleFieldX, angleFieldY, objectDistance):
    #     ISO12233Test(lens, imageDistance=d, imageMinSample=512, realTimeUpdate=False)

        #position = bd.array([1000, 600, -o])
        #position = AngleFieldToCartesian(ax, ay, -d)
        #print("Current origin position: ", position)
        #ReflectionSpotPositionOrig(lens, position, focusDistance=1500, imageMinSample=4096, realTimeUpdate=False)
        #ReflectionSpotTesting(lens, position, focusDistance=1500, imageMinSample=2048, realTimeUpdate=False)
    
    #SpotTesting(lens, realTimeUpdate=False)

    #position = bd.array([angleFieldX[0], angleFieldY[0], -bd.array(20000)]) 
    #ReflectionSpotTesting(lens, position, focusDistance=1500, imageMinSample=2048, realTimeUpdate=False)
    #MugReflectionSpotTesting(position, Mug(), sampleSize=4096, saveIterationCount=10240, realTimeUpdate=False)

    #ReflectionTesting(Mug())

    # testRP.DrawPath()

    # lens.DrawLens()
    # DrawPoints(lens.GetFirstElementSamples())
    # print(lens.GetFirstElementSamples())

    # SetUnifScale(50)
    # AddXYZ()
    # RemoveBG()
    # plt.show()

    SoundAlarm()
    


if __name__ == "__main__":
    main()