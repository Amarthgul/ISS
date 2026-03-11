

from PIL import Image
import time
import matplotlib.pyplot as plt
import OpenEXR

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, ImageConversionAverage, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG
from ZmxReader import LensFromZmx
from Util.Sampling import CircularDistribution
from Util.Misc import ProgressBar, AngleFieldToCartesian, SoundAlarm, RectPath
from Util.Globals import PRECISION_TYPE, INFINITY
from Util.MaterialLookup import FindClosestMaterials, ReadSheet
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18, ZeissHologon15mmf8, Mug, Sonnar50mmF15, CanonEF50mmf12L, Zhongyi50f095, Industar50_50mmf35
from Imagers.Standard import StdImager
from Imagers.PDA import PDA
from Surfaces.Surface import Surface
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2DFlat
from ObjectSpace.ImageVariDepth import Image2DVariDepth
from Raytracing.Emission import EmitField, EmitFieldMultispectral
from Raytracing.Raypath import RayPath


FrameCount = 0 

# This is used to reduce the amount of samples when running on local machines that does not have too much power to dispose
sampleMultiplier = 0.1

def ISO12233Test(lens, imageDistance = 200000, computeTime = 4096, realTimeUpdate = False):
    
    print("New test w/ im Distance ", imageDistance)

    # AddRearGroup AddFrontGroup
    # lens.AddRearGroup([
    #     Surface(200, 2, 15, "FK5"),
    #     Surface(INFINITY, 1, 15)
    # ])
    # lens.UpdateLens()
    print(lens.GetInfo())
    print("Best focus",  lens.BestFocusBFD(imageDistance)) #32.926564?

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4 lens.BestFocusBFD(imageDistance)
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    sourceImage = Image2DFlat()
    sourceImage.distance = imageDistance
    sourceImage.horizontalAoV = lens.GetAoV()[0]*2
    sourceImage.imageDimensionOverride = 1920
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png")
    # Henri-Cartier-Bresson.png ISO12233-4k.png  CustomSheet.png Grid.png

    start = time.time()

    iterationCount = 0
    perIterRays = int(40960 * sampleMultiplier) #40960

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    

    while(True):


        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), 20480)
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
        ProgressBar(elpased / computeTime, 100)

        iterationCount += 1
        
        if(elpased > computeTime):
            image /= 100 
            global FrameCount
            fn = r"NewPDFTest"+str(imageDistance)+"_"+str(FrameCount)
            SaveAsEXR(image, r"resources/Results/ISO12233", fn)
            break

    FrameCount += 1

    return elpased 


def SpotTesting(lens, objectDistance = 13500, focusDistance = 1500, computeTime = 5120, realTimeUpdate = False, lensName=None):
    global FrameCount


    print("Start spot testing")
    source = PointsSource()
    source.isCartesian = False
    ratio = 0.92
    xAngle = ratio * lens.GetAoV()[0] # 19*ratio
    yAngle = ratio * lens.GetAoV()[1] #12*ratio
    sample = 9
    source.GenerateGridSpots(xAngle, yAngle, dist=objectDistance, sampleField=sample)

    # AddRearGroup AddFrontGroup
    # lens.AddFrontGroup([
    #     Surface(100, 5, 15, "FK5"),
    #     Surface(INFINITY, 1, 15)
    # ])
    # lens.UpdateLens()
    # print(lens.GetInfo())

    imager = StdImager(73 , horiPx=6000)
    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=1920)
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    start = time.time()



    iterationCount = 0
    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))
        

    while(True):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(2048), sample*sample, addSecondary=9)

        mainRB, mainRP, _ = lens.Propagate(mainRB)

        print("Size of exiting RB: ", mainRB.value.shape)

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        # DrawRaybatch(mainRB)
        # plt.draw()
        # plt.pause(5)

        image = imager.IntegralRays(mainRB, baseImg=image)
        
        #print(source.sampleRecord)
        elpased = time.time() - start

        print("- Focusing ", focusDistance, " for obj at ", objectDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased) )
        ProgressBar(elpased / computeTime, 100)

        if(realTimeUpdate):
            print("Max ", bd.max(image))
            im.set_data(ImageConversion(image, maxModifier=0.5)) #0.002
            plt.draw()
            plt.pause(0.01)

        if(elpased > computeTime):
            fn = ((r"SpeedMasterSpots" if lensName is None else lensName)+
                  str(objectDistance)+"_"+str(focusDistance))
            SaveAsEXR(image, r"resources/Results", fn)
            break

        iterationCount += 1


    FrameCount +=1


def ReflectionSpotTesting(lens, position, focusDistance = 5000, computeTime = 300, refIte=4, realTimeUpdate = True):


    source = PointsSource()
    source.GenerateFixPoint(position)

    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=1920) 
    #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty(dataType=PRECISION_TYPE)

    start = time.time()
    elpased = time.time() - start

    if(realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    iterationCount = 0
    pupilSampleCount = 128
    print("Pupil sample per iter at ", pupilSampleCount)

    while(elpased < computeTime):
        mainRB = source.EmitSamplesToward(lens.GetFirstElementSamples(pupilSampleCount), 5, addSecondary=5)

        _mainRB, mainRP, mainRB = lens.Propagate(mainRB, reflection=True, iteCount=refIte)
  
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
        ProgressBar(elpased / computeTime, 100)

        iterationCount += 1


    image /= 10.0
    global FrameCount
    fn = r"Batis85StoppedDown"+str(refIte)
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
    image = imager.AcquireEmpty()

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
   

def RayPathTesting(lens, AoV, imageDistance = 200000, imageMinSample = 320, realTimeUpdate = False):

    lens.DrawLens()
    SetUnifScale(50)
    AddXYZ()
    RemoveBG()

    print("New test w/ im Distance ", imageDistance, " sample min ", imageMinSample)

    imager = StdImager(lens.BestFocusBFD(imageDistance)) #32.4
    # Assemble the imaging system 
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

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


def PDATest(lens, tUVIR = 1, AoV =40, imageDistance =200000, imageMinSample=320, realTimeUpdate=True):
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
    image = imager.AcquireEmpty()

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


def CatadioptricTest():
    from Surfaces.Stop import Stop
    from Lens import Lens
    from Material import Material

    SetUnifScale(50)
    AddXYZ()
    RemoveBG()

    s1 = Surface(-147.6, 10.1, 32, "E-FEL2")
    s1.minAperture = 12
    s2 = Stop(-10.1)
    s2.radius = -228.3
    s2.clearSemiDiameter = 32
    s2.minAperture = 12
    s2.material = Material("MIRROR")
    s3 = Surface(-147.6, 10.1, 32)
    s3.minAperture = 12

    cata = Lens()
    cata.AddSurface(s1)
    cata.AddSurface(s2)
    cata.AddSurface(s3)
    cata.UpdateLens()
    cata.DrawLens()

    point = PointsSource(bd.array([[0, 0, -5000, 1, 1, 1]]))
    RB = point.EmitSamplesToward(cata.GetFirstElementSamples(64), 1)

    # pRI = Material("AIR").RI(RB.Wavelength())
    # RB, _TIR, _BoolVig, _Stray = cata.surfaces[0].Trace(RB, pRI)
    #
    # pRI = cata.surfaces[0].material.RI(RB.Wavelength())
    # RB, _TIR, _BoolVig, _Stray = cata.surfaces[1].Trace(RB, pRI)

    RB, _, ref = cata.Propagate(RB)


    DrawRaybatch(RB)

    plt.show()


def MaterialLookUpTest():
    # Example usage:
    excel_file = ReadSheet()

    # Suppose you have n=1.5168 and V=64.1 for the 'd' line
    line = 'D'
    stats = [
        [1.51633 ,  64.14],
        [1.80835 ,  40.55],
        [1.883   ,40.69],
        [1.71736 ,  29.5],
        [1.72825 ,  28.32],
        [1.883   ,40.69],
        [1.883   ,40.69],
        [1.60342 ,  38.03],
        [1.80835 ,  40.55]
    ]

    for item in stats:
        n_val = item[0]
        v_val = item[1]

        result_df = FindClosestMaterials(excel_file, line, n_val, v_val, top_k=5).to_string(index=False)
        print("Closest matches:")
        print(result_df)


def RefDepthTest():
    refDepthList = [4, 5, 6, 7]
    lens = Biotar50mmf14()
    position = AngleFieldToCartesian(18, 10, -200000)

    for r in refDepthList:
        ReflectionSpotTesting(lens, position, refIte=r, focusDistance=1500, imageMinSample=8192, realTimeUpdate=False)


def AsphericTest():
    from Surfaces.EvenAspheric import EvenAspheric


def DefocusTests():


    reader = LensFromZmx(RectPath(r"resources/Zmx/Helios-44.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()
    SpotTesting(lens, objectDistance=13500, focusDistance=500, computeTime=30 * 60, realTimeUpdate=True, lensName="Helios")
    SpotTesting(lens, objectDistance=13500, focusDistance=1500, computeTime=30 * 60, realTimeUpdate=False, lensName="Helios")
    SpotTesting(lens, objectDistance=13500, focusDistance=135000, computeTime=30 * 60, realTimeUpdate=False, lensName="Helios")

    reader = LensFromZmx(RectPath(r"resources/Zmx/Jupiter-12.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()
    SpotTesting(lens, objectDistance=13500, focusDistance=1000, computeTime=30 * 60, realTimeUpdate=False, lensName="Jupiter")
    SpotTesting(lens, objectDistance=13500, focusDistance=1500, computeTime=30 * 60, realTimeUpdate=False, lensName="Jupiter")
    SpotTesting(lens, objectDistance=13500, focusDistance=135000, computeTime=30 * 60, realTimeUpdate=False, lensName="Jupiter")

    reader = LensFromZmx(RectPath(r"resources/Zmx/LeicaSummicron50f2.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()
    SpotTesting(lens, objectDistance=13500, focusDistance=1000, computeTime=30 * 60, realTimeUpdate=False, lensName="Summicron")
    SpotTesting(lens, objectDistance=13500, focusDistance=1500, computeTime=30 * 60, realTimeUpdate=False, lensName="Summicron")
    SpotTesting(lens, objectDistance=13500, focusDistance=135000, computeTime=30 * 60, realTimeUpdate=False, lensName="Summicron")

    reader = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()
    SpotTesting(lens, objectDistance=13500, focusDistance=1500, computeTime=30 * 60, realTimeUpdate=False, lensName="CanonL")
    SpotTesting(lens, objectDistance=13500, focusDistance=135000, computeTime=30 * 60, realTimeUpdate=False, lensName="CanonL")


def NewWavelengthTest():
    from Util.ColorPDF import ColorPDF

    colorData = bd.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, .5, 0],
                          [.5, 1, .5],
                          [0, .5, 1]])

    converter = ColorPDF()
    wa = converter.ColorToWavelength(colorData, perChannelSample=64)

    # print(wa)
    RGBack = converter.SpectralResponse(wa[:, 0], wa[:, 1])
    print(bd.sort(RGBack))
    converter.PlotDistribution()


def FilmTest():
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF
    from Imagers.Technicolor import Technicolor
    from ObjectSpace.Images import Image2DFlat
    from Util.Misc import RectPath

    converter = ColorPDF()
    converter.gainR = 2
    converter.gainB = 2

    im = Image2DFlat()
    im.LoadFromEXR(RectPath(r"resources/Results/NewZDepthClose9hr.exr"))
    im.rgbArray = im.rgbArray/128
    #im.Show2D()

    f = Technicolor(converter)

    print(f.ImageStats(im.rgbArray))

    # im.rgbArray = f.Halation2D(im.rgbArray)[1]
    im.rgbArray = f.ApplyGrainAndNoise(im.rgbArray)
    #im.rgbArray = f.DensityCurve(im.rgbArray)

    print(f.ImageStats(im.rgbArray))
    # im.Show2D()

    SaveAsEXR(im.rgbArray, r"resources/Results", "Technicolor")


def SpeedMasterTest(lens=None, imageDistance=1500, focusDistance=1500, computeTime=2*60*60, realTimeUpdate=False):
    print("New test w/ im Distance ", imageDistance)

    reader = LensFromZmx(RectPath(r"resources/Zmx/SpeedMaster50f0.95.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()
    # lens.SetAperture(5.6)
    print(lens.GetInfo())

    print("Best focus", lens.BestFocusBFD(focusDistance))  # 32.926564?
    imager = StdImager(lens.BestFocusBFD(focusDistance))  # 32.4 lens.BestFocusBFD(imageDistance)

    # Assemble the imaging system
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    sourceImage = Image2DFlat()
    sourceImage.distance = imageDistance
    sourceImage.horizontalAoV = lens.GetAoV()[0] * 2
    sourceImage.imageDimensionOverride = 1920
    sourceImage.LoadFrom8bit(r"resources/ISO12233-4k.png")
    # Henri-Cartier-Bresson.png ISO12233-4k.png  CustomSheet.png Grid.png

    start = time.time()

    iterationCount = 0

    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    while (True):

        mainRB = sourceImage.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), 20480)
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
        ProgressBar(elpased / computeTime, 100)

        iterationCount += 1

        if (elpased > computeTime):
            image /= 100
            global FrameCount
            fn = r"SpeedMasterTest2hr"
            SaveAsEXR(image, r"resources/Results/ISO12233", fn)
            break

    FrameCount += 1

    return elpased


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

    #MaterialLookUpTest()

    objectDistance = [
        450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000
    ]
    # objectDistance = [ 
    #     850, 875, 900, 925, 950, 975, 1000, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 
    #     5000, 6000, 7000, 8000, 10000, 12500, 15000, 20000, 30000, 50000, 75000, 100000
    # ]
    
    angleFieldX = bd.linspace(-20, 20, len(objectDistance)) * 0.3
    angleFieldY = bd.linspace(-13, 13, len(objectDistance)) * 0.3

    # lens = Zhongyi50f095()
    # lens = Industar50_50mmf35()
    # lens = ZeissHologon15mmf8() #AoV 104
    # lens = Sonnar50mmF15()
    # lens = CanonFD50mmf18()
    # lens = CanonEF50mmf12L()
    reader = LensFromZmx(RectPath(r"resources/Zmx/AdaptAll500mmf8.zmx"))
    reader = LensFromZmx(RectPath(r"resources/Zmx/Helios-44.zmx"))
    reader = LensFromZmx(RectPath(r"resources/Zmx/SpeedMaster50f0.95.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()
    # lens.SetAperture(5.6)
    print(lens.GetInfo())

    # ReflectionSpotTesting(lens, AngleFieldToCartesian(12, 12, -200000), focusDistance=1500, computeTime=1.5 * 60 * 60, realTimeUpdate=False)
    # return
    #ISO12233Test(lens, computeTime=10*60, realTimeUpdate=False)
    SpotTesting(lens, objectDistance=13500, focusDistance=1500,  computeTime=20 * 60, realTimeUpdate=False)

    return

    apertureValue = [1.43, 1.5, 1.8, 2, 2.5, 2.8, 3.2, 4, 4.8, 5.6, 6.3, 8, 9, 11]
    interp_values = []
    for i in range(len(apertureValue) - 1):
        a = apertureValue[i]
        b = apertureValue[i + 1]
        midpoint = (a + b) / 2
        interp_values.extend([a, midpoint])
    interp_values.append(apertureValue[-1])
    for a in interp_values:
        lens.SetAperture(a)
        SpotTesting(lens, computeTime=45*60, realTimeUpdate=False)

    # return
    # lens.SetAperture(4)
    #RayPathTesting(lens, AoV=40)
    # for o in objectDistance:
    #     ISO12233Test(lens, AoV=32, imageDistance=o, imageMinSample=256, realTimeUpdate=True) #4096: 10 hours

    # [0, 0.15, 0.3, 0.45, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.]
    # for t in [ 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.]:
    #     PDATest(lens, t, AoV=38.75, imageDistance=100000, imageMinSample=2048, realTimeUpdate=False)

    for ax, ay, d in zip(angleFieldX, angleFieldY, objectDistance):
    # #     ISO12233Test(lens, imageDistance=d, imageMinSample=512, realTimeUpdate=False)
    #
        # position = bd.array([1000, 600, -20000])
        position = AngleFieldToCartesian(ax, ay, -d)
    #     #print("Current origin position: ", position)
        # ReflectionSpotPositionOrig(lens, position, focusDistance=1500, imageMinSample=2048, realTimeUpdate=True)
        ReflectionSpotTesting(lens, position, focusDistance=1500, computeTime=13*60*60, realTimeUpdate=False)
    #
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