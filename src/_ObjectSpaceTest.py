

import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.ImageIO import ImageConversion, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG, DrawNormal
from Util.Misc import ProgressBar
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm
from src.Util.ColorPDF import ColorPDF
from src.ZmxReader import LensFromZmx
from Util.SpatialEllipse import SpatialCircle
from Util.Misc import RectPath

from Imagers.Standard import StdImager
from ObjectSpace.Images import Image2DFlat
from ObjectSpace.ImageVariDepth import Image2DVariDepth
from ObjectSpace.Attenuator import  DepthVisualizer
from ObjectSpace.Fog import FogAttenuator
from ExampleLenses import Biotar50mmf14, CanonEF50mmf12L
from src.Surfaces.EvenAspheric import EvenAspheric
from src.Util.Globals import INFINITY
from Raytracing.Emission import EmitField
from Surfaces.MetalBoundary import MetalBoundary
from Surfaces.Surface import Surface




def AsphTest():
    SetUnifScale()
    AddXYZ(70)
    RemoveBG()


    asphS = EvenAspheric(INFINITY, 0.1510, 24, "FD60", -1.0,
                         [-7.39600E-03, 2.39000E-07, 2.21800E-09, -3.20700E-12, 1.92500E-15])
    # asphS = EvenAspheric(85.289, 8, 20, "FD60", 0,
    #                      [0, -1.473089E-06, 1.381523E-10, 2.077557E-11, -7.423427E-14, 1.589502E-16])

    asphS.SetCumulative(2)

    projectRB = EmitField(0, 10, sampleTargets=asphS.SampleFromClearAperture())

    intersections = asphS.Intersection(projectRB)
    normals = asphS.Normal(intersections[0])

    asphS.DrawSurface(drawProxy=False)

    DrawNormal(intersections[0], normals)
    DrawPoints(intersections[0])
    plt.show()


def StereoImageDisplay(imageMinSample = 128, realTimeUpdate = True):



    img = Image2DVariDepth()
    img.imageDimensionOverride = 100
    img.LoadFromEXR(r"resources/allChannels.exr")
    # img.UpdateDepthRange()

    img.DrawImage()

    RemoveBG()
    SetUnifScale(10000)
    plt.show()


def StereoImageTest(imageMinSample = 512, realTimeUpdate = True):

    #bg = Image2DFlat()
    bg = Image2DVariDepth()
    bg.distance = 200000
    #bg.LoadFrom8BitPNG(r"resources/YourTaxReturn.png")
    bg.LoadFromEXR(r"resources/DepthSceneBG.exr")

    img = Image2DVariDepth()
    #img.LoadFromEXR(r"resources/allChannels.exr")
    img.LoadFromEXR(r"resources/DepthSceneMG2.exr")
    # img.zDepthMappingRange = [2000, 10000]
    # img.LoadFrom8bit(r"resources/DualTest_RGB.png", r"resources/DualTest_Z.png")
    print(img.GetAOVNames())

    #img.DrawImage()
    #plt.show()


    lens = Biotar50mmf14()
    imager = StdImager(lens.BestFocusBFD(5000))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))

    while (True):

        mainRB = img.ReceiveAndEmitTowards(
            lens.entrancePupil.GetSamplePoints(512),
            bg.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), 1024),
            1024)

        # mainRB = img.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512),1024)

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)

        if (realTimeUpdate):
            im.set_data(ImageConversion(image, flipH=True, maxModifier=0.25))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elpased = time.time() - start
        ProgressBar(iterationCount / imageMinSample, 100)
        iterationCount += 1

        if (iterationCount > imageMinSample):
            image /= 100
            global FrameCount
            fn = r"LayerTest"
            SaveAsEXR(image, r"resources/Results", fn)
            break


def StackTest(renderTime = 20*60, focusDistance=5000, filename = r"NewPDF", aperture=None, realTimeUpdate = False):

    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF

    print("Currently using ", backend_name)

    stack = ExampleStack3D()
    att = DepthVisualizer()
    fog = FogAttenuator()

    lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens.UpdateLens()
    if aperture is not None:
        lens.SetAperture(aperture)

    sr = ColorPDF()
    # sr.normGainB = 1.25
    imager = Film(sr, lens.BestFocusBFD(focusDistance))
    #imager = StdImager(lens.BestFocusBFD(focusDistance))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))


    while (True):
        recorder = time.time()
        mainRB = stack.EmitTowards(lens.entrancePupil.GetSamplePoints(512), 40960)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        #mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)

        #mainRBZ, mainRP, reflectedRB = lens.Propagate(mainRBZ, reflection=False)
        #mainRBZ, _tir, _vig = imager.IntersectRays(mainRBZ)
        # mainRP.Append(mainRB, _tir, _vig)
        #print(mainRB.ToString(30))

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)
        #imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)


        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn)
            #SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def StackTestFilmBalance(renderTime = 20*60, focusDistance=5000, filename = r"NewPDF", aperture=None, realTimeUpdate = False):

    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF
    from Util.Globals import Channels

    print("Currently using ", backend_name)

    stack = ExampleStack3D()
    att = DepthVisualizer()
    fog = FogAttenuator()

    lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens.UpdateLens()
    if aperture is not None:
        lens.SetAperture(aperture)

    sr = ColorPDF()
    #sr.gainR = 1.75
    sr.gainG = 0.75
    sr.gainB = 1.75

    # sr.PlotDistribution()
    # plt.show()

    imager = Film(sr, lens.BestFocusBFD(focusDistance))
    imager.dyeSpectralPairs = {
        Channels.R: Channels.G,
        Channels.G: Channels.R,
        Channels.B: Channels.B,
    }
    #imager = StdImager(lens.BestFocusBFD(focusDistance))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))


    while (True):
        recorder = time.time()
        mainRB = stack.EmitTowards(lens.entrancePupil.GetSamplePoints(512), 40960)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        #mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)

        #mainRBZ, mainRP, reflectedRB = lens.Propagate(mainRBZ, reflection=False)
        #mainRBZ, _tir, _vig = imager.IntersectRays(mainRBZ)
        # mainRP.Append(mainRB, _tir, _vig)
        #print(mainRB.ToString(30))

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)
        #imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)


        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn+str(focusDistance))
            #SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def StackTestDigital(renderTime = 20*60, focusDistance=5000, filename = r"NewPDF", aperture=None, realTimeUpdate = False, infoArg=0):
    from Surfaces.ManualAperture import ManualAperture
    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from ExampleLenses import HazySonnar, ReverseHelios
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF

    print("Currently using ", backend_name)

    stack = ExampleStack3D()

    lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    # lens = LensFromZmx(RectPath(r"resources/Zmx/Helios-44.zmx")).GetLens()
    lens.UpdateLens()
    #lens.FlipElement(5, True)

    if(infoArg == 0):
        ma = ManualAperture()
        ma.isCircular = False
        lens.AddFrontGroup([ma])

    if aperture is not None:
        lens.SetAperture(aperture)

    #sr = ColorPDF()
    #sr.normGainB = 1.25
    #imager = Film(sr, lens.BestFocusBFD(focusDistance))
    imager = StdImager(lens.BestFocusBFD(focusDistance))#-0.8)+3
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()



    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))


    while (True):
        recorder = time.time()
        mainRB = stack.EmitTowards(lens.entrancePupil.GetSamplePoints(512), 20480)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        #mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)

        #mainRBZ, mainRP, reflectedRB = lens.Propagate(mainRBZ, reflection=False)
        #mainRBZ, _tir, _vig = imager.IntersectRays(mainRBZ)
        # mainRP.Append(mainRB, _tir, _vig)
        #print(mainRB.ToString(30))

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)
        #imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)


        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn+str(focusDistance))
            #SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def StackTestDigitalLenSelect(lensPath, renderTime = 20*60, focusDistance=5000, filename = r"NewPDF", aperture=None, realTimeUpdate = False):

    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF

    print("Currently using ", backend_name)

    stack = ExampleStack3D()
    att = DepthVisualizer()
    fog = FogAttenuator()

    #lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens = LensFromZmx(RectPath(lensPath)).GetLens()
    lens.UpdateLens()
    if aperture is not None:
        lens.SetAperture(aperture)

    #sr = ColorPDF()
    #sr.normGainB = 1.25
    #imager = Film(sr, lens.BestFocusBFD(focusDistance))
    imager = StdImager(lens.BestFocusBFD(focusDistance))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))


    while (True):
        recorder = time.time()
        mainRB = stack.EmitTowards(lens.entrancePupil.GetSamplePoints(512), 20480)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        #mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)

        #mainRBZ, mainRP, reflectedRB = lens.Propagate(mainRBZ, reflection=False)
        #mainRBZ, _tir, _vig = imager.IntersectRays(mainRBZ)
        # mainRP.Append(mainRB, _tir, _vig)
        #print(mainRB.ToString(30))

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)
        #imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)


        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn+str(focusDistance))
            #SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def ImgRefLenSelect(lensPath, renderTime = 20*60, focusDistance=5000, filename = r"NewPDF", aperture=None, realTimeUpdate = False):

    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF

    print("Currently using ", backend_name)

    stack = ExampleStack3D()
    att = DepthVisualizer()
    fog = FogAttenuator()

    #lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens = LensFromZmx(RectPath(lensPath)).GetLens()
    lens.UpdateLens()
    if aperture is not None:
        lens.SetAperture(aperture)

    #sr = ColorPDF()
    #sr.normGainB = 1.25
    #imager = Film(sr, lens.BestFocusBFD(focusDistance))
    imager = StdImager(lens.BestFocusBFD(focusDistance))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()
    refImage = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))


    while (True):
        recorder = time.time()
        mainRB = stack.EmitTowards(lens.entrancePupil.GetSamplePoints(512), 1024)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        #mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=True)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        image = imager.IntegralRays(mainRB, baseImg=image, polarized=True)

        reflectedRB, _tir, _vig = imager.IntersectRays(reflectedRB)
        refImage= imager.IntegralRays(reflectedRB, baseImg=refImage, polarized=True)

        #imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)


        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn+str(focusDistance))
            SaveAsEXR(refImage, r"resources/Results", fn + "Ref" +str(focusDistance))
            #SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def FocusFalloffLenSelect(lensPath, renderTime = 20*60, focusDistance=5000, filename = r"NewPDF", aperture=None, realTimeUpdate = False):

    from ObjectSpace.ImageVariDepth import  Image2DVariDepth
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF

    print("Currently using ", backend_name)

    imSource = Image2DVariDepth()
    imSource.LoadFromEXR(RectPath(r"resources/FocusFalloffGrid.exr"))

    lens = LensFromZmx(RectPath(lensPath)).GetLens()
    lens.UpdateLens()
    if aperture is not None:
        lens.SetAperture(aperture)

    #sr = ColorPDF()
    #sr.normGainB = 1.25
    #imager = Film(sr, lens.BestFocusBFD(focusDistance))
    imager = StdImager(lens.BestFocusBFD(focusDistance))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))


    while (True):
        recorder = time.time()
        mainRB = imSource.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), 20480)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        #mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)

        #mainRBZ, mainRP, reflectedRB = lens.Propagate(mainRBZ, reflection=False)
        #mainRBZ, _tir, _vig = imager.IntersectRays(mainRBZ)
        # mainRP.Append(mainRB, _tir, _vig)
        #print(mainRB.ToString(30))

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)
        #imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)


        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn+str(focusDistance))
            #SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def DoubleImgTest():
    targets = bd.array([
        [1, 2, 25],
        [2, 4, 25],
        [-2, 3, 25],
        [1, -2, 25]
    ])

    imgFG = Image2DFlat()
    imgFG.imageDimensionOverride = 200
    imgFG.distance = 200
    imgFG.LoadFrom8BitPNG(r"resources/2330D1FG.png")

    imgBG = Image2DFlat()
    imgBG.imageDimensionOverride = 200
    imgBG.distance = 500
    imgBG.LoadFrom8BitPNG(r"resources/2330D1BG.png")

    imgFG.DrawImage()
    imgBG.DrawImage()

    RemoveBG()
    SetUnifScale(700)
    plt.show()


def MetalTest():
    SetUnifScale()
    AddXYZ(70)
    RemoveBG()

    pseudoSurface = Surface(INFINITY, 1, 10)
    pseudoSurface.SetCumulative(0)

    e1 = SpatialCircle(0, 10)
    e2 = SpatialCircle(5, 5)
    mb = MetalBoundary(e1, e2)

    RB = EmitField(0, 0, 5000, sampleTargets=pseudoSurface.SampleFromClearAperture())

    inters = mb.Intersection(RB)
    normals = mb.Normal(inters[0])
    refl, _ = mb.Trace(RB)

    DrawRaybatch(refl, lLength=2)
    DrawNormal(inters[0], normals)

    mb.DrawSurface()
    plt.show()


def ZmxParse():
    print("=================Parse===============")
    reader = LensFromZmx(RectPath(r"resources/Zmx/LeicaSummicron50f2.zmx"))

    exampleLens = reader.GetLens()

    exampleLens.UpdateLens()

    SetUnifScale(50)
    AddXYZ()
    RemoveBG()
    print(exampleLens.GetInfo())
    print(exampleLens.SurfaceReport())

    exampleLens.DrawLens()
    # exampleLens.entrancePupil.DrawSamplePoints()
    # exampleLens.entrancePupil.DrawSurface()
    # exampleLens.frontPincipalPlane.DrawSamplePoints()

    plt.show()


def StackTest2D(iStack, renderTime = 30*60, focusDistance=1500, filename = r"Stack2DHighlightRecon", aperture=None, realTimeUpdate = False):
    from ObjectSpace.ImageStack import ImageStack, ExampleStack2D
    from Imagers.Film import Film
    from Util.ColorPDF import ColorPDF

    print("Currently using ", backend_name)

    stack = iStack #ExampleStack2D()

    lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens.UpdateLens()
    if aperture is not None:
        lens.SetAperture(aperture)

    imager = StdImager(lens.BestFocusBFD(focusDistance))
    # imager = StdImager(lens.BestFocusBFD(focusDistance))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AcquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image, flipH=True))

    while (True):
        recorder = time.time()
        mainRB = stack.EmitTowards(lens.entrancePupil.GetSamplePoints(512), 40960)
        # mainRB = fog.Attenuate(mainRB)
        # mainRB = att.ColorizeDepthZones(mainRB, 5000, 20000)
        # mainRBZ = att.Attenuate(mainRB)
        print("Creating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
        print("Propagating RB took ", time.time() - recorder)
        recorder = time.time()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)

        # mainRBZ, mainRP, reflectedRB = lens.Propagate(mainRBZ, reflection=False)
        # mainRBZ, _tir, _vig = imager.IntersectRays(mainRBZ)
        # mainRP.Append(mainRB, _tir, _vig)
        # print(mainRB.ToString(30))

        image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)
        # imageZ = imager.IntegralRays(mainRBZ, baseImg=image, polarized=False)
        print("Integral image took ", time.time() - recorder)
        recorder = time.time()

        if (realTimeUpdate):
            print("Max value ", bd.max(image))
            im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
            plt.draw()
            plt.pause(0.01)

            # print(source.sampleRecord)
        elapsed = time.time() - start
        ProgressBar(elapsed / renderTime, 100)
        iterationCount += 1

        print("House keep took ", time.time() - recorder)

        if (elapsed > renderTime):
            image /= 100
            global FrameCount
            fn = filename
            SaveAsEXR(image, r"resources/Results", fn)
            # SaveAsEXR(imageZ, r"resources/Results", fn+"Z")

            break

        recorder = time.time()


def ISTest():
    from ImagingSystem import ImagingSystem
    from ObjectSpace.ImageStack import ImageStack

    lens = LensFromZmx(RectPath(r"resources/Zmx/Elmarit90f2.8.zmx")).GetLens()
    lens.UpdateLens()
    print(lens.GetInfo())

    imager = StdImager(horiPx=2160)

    FG = Image2DVariDepth()
    FG.horizontalAoV = 23
    FG.LoadFromEXR(r"resources/LeicaFG.exr")
    print("FG Stats ======================")
    print(FG.Stats())
    BG = Image2DVariDepth()
    BG.horizontalAoV = 23
    BG.LoadFromEXR(r"resources/LeicaBG.exr")
    print("BG Stats ======================")
    print(BG.Stats())
    exampleStack = ImageStack()
    exampleStack.AddImage(BG, "BG")
    exampleStack.AddImage(FG, "FG")


    IS = ImagingSystem(lens, imager)
    IS.object = exampleStack

    IS.Render(focusDistance=1500, renderTime=300, fileName="LeicaTest", realTimeUpdate=False)


def main():

    # from ObjectSpace.ImageStack import ImageStack, ExampleStack2D, ExampleStack2DNoGain
    # StackTest2D(ExampleStack2DNoGain(), renderTime=6*60*60, filename=r"Stack2DNoRecon")
    # StackTest2D(ExampleStack2D(), renderTime=6*60*60, filename=r"Stack2DHighlightRecon")
    # return

    # 21 entries
    distance = bd.array([1, 1.25, 1.55, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 10, 13, 16, 20, 30, 50, 70, 100, 200])* 1000.0
    renderTime = 3 * 60 * 60  # For Hayes Forum testing it is 3 hours, file name HayesFocusRacking
    aperture = [None,   None, None,  1.8,     2.8,      4]
    # 11h = 39600s, 7 images, 5657 per image

    i = 7
    StackTestDigital(renderTime, distance[9], "NewRacking", realTimeUpdate=False, infoArg=1)
    StackTestDigital(renderTime, distance[10], "NewRacking", realTimeUpdate=False, infoArg=1)
    StackTestDigital(renderTime, distance[11], "NewRacking", realTimeUpdate=False, infoArg=1)
    StackTestDigital(renderTime, distance[12], "NewRacking", realTimeUpdate=False, infoArg=1)

    # FocusFalloffLenSelect(r"resources/Zmx/SpeedMaster50f0.95.zmx", renderTime, 1350, "FalloffTestSpeedMaster", realTimeUpdate=False)
    #StackTest(renderTime, distance[i], "newPDFSeriesFilm", realTimeUpdate=False)
    #StackTestFilmBalance(1.5*60*60, distance[i], "HayesWhiteBalance", realTimeUpdate=False)

    return

    # FocusFalloffLenSelect(r"resources/Zmx/SpeedMaster50f0.95.zmx", 30 * 60, 1500, "SpeedMaster50FallOff" , realTimeUpdate=False)

    # "SpeedPanchro50f2.zmx",
    # "SPii50mmf2.zmx",
    # "CanonNFD50f1.4.zmx",
    # "LeicaSummicron50f2.zmx"
    # "CanonEF50f1.2L.zmx",

    # for p in ["Industar-50.zmx",
    #           "Helios-44.zmx",
    #           "Biotar50f1.4.zmx"]:
    #     StackTestDigitalLenSelect(r"resources/Zmx/"+p, renderTime, distance[i], "LensTest"+p, realTimeUpdate=True)
        # FocusFalloffLenSelect(r"resources/Zmx/"+p, renderTime, 1250, "FalloffTest"+p, realTimeUpdate=True)
        # ImgRefLenSelect(r"resources/Zmx/"+p, 120, distance[i], "RefComp"+p, realTimeUpdate=False)

    for a in [None, 1.22, 1.4, 1.8]: #1.22, 1.4, 1.8, 2, 2.8 , 4, 5.6
        StackTestDigital(renderTime, distance[0], "MatteBox"+str(a), aperture=a, realTimeUpdate=False)
        i +=1


    # StackTest(renderTime, distance[i], "Focus" + str(distance[i]), realTimeUpdate=False)

    # BladeTest()
    # StereoImageTest()
    # StackTest()


if __name__ == "__main__":
    ISTest()

