

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
from ObjectSpace.Image2DFlat import Image2DFlat
from ObjectSpace.ImageVariDepth import Image2DVariDepth
from ObjectSpace.Attenuator import  DepthVisualizer
from ObjectSpace.Fog import FogAttenuator
from ExampleLenses import Biotar50mmf14, CanonEF50mmf12L
from src.Surfaces.EvenAspheric import EvenAspheric
from src.Util.Globals import INFINITY
from Raytracing.Emission import EmitField
from Surfaces.MetalBoundary import MetalBoundary
from Surfaces.Surface import Surface



def StereoImageDisplay(imageMinSample = 128, realTimeUpdate = True):



    img = Image2DVariDepth()
    img.imageDimensionOverride = 100
    img.LoadFromEXR(r"resources/allChannels.exr")
    # img.UpdateDepthRange()

    img.DrawImage()

    RemoveBG()
    SetUnifScale(10000)
    plt.show()


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


def SingleImgRens():
    from ImagingSystem import ImagingSystem
    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Util.Globals import RefreshRNG

    lens = LensFromZmx(RectPath(r"resources/Zmx/Elmarit90f2.8.zmx")).GetLens()
    print(lens.GetInfo())
    imager = StdImager(horiPx=1920)
    IS = ImagingSystem(lens, imager)

    FG = Image2DVariDepth()
    FG.horizontalAoV = lens.GetAoV(halfAngle=False)[0]
    FG.LoadFromEXR(r"resources/LeicaFG.exr")
    IS.singleObject = FG

    RefreshRNG(435789)
    IS.SingleLayerAlpha(focusDistance=1500, renderTime=2 * 60, fileName="leicaSingleEXRSaveTest", flareGlare=False)


def ISAnamorphicTest():
    from ImagingSystem import ImagingSystem
    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Util.Globals import RefreshRNG
    from ExampleLenses import Cooke_i_S35_40mm_NaiveCoating


    # Create or load a lens
    # lens = LensFromZmx(RectPath(r"resources/Zmx/Elmarit90f2.8.zmx")).GetLens()
    #lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens = LensFromZmx(RectPath(r"resources/Zmx/iS35_2x_40mm.zmx")).GetLens(exchangeAxis=True)
    lens.AddSurfaceDefect()
    lens = Cooke_i_S35_40mm_NaiveCoating()

    # Instantiate an imager, adjust its attributes
    imager = StdImager(horiPx=1920)
    imager.LoadS35Preset()
    imager.ScaleWH(0.9)

    # Read input images
    # FG = Image2DVariDepth()
    # Pass in the lens angle of view to establish the scene size
    # FG.horizontalAoV = lens.GetAoV(halfAngle=False)[0]
    # FG.LoadFromEXR(r"resources/LeicaFG.exr")
    # BG = Image2DVariDepth()
    # BG.horizontalAoV = lens.GetAoV(halfAngle=False)[0]
    # BG.LoadFromEXR(r"resources/LeicaBG.exr")

    # Load images into a stack, more efficient this way
    # exampleStack = ImageStack()
    # exampleStack.AddImage(BG, "BG")
    # exampleStack.AddImage(FG, "FG")


    # Assemble an imaging system
    IS = ImagingSystem(lens, imager)
    # Set the scene
    IS.object = ExampleStack3D(28)
    # Aside from a stack, many other classes in ObjectSpace can also be passed in here

    RefreshRNG(3452345)
    # Render the scene into an image
    IS.Render(focusDistance=1000, renderTime=4*60, fileName="EvennessTest4MinNewExp", realTimeUpdate=False, flareGlare=False)
    # IS.RenderFlareOnly(focusDistance=1000, renderTime= 40 * 60, fileName="AnamorphicBlueShort", realTimeUpdate=False, flareGlare=True)


def HeliosComparison():
    from ImagingSystem import ImagingSystem
    from ObjectSpace.ImageStack import ImageStack, ExampleStack3D
    from Util.Globals import RefreshRNG
    import time


    # Create or load a lens
    Helios = LensFromZmx(RectPath(r"resources/Zmx/Helios-44.zmx")).GetLens()
    Helios.AddSurfaceDefect(0.2)

    Biotar = LensFromZmx(RectPath(r"resources/Zmx/Biotar58mmf2.zmx")).GetLens()
    Biotar.AddSurfaceDefect(0.2)

    imager = StdImager(horiPx=1920)

    BiotarIS = ImagingSystem(Biotar, imager)
    BiotarIS.object = ExampleStack3D()

    HeliosIS = ImagingSystem(Helios, imager)
    HeliosIS.object = ExampleStack3D()

    RefreshRNG(25353)
    print("Scene rendering example for Helios")
    HeliosIS.Render(focusDistance=6500, renderTime=11*60*60, fileName="HeliosSceneRender", realTimeUpdate=False, flareGlare=False)
    time.sleep(15 * 60)

    print("ISO12233 example for Helios")
    HeliosIS.ISO12233(objectDistance=1500, focusDistance=1500, renderTime=2 * 60 * 60, fileName="HeliosISO12233", realTimeUpdate=False)
    time.sleep(15 * 60)

    BiotarIS.imager = StdImager(horiPx=6000)
    HeliosIS.imager = StdImager(horiPx=6000)
    for d in [1500, 2000, 3000, 4500, 6500, 9000, 12000, 16000, 21000]:
        print("Focusing at ", d)
        HeliosIS.SpotGrid(objectDistance=10000, focusDistance=1500, renderTime=10 * 60, fileName="HeliosGrid_" + str(d), realTimeUpdate=False)
        time.sleep(5*60)


def ISSpotTest():
    from ImagingSystem import ImagingSystem

    lens = LensFromZmx(RectPath(r"resources/Zmx/CanonEF50f1.2L.zmx")).GetLens()
    lens.AddSurfaceDefect()
    imager = StdImager(horiPx=6000)

    IS = ImagingSystem(lens, imager)
    IS.SpotGrid(objectDistance=20000, focusDistance=1500, renderTime=30*60, fileName="DustTest", realTimeUpdate=False)


def PureArtifactTest():
    from Surfaces.SurfaceModulator import  Dust
    from Surfaces.OnionRing import OnionRing

    #OR = OnionRing()
    #OR.semiDiameter = 22
    #OR.frontVertex = bd.array([0, 0, 0])
    #OR.Generate()

    #OR.ShowNormalMap()

    d = Dust(3, 1)
    d.semiDiameter = 22
    d.frontVertex = bd.array([0, 0, 0])
    d.Generate()
    d.ShowNormalMap(512)


def main():

    from Util.Backend import backend_name
    print("Currently using ", backend_name)

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
    # StackTestDigital(renderTime, distance[18], "NewRacking", realTimeUpdate=False, infoArg=1)
    # StackTestDigital(renderTime, distance[19], "NewRacking", realTimeUpdate=False, infoArg=1)
    # StackTestDigital(renderTime, distance[20], "NewRacking", realTimeUpdate=False, infoArg=1)
    ISAnamorphicTest()
    # PureArtifactTest()
    # HeliosComparison()
    # ISSpotTest()
    # PureArtifactTest()


    # FocusFalloffLenSelect(r"resources/Zmx/SpeedMaster50f0.95.zmx", renderTime, 1350, "FalloffTestSpeedMaster", realTimeUpdate=False)
    # StackTestDigital(5*60, distance[0], "newPDFSeriesFilm", realTimeUpdate=False)
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
    main()

