import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG, DrawNormal
from Util.Misc import ProgressBar
from Util.DiaphragmSVG import DiaphragmBlades
from src.ZmxReader import LensFromZmx
from Util.SpatialEllipse import SpatialCircle
from Util.Misc import RectPath
from Imagers.Standard import StdImager
from ObjectSpace.Images import Image2DFlat
from ObjectSpace.ImageVariDepth import Image2DVariDepth
from ExampleLenses import Biotar50mmf14
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

    projectRB = EmitField(0, 10, sampleTargets=asphS.SampleFromSD())

    intersections = asphS.Intersection(projectRB)
    normals = asphS.Normal(intersections[0])

    asphS.DrawSurface(drawProxy=False)

    DrawNormal(intersections[0], normals)
    DrawPoints(intersections[0])
    plt.show()


def StereoImageDisplay(imageMinSample = 4096, realTimeUpdate = True):

    img = Image2DVariDepth()
    img.imageDimensionOverride = 200
    # img.nearClipping = 200
    img.zDepthMappingRange = bd.array([800, 10000])

    img.LoadFrom8bit(r"resources/DualTest_RGB.png", r"resources/DualTest_Z.png")

    img.DrawImage()
    plt.show()


def StereoImageTest(imageMinSample = 10240, realTimeUpdate = False):

    img = Image2DVariDepth()
    # img.imageDimensionOverride = 200
    # img.nearClipping = 200
    img.zDepthMappingRange = bd.array([500, 10000])

    img.LoadFrom8bit(r"resources/DualTest_RGB.png", r"resources/DualTest_Z.png")

    #img.DrawImage()
    #plt.show()


    lens = Biotar50mmf14()
    imager = StdImager(lens.BestFocusBFD(1500))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty()

    iterationCount = 0
    start = time.time()
    if (realTimeUpdate):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(ImageConversion(image))

    while (True):

        mainRB = img.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), 1024)
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
        ProgressBar(iterationCount / imageMinSample, 100)
        iterationCount += 1

        if (iterationCount > imageMinSample):
            image /= 100
            global FrameCount
            fn = r"Stereo test"
            SaveAsEXR(image, r"resources/Results/ISO12233", fn)
            break


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

    RB = EmitField(0, 0, 5000, sampleTargets=pseudoSurface.SampleFromSD())

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


def BladeTest():
    rot = DiaphragmBlades(RectPath(r"resources/diaphragm.svg"))
    rot.DuplicateAroundCenter(6, 60)
    rot.RotateAllBlades(10)
    plt.show(ImageConversion(rot.toArray(512, 512)))


def main():
    BladeTest()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

