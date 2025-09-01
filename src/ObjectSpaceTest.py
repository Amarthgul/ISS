


from PIL import Image
import time
import matplotlib.pyplot as plt
import OpenEXR

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, ImageConversionAverage, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG, DrawAspherical, DrawAsphericalProfile, DrawNormal
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
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18, ZeissHologon15mmf8, Mug, Sonnar50mmF15
from src.Surfaces.EvenAspheric import EvenAspheric
from src.Util.Globals import INFINITY
from Raytracing.Emission import EmitField

def AsphTest():
    SetUnifScale()
    AddXYZ(70)
    RemoveBG()


    asphS = EvenAspheric(INFINITY, 0.1510, 24, -1.0,
                         [-7.39600E-03, 2.39000E-07, 2.21800E-09, -3.20700E-12, 1.92500E-15])
    # asphS = EvenAspheric(85.289, 8, 20, 0,
    #                      [0, -1.473089E-06, 1.381523E-10, 2.077557E-11, -7.423427E-14, 1.589502E-16])

    asphS.SetCumulative(2)

    projectRB = EmitField(0, 10, sampleTargets=asphS.SampleFromSD())

    intersections = asphS.Intersection(projectRB)
    normals = asphS.Normal(intersections[0])

    asphS.DrawSurface(drawProxy=False)

    DrawNormal(intersections[0], normals)
    DrawPoints(intersections[0])
    plt.show()


def StereoImageTest():
    targets = bd.array([
        [1, 2, 25],
        [2, 4, 25],
        [-2, 3, 25],
        [1, -2, 25]
    ])

    img = Image2DVariDepth()
    img.imageDimensionOverride = 200
    img.nearClipping = 200
    img.zDepthMappingRange = bd.array([500, 1000])

    img.LoadFrom8bit(r"resources/DualTest_RGB.png", r"resources/DualTest_Z.png")

    img.DrawImage()

    # RemoveBG()
    # SetUnifScale(1000)
    # plt.show()

    lens = Biotar50mmf14()
    imager = StdImager(lens.BestFocusBFD(50000))
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty()

    mainRB = img.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(512), 1024)
    mainRB, mainRP, reflectedRB = lens.Propagate(mainRB, reflection=False)
    mainRB, _tir, _vig = imager.IntersectRays(mainRB)
    # mainRP.Append(mainRB, _tir, _vig)

    image = imager.IntegralRays(mainRB, baseImg=image, polarized=False)


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


def LinalgTest():
    A = bd.array([[[ 0., -0.03225487],
                  [ 1., -0.99947968]],
                 [[ 0., -0.06186982],
                  [ 1., -0.99808423]],
                 [[ 0., -0.08692668],
                  [ 1., -0.99621471]],
                 [[ 0., -0.10661849],
                  [ 1., -0.9943    ]],
                 [[ 0., -0.12116925],
                  [ 1., -0.99263186]],
                 [[ 0., -0.13142925],
                  [ 1., 0.99132555]],
                 [[ 0., -0.13843404],
                  [ 1., -0.99037166]],
                 [[ 0., -0.14312024],
                  [ 1., -0.98970531]],
                 [[ 0., -0.14621826],
                  [ 1., -0.98925235]],
                 [[ 0.,  -0.14825282],
                  [ 1.,  -0.98894949]],
                 [[ 0.,  -0.1495842 ],
                  [ 1.,  -0.98874899]],
                 [[ 0.,  -0.15045374],
                  [ 1.,  -0.98861705]],
                 [[ 0.,  -0.15102104],
                  [ 1.,  -0.98853055]],
                 [[ 0.,  -0.15139092],
                  [ 1.,  -0.98847397]]])

    B = bd.array([
                 [2.83727364e-01,5.00018818e+05],
                 [5.33810970e-01,5.00018705e+05],
                 [7.29764363e-01,5.00018551e+05],
                 [8.68509736e-01,5.00018389e+05],
                 [9.59310383e-01,5.00018243e+05],
                 [1.01566953e+00,5.00018124e+05],
                 [1.04968819e+00,5.00018033e+05],
                 [1.07007025e+00,5.00017968e+05],
                 [1.08235910e+00,5.00017922e+05],
                 [1.08986636e+00,5.00017891e+05],
                 [1.09452072e+00,5.00017870e+05],
                 [1.09744508e+00,5.00017856e+05],
                 [1.09930223e+00,5.00017847e+05],
                 [1.10049110e+00,5.00017841e+05]])

    print("A.shape", A.shape)  # should be (N, 2, 2)
    print("B.shape", B.shape)  # must be (N, 2)

    t = bd.linalg.solve(A, B)

def main():
    LinalgTest()


if __name__ == "__main__":
    main()

