


from PIL import Image
import time
import matplotlib.pyplot as plt
import OpenEXR

from Util.Backend import backend as bd
from Util.ImageIO import ImageConversion, ImageConversionAverage, SaveAsEXR
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints, RemoveBG, DrawAspherical, DrawAsphericalProfile
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


def PlotTest():
    SetUnifScale()
    AddXYZ(70)

    DrawAspherical(radius=50.0,
                   k=-1.0,
                   A=[1e-5, -2e-7],
                   clearSemiDiameter=15.0,
                   cumulativeThickness=0,
                   opacity=0.2)

    DrawAsphericalProfile(
        radius=50.0,
        k=-1.0,
        A=[1e-5, -2e-7],
        clearSemiDiameter=15.0,
        cumulativeThickness=0,
        axis="x",  # or "y"
        lineWidth=1.5
    )

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



def main():
    PlotTest()


if __name__ == "__main__":
    main()

