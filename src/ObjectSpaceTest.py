


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
    DoubleImgTest()


if __name__ == "__main__":
    main()

