

from Util.Backend import backend as bd
from Util.Misc import RectPath
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm
from Util.Diffraction import Diffraction, SensorSpec
from Util.Misc import NumpyConversion

def BladeTest():
    import matplotlib.pyplot as plt

    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8
    rot.randOffsetMax = 1.5

    # for i in range(25):
    #     rot.Reset()
    #     rot.DuplicateAroundCenter()
    #     rot.RotateAllBlades(-i)
    #
    #     rgb = rot.toImage()
    #     plt.imshow(rgb)
    #
    #     plt.draw()
    #     plt.pause(0.5)

    rot.DuplicateAroundCenter()
    for i in range(40):
        #rot.Reset()
        # rot.DuplicateAroundCenter()
        rot.RotateAllBlades(-1)

        rgb = rot.toImage()
        plt.imshow(rgb)

        plt.draw()
        plt.pause(0.5)

    plt.show()


def DifTest():
    from ObjectSpace.Images import Image2DFlat
    from Util.ImageIO import ImageConversion, CleanDisplay
    import matplotlib.pyplot as plt

    image = Image2DFlat()
    image.LoadFromEXR(RectPath(r"resources/CanonEFLSpotTest.exr"))
    # image.LoadFrom8bit(RectPath(r"resources/ISO12233.jpg"))

    # plt.imshow(image.Show2D(show=False))


    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8
    rot.randOffsetMax = 1.5
    rot.Reset()
    rot.DuplicateAroundCenter()
    rot.RotateAllBlades(-40)
    rgb = rot.toImage()
    CleanDisplay(rgb)

    sensor = SensorSpec(36, 24, 1920, 1280)

    dif = Diffraction(rgb[:, :, 0], sensor)

    psfs = dif.PSF([0, 5, 10])

    #CleanDisplay(NumpyConversion(psfs[2]*100000))

    #difImage = dif.ApplyDiffraction(image.Show2D(show=False), 0.1, 0)

    #plt.imshow(ImageConversion(difImage, flipH=True, flipV=True, maxModifier=0.00001))



    plt.show()


def CentroidTest():
    from Util.Centroid import Centroid
    from ObjectSpace.Images import Image2DFlat
    from Util.ImageIO import ImageConversion, CleanDisplay
    import matplotlib.pyplot as plt

    image = Image2DFlat()
    image.LoadFromEXR(RectPath(r"resources/CanonEFLSpotTest.exr"))
    # image.LoadFrom8bit(RectPath(r"resources/ISO12233.jpg"))

    # plt.imshow(image.Show2D(show=False))

    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8
    rot.randOffsetMax = 1.5
    rot.Reset()
    rot.DuplicateAroundCenter()
    rot.RotateAllBlades(-40)
    rgb = rot.toImage()
    CleanDisplay(rgb)

    cen = Centroid(rgb)
    print(cen.centroidAngle)

    plt.show()


def DistTest():
    from Analysis import Analysis
    from ZmxReader import LensFromZmx
    from Imagers.Standard import StdImager

    reader = LensFromZmx(RectPath(r"resources/Zmx/CanonFD50f1.8.zmx"))
    lens = reader.GetLens()
    lens.UpdateLens()

    imager = StdImager(lens.BestFocusBFD(200000), horiPx=6000)
    imager.SetLensLength(lens.totalAxialLength)

    ansys = Analysis(lens, imager)
    sth = ansys.Distortion(200000, samplePoints=7)
    print(ansys.distortionData)
    ansys.PlotDistortionPercentage()


def main():
    # DifTest()
    # CentroidTest()
    # StereoImageDisplay()
    DistTest()


if __name__ == "__main__":
    main()

