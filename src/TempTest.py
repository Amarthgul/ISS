import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import RectPath
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm
from Util.Diffraction import Diffraction, SensorSpec
from Util.Misc import NumpyConversion

def BladeTest():
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8

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
    for i in range(25):
        #rot.Reset()
        # rot.DuplicateAroundCenter()
        rot.RotateAllBlades(-1.5)

        rgb = rot.toImage()
        plt.imshow(rgb)

        plt.draw()
        plt.pause(0.5)

    plt.show()


def DifTest():
    from ObjectSpace.Images import Image2DFlat
    from Util.ImageIO import ImageConversion

    image = Image2DFlat()
    image.LoadFromEXR(RectPath(r"resources/Focus8000.exr"))
    # image.LoadFrom8bit(RectPath(r"resources/ISO12233.jpg"))

    #plt.imshow(image.Show2D(show=False))


    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8
    rot.randOffsetMax = 1.5
    rot.Reset()
    rot.DuplicateAroundCenter()
    rot.RotateAllBlades(-35)
    rgb = rot.toImage()

    sensor = SensorSpec(36, 24, 1920, 1280)

    dif = Diffraction(rgb[:, :, 0], sensor)

    # psfs = dif.PSF([0, 5, 10])
    difImage = dif.ApplyDiffraction(image.Show2D(show=False), 0.8, 0)

    plt.imshow(ImageConversion(difImage, flipH=True, flipV=True))

    # plt.imshow(NumpyConversion(psfs[2]*32000))

    plt.show()


def main():
    DifTest()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

