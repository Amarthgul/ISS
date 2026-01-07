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
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8
    rot.randOffsetMax = 1.5
    rot.Reset()
    rot.DuplicateAroundCenter()
    rot.RotateAllBlades(-35)
    rgb = rot.toImage()

    sensor = SensorSpec(36, 24, 6000, 4000)

    dif = Diffraction(rgb[:, :, 0], sensor)
    psfs = dif.PSF([0, 5, 10])

    plt.imshow(NumpyConversion(psfs[2]*32000))

    plt.show()


def main():
    DifTest()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

