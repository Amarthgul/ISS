import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import RectPath
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm
from Util.ImageIO import rgbFromRGBA



def BladeTest():
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragmL.svg"))
    rot.bladeCount = 8

    for i in range(25):
        rot.Reset()
        rot.DuplicateAroundCenter()
        rot.RotateAllBlades(-i)

        rgb = rot.toImage()
        plt.imshow(rgb)

        plt.draw()
        plt.pause(0.5)

    plt.show()


def BladeTune():
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragm.svg"))
    rot.StopDownToRatio(0.25)

    print(rot.CalculateRatio())

    plt.imshow(rot.toImage())
    plt.show()


def main():
    BladeTest()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

