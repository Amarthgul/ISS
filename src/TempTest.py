import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import RectPath
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm
from Util.ImageIO import rgbFromRGBA



def BladeTest():
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragm.svg"))

    for i in range(25):
        rot.Reset()
        rot.DuplicateAroundCenter()
        rot.RotateAllBlades(-i)
        arr = rot.toArray()

        rgb = rgbFromRGBA(bd.asnumpy(arr))
        plt.imshow(rgb)

        plt.draw()
        plt.pause(1)

    plt.show()


def BladeTune():
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragm.svg"))
    rot.StopDownToRatio(0.25)

    arr = rot.toArray()

    rgb = rgbFromRGBA(bd.asnumpy(arr))
    print(rot.CalculateRatio())

    plt.imshow(rgb)
    plt.show()


def main():
    BladeTune()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

