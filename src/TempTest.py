import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import RectPath
from Util.DiaphragmSVG import SingleEndPinnedDiaphragm
from Util.ImageIO import rgbFromRGBA



def BladeTest():
    rot = SingleEndPinnedDiaphragm(RectPath(r"resources/diaphragm.svg"))
    rot.DuplicateAroundCenter(10, 32.72)
    rot.RotateAllBlades(-25)
    arr = rot.toArray()

    rgb = rgbFromRGBA(bd.asnumpy(arr))
    plt.imshow(rgb)
    # plt.imshow(rgb)
    # plt.imshow(bd.asnumpy(arr))

    plt.show()


def main():
    BladeTest()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

