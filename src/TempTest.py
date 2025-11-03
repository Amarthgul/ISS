import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import RectPath
from Util.DiaphragmSVG import DiaphragmBlades
from Util.ImageIO import rgbFromRGBA

def BladeTest():
    rot = DiaphragmBlades(RectPath(r"resources/diaphragm.svg"))
    rot.DuplicateAroundCenter(5, 60)
    rot.RotateAllBlades(-25)
    arr = rot.toArray()
    plt.imshow(bd.asnumpy(arr))
    plt.show()


def main():
    BladeTest()
    # StereoImageDisplay()


if __name__ == "__main__":
    main()

